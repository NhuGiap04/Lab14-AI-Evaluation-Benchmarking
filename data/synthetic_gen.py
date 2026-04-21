import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


DOC_DIR_CANDIDATES = ("dataset/docs", "data/docs")
HARD_CASE_GUIDE_CANDIDATES = ("hard_cases_guide.md", "data/HARD_CASES_GUIDE.md")


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[synthetic_gen {timestamp}] {message}", flush=True)


def _extract_json_array(raw_text: str) -> List[Dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output does not contain a valid JSON array.")
    data = json.loads(match.group(1))
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not an array.")
    return data


def _find_existing_path(candidates: Sequence[str]) -> Optional[Path]:
    for raw in candidates:
        p = Path(raw)
        if p.exists():
            return p
    return None


def _discover_doc_files() -> List[Path]:
    existing_dirs = [Path(d) for d in DOC_DIR_CANDIDATES if Path(d).exists()]
    if not existing_dirs:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục docs. Đã thử: {', '.join(DOC_DIR_CANDIDATES)}"
        )
    files: List[Path] = []
    for d in existing_dirs:
        files.extend(sorted(d.glob("*.txt")))
    if not files:
        raise FileNotFoundError("Không tìm thấy file .txt nào trong thư mục docs.")
    return files


def _load_hard_case_guide() -> str:
    guide_path = _find_existing_path(HARD_CASE_GUIDE_CANDIDATES)
    if not guide_path:
        return ""

    return guide_path.read_text(encoding="utf-8")


def _normalize_case(
    case: Dict,
    fallback_doc_id: str,
    default_type: str = "fact-check",
) -> Dict:
    question = str(case.get("question", "")).strip()
    expected_answer = str(case.get("expected_answer", "")).strip()
    context = str(case.get("context", "")).strip()
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    doc_ids = case.get("expected_retrieval_ids") or case.get("ground_truth_ids")
    if not isinstance(doc_ids, list) or not doc_ids:
        doc_ids = [fallback_doc_id]

    metadata.setdefault("difficulty", "medium")
    metadata.setdefault("type", default_type)
    metadata.setdefault("source_doc", fallback_doc_id)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "context": context,
        "expected_retrieval_ids": doc_ids,
        "ground_truth_ids": doc_ids,
        "metadata": metadata,
    }


async def generate_qa_from_text(
    text: str,
    num_pairs: int = 8,
    doc_id: str = "unknown_doc",
    client: Optional[AsyncOpenAI] = None,
    model: Optional[str] = None,
    mode: str = "mixed",
    hard_case_guide: str = "",
) -> List[Dict]:
    """
    Sinh QA từ tài liệu bằng OpenAI API.
    mode:
    - mixed: case thường (easy/medium/hard)
    - hard: case khó theo hard case guide
    """
    if client is None:
        if AsyncOpenAI is None:
            raise ImportError(
                "Thiếu package 'openai'. Chạy: pip install -r requirements.txt"
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Thiếu OPENAI_API_KEY trong environment hoặc .env")

        client = AsyncOpenAI(api_key=api_key)

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = (
        "Bạn là chuyên gia tạo synthetic dataset cho RAG evaluation. "
        "Chỉ trả về JSON array hợp lệ, không markdown."
    )
    if mode == "hard":
        mode_instruction = f"""
Mục tiêu: chỉ sinh HARD TEST CASES.
Ràng buộc bổ sung:
- difficulty phải là "hard" cho mọi sample.
- metadata.type phải thuộc 1 trong các nhóm:
  adversarial_prompt_injection, goal_hijacking, out_of_context, ambiguous_question,
  conflicting_information, multi_turn_carry_over, multi_turn_correction,
  latency_stress, cost_efficiency, edge_case.
- Câu hỏi phải thử thách và tránh trùng nhau.
- expected_answer phải có tính kiểm chứng, không bịa.

HARD CASE GUIDE:
{hard_case_guide[:4000] if hard_case_guide else "(không có file guide, tự suy luận hard cases chuẩn RAG eval)"}
""".strip()
    else:
        mode_instruction = """
Mục tiêu: sinh bộ QA chất lượng cao cho benchmark.
Ràng buộc bổ sung:
- Ưu tiên easy/medium, có thể có hard.
- Đa dạng loại câu hỏi, tránh trùng.
- expected_answer ngắn gọn, kiểm chứng được.
""".strip()

    user_prompt = f"""
Tạo {num_pairs} test cases tiếng Việt từ tài liệu dưới đây.

Ràng buộc bắt buộc:
1) Mỗi phần tử JSON có đủ:
   - question (string)
   - expected_answer (string)
   - context (string, trích đoạn liên quan trong tài liệu)
   - expected_retrieval_ids (array string, luôn chứa "{doc_id}")
   - ground_truth_ids (array string, luôn chứa "{doc_id}")
   - metadata (object) gồm:
       - difficulty: easy | medium | hard
       - type: fact-check | multi-hop | policy-check | adversarial | edge-case
       - source_doc: "{doc_id}"
2) Không được bịa thông tin ngoài tài liệu.
3) Trả về DUY NHẤT JSON array.

{mode_instruction}

Tài liệu:
{text}
""".strip()

    _log(
        f"[{doc_id}] bắt đầu sinh {num_pairs} QA (mode={mode}) bằng model "
        f"{model_name}"
    )
    for attempt in range(3):
        _log(f"[{doc_id}] gọi model, lần thử {attempt + 1}/3")
        response = await client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = _extract_json_array(content)
            normalized = [
                _normalize_case(case, fallback_doc_id=doc_id)
                for case in parsed
                if isinstance(case, dict)
            ]
            normalized = [
                c
                for c in normalized
                if c["question"] and c["expected_answer"] and c["context"]
            ]
            min_required = min(num_pairs, 3)
            if len(normalized) >= min_required:
                _log(f"[{doc_id}] sinh thành công {len(normalized[:num_pairs])} QA")
                return normalized[:num_pairs]
        except Exception:
            if attempt == 2:
                raise
            _log(f"[{doc_id}] parse lỗi output, retry sau backoff")
            await asyncio.sleep(1.2 + attempt)

    raise RuntimeError(f"Không thể parse output từ model cho doc {doc_id}.")


def _allocate_counts(total_cases: int, keys: List[str]) -> Dict[str, int]:
    if total_cases <= 0 or not keys:
        return {k: 0 for k in keys}
    counts = {k: 0 for k in keys}
    for i in range(total_cases):
        counts[keys[i % len(keys)]] += 1
    return counts


async def generate_hard_cases_from_guide(
    docs: Dict[str, str],
    guide_text: str,
    total_cases: int,
    client: AsyncOpenAI,
    model: str,
    concurrency: int = 4,
) -> List[Dict]:
    doc_ids = list(docs.keys())
    allocations = _allocate_counts(total_cases, doc_ids)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _generate_hard_for_doc(
        doc_id: str,
        text: str,
        target: int,
    ) -> tuple[str, List[Dict], float]:
        if target <= 0:
            return doc_id, [], 0.0

        start = time.perf_counter()
        async with semaphore:
            cases = await generate_qa_from_text(
                text=text,
                num_pairs=target,
                doc_id=doc_id,
                client=client,
                model=model,
                mode="hard",
                hard_case_guide=guide_text,
            )
        return doc_id, cases, time.perf_counter() - start

    tasks = [
        asyncio.create_task(_generate_hard_for_doc(doc_id, docs[doc_id], allocations[doc_id]))
        for doc_id in doc_ids
        if allocations[doc_id] > 0
    ]
    results: List[Dict] = []
    done = 0
    total_tasks = len(tasks)
    for task in asyncio.as_completed(tasks):
        doc_id, cases, elapsed = await task
        done += 1
        _log(
            f"tiến độ hard-cases: {done}/{total_tasks} docs | "
            f"vừa xong={doc_id} ({len(cases)} cases, {elapsed:.2f}s)"
        )
        results.extend(cases)
    return results[:total_cases]


async def generate_top_up_cases_with_llm(
    docs: Dict[str, str],
    needed: int,
    client: AsyncOpenAI,
    model: str,
    guide_text: str,
    concurrency: int = 4,
    prefer_hard: bool = False,
) -> List[Dict]:
    if needed <= 0:
        return []

    doc_ids = list(docs.keys())
    allocations = _allocate_counts(needed, doc_ids)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _top_up_for_doc(
        doc_id: str,
        text: str,
        target: int,
    ) -> tuple[str, List[Dict], float]:
        if target <= 0:
            return doc_id, [], 0.0

        start = time.perf_counter()
        async with semaphore:
            cases = await generate_qa_from_text(
                text=text,
                num_pairs=target,
                doc_id=doc_id,
                client=client,
                model=model,
                mode="hard" if prefer_hard else "mixed",
                hard_case_guide=guide_text,
            )
        return doc_id, cases, time.perf_counter() - start

    tasks = [
        asyncio.create_task(_top_up_for_doc(doc_id, docs[doc_id], allocations[doc_id]))
        for doc_id in doc_ids
        if allocations[doc_id] > 0
    ]
    output: List[Dict] = []
    completed = 0
    total_tasks = len(tasks)
    for task in asyncio.as_completed(tasks):
        doc_id, cases, elapsed = await task
        completed += 1
        _log(
            f"tiến độ top-up: {completed}/{total_tasks} docs | "
            f"vừa xong={doc_id} ({len(cases)} cases, {elapsed:.2f}s, prefer_hard={prefer_hard})"
        )
        output.extend(cases)
    return output[:needed]


def _deduplicate_cases(cases: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for case in cases:
        key = " ".join(case.get("question", "").lower().split())
        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(case)

    return deduped


async def main() -> None:
    load_dotenv()
    if AsyncOpenAI is None:
        raise ImportError("Thiếu package 'openai'. Chạy: pip install -r requirements.txt")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu OPENAI_API_KEY. Hãy thêm vào environment hoặc file .env")

    doc_files = _discover_doc_files()
    docs = {doc.stem: doc.read_text(encoding="utf-8") for doc in doc_files}
    guide_text = _load_hard_case_guide()

    target_total = int(os.getenv("TARGET_DATASET_SIZE", "55"))
    per_doc = int(os.getenv("PAIRS_PER_DOC", "9"))
    hard_target = int(os.getenv("HARD_CASES_TARGET", "12"))
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    concurrency = int(os.getenv("GEN_CONCURRENCY", "4"))
    _log(
        "khởi tạo pipeline | "
        f"docs={len(doc_files)}, per_doc={per_doc}, target_total={target_total}, "
        f"hard_target={hard_target}, concurrency={concurrency}, model={model_name}"
    )

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _generate_for_doc(
        doc_id: str,
        text: str,
    ) -> tuple[str, List[Dict], float]:
        start = time.perf_counter()
        async with semaphore:
            qa_pairs = await generate_qa_from_text(
                text=text,
                num_pairs=per_doc,
                doc_id=doc_id,
                client=client,
                model=model_name,
            )
        elapsed = time.perf_counter() - start
        return doc_id, qa_pairs, elapsed

    tasks = [
        asyncio.create_task(_generate_for_doc(doc_id, text))
        for doc_id, text in docs.items()
    ]
    total_docs = len(tasks)
    completed_docs = 0
    generated_by_doc: Dict[str, List[Dict]] = {}
    llm_total = 0
    for task in asyncio.as_completed(tasks):
        doc_id, qa_pairs, elapsed = await task
        generated_by_doc[doc_id] = qa_pairs
        completed_docs += 1
        llm_total += len(qa_pairs)
        _log(
            f"tiến độ LLM: {completed_docs}/{total_docs} docs | "
            f"vừa xong={doc_id} ({len(qa_pairs)} cases, {elapsed:.2f}s) | "
            f"tổng llm_cases={llm_total}"
        )

    generated_per_doc = [generated_by_doc[doc.stem] for doc in doc_files]
    llm_cases = [item for batch in generated_per_doc for item in batch]
    _log(f"hoàn tất sinh từ LLM: {len(llm_cases)} cases")

    hard_cases = await generate_hard_cases_from_guide(
        docs=docs,
        guide_text=guide_text,
        total_cases=hard_target,
        client=client,
        model=model_name,
        concurrency=concurrency,
    )
    _log(f"hard cases sinh từ LLM+guide: {len(hard_cases)}")

    all_cases = _deduplicate_cases(llm_cases + hard_cases)
    _log(f"sau deduplicate: {len(all_cases)} cases")

    for round_idx in range(1, 4):
        if len(all_cases) >= target_total:
            break
        needed = target_total - len(all_cases)
        current_hard = sum(
            1
            for c in all_cases
            if "hard" in str(c.get("metadata", {}).get("difficulty", "")).lower()
        )
        prefer_hard = current_hard < hard_target
        _log(
            f"thiếu {needed} case để đạt target, top-up round {round_idx}/3 "
            f"(prefer_hard={prefer_hard})"
        )
        extra_cases = await generate_top_up_cases_with_llm(
            docs=docs,
            needed=needed,
            client=client,
            model=model_name,
            guide_text=guide_text,
            concurrency=concurrency,
            prefer_hard=prefer_hard,
        )
        all_cases = _deduplicate_cases(all_cases + extra_cases)
        _log(f"sau top-up round {round_idx}: {len(all_cases)} cases")

    if len(all_cases) < target_total:
        raise RuntimeError(
            f"Không đạt được TARGET_DATASET_SIZE={target_total} sau 3 vòng top-up bằng LLM. "
            "Hãy tăng PAIRS_PER_DOC hoặc giảm TARGET_DATASET_SIZE."
        )

    all_cases = all_cases[:target_total]
    output_path = Path("data/golden_set.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        total_cases = len(all_cases)
        for idx, pair in enumerate(all_cases, start=1):
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            if idx % 10 == 0 or idx == total_cases:
                _log(f"đang ghi file: {idx}/{total_cases}")

    hard_count = sum(
        1 for c in all_cases if "hard" in str(c.get("metadata", {}).get("difficulty", "")).lower()
    )
    _log(
        f"Done! Saved {len(all_cases)} cases to {output_path}. "
        f"Hard cases: {hard_count}. Model: {model_name}"
    )


if __name__ == "__main__":
    asyncio.run(main())
