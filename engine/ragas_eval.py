"""
RAGAS-style Faithfulness & Answer Relevancy evaluator.

Faithfulness (theo đúng thuật toán RAGAS):
  Bước 1 — LLM tách answer thành các atomic statements S = {s1, s2, ..., sn}
  Bước 2 — Với mỗi s_i, LLM kiểm tra context có entail không → đếm |V|
  Faithfulness = |V| / |S|

Answer Relevancy:
  LLM sinh lại câu hỏi từ answer, so sánh với question gốc bằng
  token overlap (Jaccard) — không cần embedding, tiết kiệm token.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from engine.retrieval_eval import RetrievalEvaluator


def _extract_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fallback: tìm array hoặc object đầu tiên
    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Không parse được JSON từ output: {raw[:200]}")


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class RAGASEvaluator:
    """
    Tính Faithfulness và Answer Relevancy theo thuật toán RAGAS.
    Dùng cùng NVIDIA_API_KEY / OPENAI_API_KEY như LLMJudge.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        top_k: int = 3,
    ):
        load_dotenv()
        self.model = model or os.getenv("RAGAS_MODEL") or os.getenv("LLM_MODEL_1", "openai/gpt-oss-20b")
        self.top_k = top_k
        self.retrieval = RetrievalEvaluator()

        resolved_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_url = base_url or os.getenv("NVIDIA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not resolved_key:
            raise EnvironmentError("Thiếu NVIDIA_API_KEY hoặc OPENAI_API_KEY để chạy RAGASEvaluator.")

        kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        self.client = AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------
    # Bước 1: tách answer → atomic statements
    # ------------------------------------------------------------------
    async def _extract_statements(self, answer: str) -> List[str]:
        prompt = (
            "Hãy tách câu trả lời sau thành các câu phát biểu độc lập (atomic statements).\n"
            "Mỗi câu phát biểu phải:\n"
            "- Chứa đúng 1 thông tin duy nhất\n"
            "- Có thể kiểm chứng độc lập\n"
            "- Không trùng lặp với câu khác\n\n"
            "Trả về DUY NHẤT JSON array of strings, ví dụ:\n"
            '["Câu 1.", "Câu 2.", "Câu 3."]\n\n'
            f"Câu trả lời:\n{answer}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích ngôn ngữ."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = _extract_json(raw)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except ValueError:
            pass
        # fallback: coi cả answer là 1 statement
        return [answer] if answer.strip() else []

    # ------------------------------------------------------------------
    # Bước 2: kiểm tra từng statement có được context entail không
    # ------------------------------------------------------------------
    async def _verify_statements(self, statements: List[str], contexts: List[str]) -> List[bool]:
        if not statements:
            return []
        context_text = "\n---\n".join(contexts) if contexts else "(không có context)"
        statements_json = json.dumps(statements, ensure_ascii=False)

        prompt = (
            "Dựa vào CONTEXT bên dưới, hãy kiểm tra từng câu phát biểu:\n"
            "- true  → context hỗ trợ / xác minh câu này\n"
            "- false → context không đề cập hoặc mâu thuẫn\n\n"
            "Trả về DUY NHẤT JSON array of boolean, cùng thứ tự với danh sách statements.\n"
            "Ví dụ: [true, false, true]\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"STATEMENTS:\n{statements_json}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia kiểm chứng thông tin."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = _extract_json(raw)
            if isinstance(parsed, list) and len(parsed) == len(statements):
                return [bool(v) for v in parsed]
        except ValueError:
            pass
        # fallback: conservative — coi tất cả là false
        return [False] * len(statements)

    # ------------------------------------------------------------------
    # Faithfulness = |V| / |S|
    # ------------------------------------------------------------------
    async def compute_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> Dict[str, Any]:
        if not answer.strip():
            return {"score": 0.0, "verified": 0, "total": 0, "statements": []}

        statements = await self._extract_statements(answer)
        if not statements:
            return {"score": 0.0, "verified": 0, "total": 0, "statements": []}

        verdicts = await self._verify_statements(statements, contexts)
        verified = sum(1 for v in verdicts if v)
        score = verified / len(statements)

        return {
            "score": round(score, 4),
            "verified": verified,
            "total": len(statements),
            "statements": [
                {"text": s, "entailed": v}
                for s, v in zip(statements, verdicts)
            ],
        }

    # ------------------------------------------------------------------
    # Answer Relevancy — Jaccard giữa question gốc và câu hỏi LLM sinh lại
    # ------------------------------------------------------------------
    async def compute_answer_relevancy(
        self, question: str, answer: str
    ) -> Dict[str, Any]:
        if not answer.strip():
            return {"score": 0.0, "generated_question": ""}

        prompt = (
            "Dựa vào câu trả lời bên dưới, hãy sinh lại câu hỏi mà câu trả lời này đang trả lời.\n"
            "Chỉ trả về câu hỏi, không giải thích thêm.\n\n"
            f"Câu trả lời:\n{answer}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích QA."},
                {"role": "user", "content": prompt},
            ],
        )
        generated_q = (resp.choices[0].message.content or "").strip()
        similarity = _jaccard(question, generated_q)

        return {
            "score": round(similarity, 4),
            "generated_question": generated_q,
        }

    # ------------------------------------------------------------------
    # score() — interface chính, gọi từ ExpertEvaluator
    # ------------------------------------------------------------------
    async def score(self, case: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        answer = resp.get("answer", "") if isinstance(resp, dict) else ""
        contexts: List[str] = resp.get("contexts", []) if isinstance(resp, dict) else []
        if not isinstance(contexts, list):
            contexts = []
        question = case.get("question", "")

        # Retrieval metrics — đọc retrieved_ids từ metadata (ưu tiên) hoặc sources
        metadata = resp.get("metadata", {}) if isinstance(resp, dict) else {}
        if isinstance(metadata, dict):
            retrieved_ids = [str(x).strip() for x in metadata.get("retrieved_ids", []) if str(x).strip()]
            if not retrieved_ids:
                retrieved_ids = [str(x).strip() for x in metadata.get("sources", []) if str(x).strip()]
        else:
            retrieved_ids = []

        expected_ids = case.get("expected_retrieval_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            case_meta = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
            expected_ids = case_meta.get("expected_retrieval_ids") or []

        hit_rate = self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
        mrr = self.retrieval.calculate_mrr(expected_ids, retrieved_ids)

        # RAGAS metrics (chạy song song)
        faithfulness_result, relevancy_result = await asyncio.gather(
            self.compute_faithfulness(answer, contexts),
            self.compute_answer_relevancy(question, answer),
        )

        return {
            "faithfulness": faithfulness_result["score"],
            "faithfulness_detail": faithfulness_result,
            "relevancy": relevancy_result["score"],
            "relevancy_detail": relevancy_result,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top_k": self.top_k,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }



