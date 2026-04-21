"""
RAGAS-style Faithfulness & Answer Relevancy evaluator.

Faithfulness:
  1) Tách answer thành atomic statements.
  2) Kiểm tra từng statement có được context hỗ trợ không.
  3) Faithfulness = verified / total_statements.

Answer Relevancy:
  Sinh lại câu hỏi từ answer, so sánh với question gốc bằng Jaccard token overlap.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, NotFoundError

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

    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Không parse được JSON từ output: {raw[:200]}")


def _jaccard(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class RAGASEvaluator:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: int = 3,
    ):
        load_dotenv()
        self.model = model or os.getenv("RAGAS_MODEL") or os.getenv("LLM_MODEL_1", "openai/gpt-oss-20b")
        self.top_k = top_k
        self.retrieval = RetrievalEvaluator()
        self.provider_name, self.client, self.base_url = self._build_client(
            api_key=api_key,
            base_url=base_url,
        )

    @staticmethod
    def _looks_like_openai_model(model: str) -> bool:
        name = (model or "").strip().lower()
        if not name:
            return False
        if name.startswith("openai/"):
            return False
        return name.startswith(("gpt-", "o1", "o3", "o4", "o5", "text-embedding-", "whisper-"))

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        normalized = (url or "").strip()
        if not normalized:
            return normalized
        if "/v1" not in normalized:
            normalized = normalized.rstrip("/") + "/v1"
        return normalized.rstrip("/") + "/"

    def _build_client(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Tuple[str, AsyncOpenAI, str]:
        if api_key:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            return "custom", AsyncOpenAI(**kwargs), base_url or "default"

        nvidia_key = os.getenv("NVIDIA_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        nvidia_base = self._normalize_base_url(
            os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/")
        )
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip()

        if self._looks_like_openai_model(self.model) and openai_key:
            kwargs = {"api_key": openai_key}
            if openai_base:
                kwargs["base_url"] = openai_base
            return "openai", AsyncOpenAI(**kwargs), openai_base or "https://api.openai.com/v1/"

        if nvidia_key:
            return "nvidia", AsyncOpenAI(api_key=nvidia_key, base_url=nvidia_base), nvidia_base

        if openai_key:
            kwargs = {"api_key": openai_key}
            if openai_base:
                kwargs["base_url"] = openai_base
            return "openai", AsyncOpenAI(**kwargs), openai_base or "https://api.openai.com/v1/"

        raise EnvironmentError("Thiếu NVIDIA_API_KEY hoặc OPENAI_API_KEY để chạy RAGASEvaluator.")

    async def _chat_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except NotFoundError as error:
            raise RuntimeError(
                f"RAGAS model/endpoint không tồn tại (404). model={self.model}, "
                f"provider={self.provider_name}, base_url={self.base_url}. "
                "Hãy kiểm tra RAGAS_MODEL và API provider tương ứng."
            ) from error
        return (response.choices[0].message.content or "").strip()

    async def _extract_statements(self, answer: str) -> List[str]:
        prompt = (
            "Hãy tách câu trả lời sau thành các câu phát biểu độc lập (atomic statements).\n"
            "Mỗi câu phát biểu chứa đúng 1 thông tin có thể kiểm chứng.\n"
            "Trả về DUY NHẤT JSON array of strings.\n\n"
            f"Câu trả lời:\n{answer}"
        )
        raw = await self._chat_json(
            "Bạn là chuyên gia phân tích ngôn ngữ.",
            prompt,
        )
        try:
            parsed = _extract_json(raw)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except ValueError:
            pass
        return [answer] if answer.strip() else []

    async def _verify_statements(self, statements: List[str], contexts: List[str]) -> List[bool]:
        if not statements:
            return []
        context_text = "\n---\n".join(contexts) if contexts else "(không có context)"
        statements_json = json.dumps(statements, ensure_ascii=False)
        prompt = (
            "Dựa vào CONTEXT bên dưới, kiểm tra từng statement:\n"
            "- true nếu context hỗ trợ\n"
            "- false nếu context không hỗ trợ hoặc mâu thuẫn\n"
            "Trả về DUY NHẤT JSON array booleans, đúng thứ tự statements.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"STATEMENTS:\n{statements_json}"
        )
        raw = await self._chat_json(
            "Bạn là chuyên gia kiểm chứng thông tin.",
            prompt,
        )
        try:
            parsed = _extract_json(raw)
            if isinstance(parsed, list) and len(parsed) == len(statements):
                return [bool(v) for v in parsed]
        except ValueError:
            pass
        return [False] * len(statements)

    async def compute_faithfulness(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
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
            "statements": [{"text": s, "entailed": v} for s, v in zip(statements, verdicts)],
        }

    async def compute_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        if not answer.strip():
            return {"score": 0.0, "generated_question": ""}
        prompt = (
            "Dựa vào câu trả lời bên dưới, sinh lại câu hỏi phù hợp nhất.\n"
            "Chỉ trả về câu hỏi, không giải thích.\n\n"
            f"Câu trả lời:\n{answer}"
        )
        generated_q = await self._chat_json(
            "Bạn là chuyên gia phân tích QA.",
            prompt,
        )
        return {
            "score": round(_jaccard(question, generated_q), 4),
            "generated_question": generated_q,
        }

    async def score(self, case: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(resp.get("answer", "")) if isinstance(resp, dict) else ""

        raw_contexts = resp.get("contexts", []) if isinstance(resp, dict) else []
        if not isinstance(raw_contexts, list):
            raw_contexts = []
        contexts = [str(c) for c in raw_contexts if str(c).strip()]

        question = str(case.get("question", ""))

        metadata = resp.get("metadata", {}) if isinstance(resp, dict) else {}
        if isinstance(metadata, dict):
            retrieved_ids = [str(x).strip() for x in metadata.get("retrieved_ids", []) if str(x).strip()]
            if not retrieved_ids:
                retrieved_ids = [str(x).strip() for x in metadata.get("sources", []) if str(x).strip()]
        else:
            retrieved_ids = []

        expected_ids = case.get("expected_retrieval_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            expected_ids = case.get("ground_truth_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            case_meta = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
            expected_ids = case_meta.get("expected_retrieval_ids") or []
        if not isinstance(expected_ids, list):
            expected_ids = []

        hit_rate = self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
        mrr = self.retrieval.calculate_mrr(expected_ids, retrieved_ids)

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
