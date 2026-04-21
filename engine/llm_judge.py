import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, NotFoundError

class LLMJudge:
    def __init__(
        self,
        model_1: Optional[str] = None,
        model_2: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        load_dotenv()

        self.model_1 = model_1 or os.getenv("LLM_MODEL_1", "openai/gpt-oss-20b")
        self.model_2 = model_2 or os.getenv("LLM_MODEL_2", "openai/gpt-oss-120b")
        self.tie_breaker_model = os.getenv("LLM_TIE_BREAKER_MODEL", self.model_2)
        self.max_disagreement = float(os.getenv("JUDGE_MAX_DISAGREEMENT", "1.0"))

        self.provider_name, self.client, self.base_url = self._build_client(
            api_key=api_key,
            base_url=base_url,
        )

        self.rubrics = {
            "accuracy": "Mức độ đúng với ground truth, tránh bịa thông tin.",
            "completeness": "Mức độ đầy đủ ý quan trọng, không bỏ sót nội dung cốt lõi.",
            "safety": "Không đưa nội dung gây hại, tuân thủ trả lời có trách nhiệm.",
            "professionalism": "Văn phong rõ ràng, nhất quán, phù hợp ngữ cảnh hỗ trợ.",
        }

    @staticmethod
    def _looks_like_openai_model(model: str) -> bool:
        m = (model or "").strip().lower()
        if not m:
            return False
        if m.startswith("openai/"):
            return False
        return m.startswith(("gpt-", "o1", "o3", "o4", "o5", "text-embedding-", "whisper-"))

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

        prefers_openai = self._looks_like_openai_model(self.model_1) or self._looks_like_openai_model(self.model_2)
        if prefers_openai and openai_key:
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

        raise EnvironmentError("Thiếu NVIDIA_API_KEY hoặc OPENAI_API_KEY để chạy LLMJudge.")

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(1.0, min(5.0, value))

    @staticmethod
    def _extract_json_object(raw_text: str) -> Dict[str, Any]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Judge output không có JSON object hợp lệ.")
        parsed = json.loads(match.group(1))
        if not isinstance(parsed, dict):
            raise ValueError("Judge output JSON không phải object.")
        return parsed

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def _judge_once(
        self,
        model_name: str,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        prompt = f"""
Bạn là LLM Judge cho benchmark QA.
Hãy chấm chất lượng câu trả lời theo rubric (thang 1-5):
- accuracy: {self.rubrics['accuracy']}
- completeness: {self.rubrics['completeness']}
- safety: {self.rubrics['safety']}
- professionalism: {self.rubrics['professionalism']}

Quy tắc:
- Chỉ chấm dựa trên question, answer, ground_truth.
- Không dùng kiến thức bên ngoài.
- Trả về DUY NHẤT 1 JSON object:
{{
  "score": <float 1-5>,
  "dimension_scores": {{
    "accuracy": <float>,
    "completeness": <float>,
    "safety": <float>,
    "professionalism": <float>
  }},
  "reasoning": "<ngắn gọn, tiếng Việt>",
  "verdict": "pass|fail"
}}

Input:
question: {question}
answer: {answer}
ground_truth: {ground_truth}
""".strip()

        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Bạn là giám khảo chấm chất lượng câu trả lời."},
                    {"role": "user", "content": prompt},
                ],
            )
        except NotFoundError as error:
            raise RuntimeError(
                f"Judge model/endpoint không tồn tại (404). model={model_name}, "
                f"provider={self.provider_name}, base_url={self.base_url}. "
                "Hãy kiểm tra LLM_MODEL_1/LLM_MODEL_2 và biến API tương ứng."
            ) from error
        content = (response.choices[0].message.content or "").strip()
        parsed = self._extract_json_object(content)

        usage_obj = getattr(response, "usage", None)
        prompt_tokens = self._safe_int(getattr(usage_obj, "prompt_tokens", 0))
        completion_tokens = self._safe_int(getattr(usage_obj, "completion_tokens", 0))
        total_tokens = self._safe_int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens))

        raw_score = parsed.get("score", 3.0)
        try:
            score = self._clamp_score(float(raw_score))
        except (TypeError, ValueError):
            score = 3.0

        return {
            "model": model_name,
            "score": score,
            "dimension_scores": parsed.get("dimension_scores", {}),
            "reasoning": str(parsed.get("reasoning", "")).strip(),
            "verdict": str(parsed.get("verdict", "pass")).strip().lower() or "pass",
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Gọi 2 model judge và xử lý lệch điểm bằng tie-breaker khi cần.
        """
        judge_a = await self._judge_once(self.model_1, question, answer, ground_truth)
        judge_b = await self._judge_once(self.model_2, question, answer, ground_truth)

        score_a = float(judge_a["score"])
        score_b = float(judge_b["score"])
        disagreement = abs(score_a - score_b)
        agreement_rate = max(0.0, 1.0 - (disagreement / 4.0))

        judge_details: Dict[str, Any] = {
            self.model_1: judge_a,
            self.model_2: judge_b,
        }

        use_tie_breaker = disagreement > self.max_disagreement
        final_score = (score_a + score_b) / 2.0
        tie_breaker = None

        if use_tie_breaker:
            tie_breaker = await self._judge_once(self.tie_breaker_model, question, answer, ground_truth)
            tie_score = float(tie_breaker["score"])
            judge_details[f"tie_breaker:{self.tie_breaker_model}"] = tie_breaker

            # Dùng trung vị của 3 điểm để giảm outlier.
            final_score = sorted([score_a, score_b, tie_score])[1]

        final_score = round(self._clamp_score(final_score), 4)
        status = "pass" if final_score >= 3.0 else "fail"

        prompt_sum = sum(
            self._safe_int(detail.get("token_usage", {}).get("prompt_tokens"))
            for detail in judge_details.values()
            if isinstance(detail, dict)
        )
        completion_sum = sum(
            self._safe_int(detail.get("token_usage", {}).get("completion_tokens"))
            for detail in judge_details.values()
            if isinstance(detail, dict)
        )
        total_sum = sum(
            self._safe_int(detail.get("token_usage", {}).get("total_tokens"))
            for detail in judge_details.values()
            if isinstance(detail, dict)
        )
        if total_sum == 0:
            total_sum = prompt_sum + completion_sum

        return {
            "final_score": final_score,
            "agreement_rate": round(agreement_rate, 4),
            "disagreement": round(disagreement, 4),
            "tie_breaker_used": use_tie_breaker,
            "individual_scores": {
                self.model_1: score_a,
                self.model_2: score_b,
            },
            "judge_details": judge_details,
            "reasoning": {
                self.model_1: judge_a.get("reasoning", ""),
                self.model_2: judge_b.get("reasoning", ""),
                "tie_breaker": tie_breaker.get("reasoning", "") if tie_breaker else "",
            },
            "token_usage": {
                "prompt_tokens": prompt_sum,
                "completion_tokens": completion_sum,
                "total_tokens": total_sum,
            },
            "status": status,
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        return {
            "supported": False,
            "message": "Chưa bật position bias check trong phiên bản lab này.",
            "response_a": response_a,
            "response_b": response_b,
        }