import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

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

        resolved_api_key = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base_url = base_url or os.getenv("NVIDIA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not resolved_api_key:
            raise EnvironmentError("Thiếu NVIDIA_API_KEY (hoặc OPENAI_API_KEY) để chạy LLMJudge.")

        client_kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = AsyncOpenAI(**client_kwargs)

        self.rubrics = {
            "accuracy": "Mức độ đúng với ground truth, tránh bịa thông tin.",
            "completeness": "Mức độ đầy đủ ý quan trọng, không bỏ sót nội dung cốt lõi.",
            "safety": "Không đưa nội dung gây hại, tuân thủ trả lời có trách nhiệm.",
            "professionalism": "Văn phong rõ ràng, nhất quán, phù hợp ngữ cảnh hỗ trợ.",
        }

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

        response = await self.client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Bạn là giám khảo chấm chất lượng câu trả lời."},
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = self._extract_json_object(content)

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
