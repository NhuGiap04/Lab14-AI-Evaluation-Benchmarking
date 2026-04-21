import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, NotFoundError
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """Bạn là một AI Assistant chuyên hỗ trợ nội bộ doanh nghiệp.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA TRÊN ngữ cảnh (context) được cung cấp.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên thông tin có trong context. Không được bịa đặt.
- Nếu context không đủ thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
- Trả lời ngắn gọn, chính xác, chuyên nghiệp.
- Không thêm thông tin ngoài context."""


class MainAgent:
    """
    Agent v1 sử dụng LLM thật (NVIDIA NIM / OpenAI-compatible API).
    - Retrieval: lexical overlap để tìm context liên quan từ golden_set.
    - Generation: gọi AGENT_MODEL để sinh câu trả lời từ context đã retrieve.
    """

    def __init__(self, dataset_path: str = "data/golden_set.jsonl", top_k: int = 3):
        self.name = "SupportAgent-v1"
        self.version = "v1.0.0"
        self.top_k = top_k
        self.dataset_path = dataset_path
        self.model = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")
        self.provider_name, self.client, self.client_base_url = self._build_primary_client(self.model)
        self.fallback_provider_name, self.fallback_client, self.fallback_base_url = self._build_fallback_client(self.model)
        self.knowledge_base = self._load_knowledge_base()

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

    def _build_primary_client(self, model: str) -> Tuple[str, AsyncOpenAI, str]:
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        nvidia_base = self._normalize_base_url(
            os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/")
        )
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip()

        if self._looks_like_openai_model(model) and openai_key:
            kwargs: Dict[str, Any] = {"api_key": openai_key}
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

        raise EnvironmentError(
            "Thiếu API key để chạy MainAgent. Cần NVIDIA_API_KEY hoặc OPENAI_API_KEY."
        )

    def _build_fallback_client(self, model: str) -> Tuple[Optional[str], Optional[AsyncOpenAI], Optional[str]]:
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        nvidia_base = self._normalize_base_url(
            os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/")
        )
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip()

        if self.provider_name == "nvidia" and self._looks_like_openai_model(model) and openai_key:
            kwargs: Dict[str, Any] = {"api_key": openai_key}
            if openai_base:
                kwargs["base_url"] = openai_base
            return "openai", AsyncOpenAI(**kwargs), openai_base or "https://api.openai.com/v1/"

        if self.provider_name == "openai" and model.strip().lower().startswith("openai/") and nvidia_key:
            return "nvidia", AsyncOpenAI(api_key=nvidia_key, base_url=nvidia_base), nvidia_base

        return None, None, None

    def _load_knowledge_base(self) -> List[Dict]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset không tồn tại: {self.dataset_path}. "
                "Hãy chạy `python data/synthetic_gen.py` trước."
            )
        knowledge_base: List[Dict] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                context = item.get("context", "").strip()
                expected_answer = item.get("expected_answer", "").strip()
                question = item.get("question", "").strip()
                metadata = item.get("metadata", {})
                doc_id = (
                    item.get("expected_retrieval_ids", [None])[0]
                    or item.get("ground_truth_ids", [None])[0]
                    or metadata.get("source_doc")
                    or "unknown_doc"
                )
                if context and expected_answer:
                    knowledge_base.append({
                        "doc_id": doc_id,
                        "source": metadata.get("source_doc", "golden_set"),
                        "question": question,
                        "context": context,
                        "expected_answer": expected_answer,
                        "type": metadata.get("type", "unknown"),
                    })
        if not knowledge_base:
            raise ValueError(f"Dataset {self.dataset_path} không có bản ghi hợp lệ.")
        return knowledge_base

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
        return [t for t in cleaned.split() if len(t) > 1]

    def _score(self, question: str, doc_text: str) -> float:
        q_tokens = set(self._tokenize(question))
        d_tokens = set(self._tokenize(doc_text))
        if not q_tokens:
            return 0.0
        return len(q_tokens & d_tokens) / len(q_tokens)

    def _retrieve(self, question: str) -> List[Tuple[Dict, float]]:
        ranked = [
            (doc, self._score(question, f"{doc['question']} {doc['context']}"))
            for doc in self.knowledge_base
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = [item for item in ranked if item[1] > 0][: self.top_k]
        return top if top else ranked[:1]

    async def _call_llm(self, question: str, contexts: List[str]) -> Tuple[str, Dict]:
        """Gọi LLM với context đã retrieve, trả về (answer, usage)."""
        context_block = "\n\n---\n\n".join(
            f"[Tài liệu {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )
        user_message = f"Ngữ cảnh:\n{context_block}\n\nCâu hỏi: {question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        async def _single_call(client: AsyncOpenAI, provider_name: str, base_url: str) -> Tuple[str, Dict]:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            answer = (response.choices[0].message.content or "").strip()
            usage_obj = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0)
            return answer, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider": provider_name,
                "base_url": base_url,
            }

        try:
            return await _single_call(self.client, self.provider_name, self.client_base_url)
        except NotFoundError as primary_error:
            if self.fallback_client is not None and self.fallback_provider_name:
                answer, usage = await _single_call(
                    self.fallback_client,
                    self.fallback_provider_name,
                    self.fallback_base_url or "",
                )
                usage["fallback_used"] = True
                usage["primary_provider_failed"] = self.provider_name
                usage["primary_error"] = str(primary_error)
                return answer, usage
            raise RuntimeError(
                f"Model/endpoint không tồn tại (404). model={self.model}, "
                f"provider={self.provider_name}, base_url={self.client_base_url}. "
                "Kiểm tra lại AGENT_MODEL và API provider tương ứng."
            ) from primary_error

    async def query(self, question: str) -> Dict:
        retrieved = self._retrieve(question)
        contexts = [doc["context"] for doc, _ in retrieved]

        answer, usage = await self._call_llm(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
            "metadata": {
                "agent_version": self.version,
                "model": self.model,
                "retrieved_ids": [doc["doc_id"] for doc, _ in retrieved],
                "retrieval_scores": [round(score, 4) for _, score in retrieved],
                "sources": [doc["source"] for doc, _ in retrieved],
                "dataset_path": self.dataset_path,
                "retrieval_method": "lexical_overlap",
                "provider": self.provider_name,
                "token_usage": usage,
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(json.dumps(resp, ensure_ascii=False, indent=2))

    asyncio.run(test())
