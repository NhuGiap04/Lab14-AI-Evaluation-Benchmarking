import asyncio
import json
import os
import re
from typing import Dict, List, Tuple

from openai import AsyncOpenAI
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
        self.client = AsyncOpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/"),
        )
        self.knowledge_base = self._load_knowledge_base()

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

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return answer, usage

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
                "token_usage": usage,
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(json.dumps(resp, ensure_ascii=False, indent=2))

    asyncio.run(test())
