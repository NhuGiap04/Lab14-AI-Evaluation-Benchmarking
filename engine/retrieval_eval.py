from typing import Any, Dict, List

class RetrievalEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def _as_str_list(values: Any) -> List[str]:
        """Chuẩn hóa các kiểu dữ liệu đầu vào thành list[str] không rỗng."""
        if values is None:
            return []

        if isinstance(values, (list, tuple, set)):
            items = values
        else:
            items = [values]

        result: List[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    def _extract_expected_ids(self, case: Dict[str, Any]) -> List[str]:
        """Lấy expected_retrieval_ids từ nhiều vị trí phổ biến trong test case."""
        top_level = self._as_str_list(case.get("expected_retrieval_ids"))
        if top_level:
            return top_level

        metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
        from_metadata = self._as_str_list(metadata.get("expected_retrieval_ids"))
        if from_metadata:
            return from_metadata

        source_doc_id = metadata.get("source_doc_id")
        return self._as_str_list(source_doc_id)

    def _extract_retrieved_ids(self, case: Dict[str, Any]) -> List[str]:
        """Lấy retrieved_ids từ output retrieval/agent response ở nhiều format."""
        top_level = self._as_str_list(case.get("retrieved_ids"))
        if top_level:
            return top_level

        retrieval = case.get("retrieval", {}) if isinstance(case.get("retrieval"), dict) else {}
        from_retrieval = self._as_str_list(retrieval.get("retrieved_ids"))
        if from_retrieval:
            return from_retrieval

        agent_response = case.get("agent_response", {}) if isinstance(case.get("agent_response"), dict) else {}
        metadata = (
            agent_response.get("metadata", {})
            if isinstance(agent_response.get("metadata"), dict)
            else {}
        )
        from_sources = self._as_str_list(metadata.get("sources"))
        if from_sources:
            return from_sources

        return []

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Hit@K: có ít nhất 1 expected_id trong top_k retrieved_ids thì tính 1.0, ngược lại 0.0.
        """
        if top_k <= 0:
            return 0.0
        if not expected_ids or not retrieved_ids:
            return 0.0

        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        MRR cho một query.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        """
        Chạy retrieval eval cho toàn bộ dataset và trả về thống kê tổng hợp.

        Hỗ trợ nhiều format input:
        - expected IDs: expected_retrieval_ids (top-level) hoặc metadata.expected_retrieval_ids
        - retrieved IDs: retrieved_ids (top-level) hoặc retrieval.retrieved_ids hoặc agent_response.metadata.sources
        """
        if not dataset:
            return {
                "avg_hit_rate": 0.0,
                "avg_mrr": 0.0,
                "total_cases": 0,
                "evaluated_cases": 0,
                "skipped_cases": 0,
                "top_k": top_k,
                "per_case": [],
            }

        per_case: List[Dict[str, Any]] = []
        hit_sum = 0.0
        mrr_sum = 0.0
        evaluated_cases = 0

        for idx, case in enumerate(dataset):
            expected_ids = self._extract_expected_ids(case)
            retrieved_ids = self._extract_retrieved_ids(case)

            if not expected_ids:
                per_case.append(
                    {
                        "index": idx,
                        "question": case.get("question", ""),
                        "status": "skipped",
                        "reason": "missing_expected_ids",
                    }
                )
                continue

            hit = self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)

            hit_sum += hit
            mrr_sum += mrr
            evaluated_cases += 1

            per_case.append(
                {
                    "index": idx,
                    "question": case.get("question", ""),
                    "expected_ids": expected_ids,
                    "retrieved_ids": retrieved_ids,
                    "hit_rate": hit,
                    "mrr": mrr,
                    "status": "ok",
                }
            )

        skipped_cases = len(dataset) - evaluated_cases
        avg_hit_rate = hit_sum / evaluated_cases if evaluated_cases else 0.0
        avg_mrr = mrr_sum / evaluated_cases if evaluated_cases else 0.0

        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "total_cases": len(dataset),
            "evaluated_cases": evaluated_cases,
            "skipped_cases": skipped_cases,
            "top_k": top_k,
            "per_case": per_case,
        }
