import asyncio
import time
from typing import Any, Dict, List
# Import other components...

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        question = str(test_case.get("question", "")).strip()
        if not question:
            raise ValueError("Test case thiếu trường 'question'.")

        expected_answer = str(test_case.get("expected_answer", "")).strip()
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response: Dict[str, Any] = await self.agent.query(question)
        latency = time.perf_counter() - start_time
        agent_answer = str(response.get("answer", "")).strip()
        
        # 2. Chạy RAGAS metrics
        ragas_scores = await self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            question,
            agent_answer,
            expected_answer,
        )
        
        return {
            "test_case": question,
            "expected_answer": expected_answer,
            "agent_response": agent_answer,
            "agent_contexts": response.get("contexts", []),
            "agent_metadata": response.get("metadata", {}),
            # Backward-compatible alias cho code cũ
            "agent_response_meta": response.get("metadata", {}),
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if float(judge_result.get("final_score", 0)) < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
