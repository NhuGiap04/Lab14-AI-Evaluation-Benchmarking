import asyncio
import json
import os
import time
from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent

# Giả lập các components Expert
class ExpertEvaluator:
    def __init__(self, top_k: int = 3):
        self.retrieval = RetrievalEvaluator()
        self.top_k = top_k

    async def score(self, case, resp): 
        retrieved_ids = []
        metadata = resp.get("metadata", {}) if isinstance(resp, dict) else {}
        if isinstance(metadata, dict):
            sources = metadata.get("sources")
            if isinstance(sources, list):
                retrieved_ids = [str(x).strip() for x in sources if str(x).strip()]

        expected_ids = case.get("expected_retrieval_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            case_meta = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
            expected_ids = case_meta.get("expected_retrieval_ids")
        if not isinstance(expected_ids, list) or not expected_ids:
            expected_ids = []

        hit_rate = self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, top_k=self.top_k)
        mrr = self.retrieval.calculate_mrr(expected_ids, retrieved_ids)

        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top_k": self.top_k,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            }
        }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    load_dotenv()
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    judge = LLMJudge()
    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(top_k=top_k), judge)
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
