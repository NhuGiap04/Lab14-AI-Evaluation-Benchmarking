import asyncio
import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

from agent.main_agent import MainAgent, SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


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
            },
        }


def _require_tiktoken() -> None:
    if tiktoken is None:
        raise ImportError("Thiếu package 'tiktoken'. Hãy cài: pip install tiktoken")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _encoding_for_model(model_name: str):
    _require_tiktoken()
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, model_name: str) -> int:
    if not text:
        return 0
    enc = _encoding_for_model(model_name)
    return len(enc.encode(text))


def _build_agent_user_message(question: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Tài liệu {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return f"Ngữ cảnh:\n{context_block}\n\nCâu hỏi: {question}"


def _estimate_judge_tokens_from_text(
    question: str,
    answer: str,
    ground_truth: str,
    judge_reasoning: Dict[str, Any],
    judge_model_name: str,
    judge_call_count: int,
) -> Dict[str, int]:
    base_prompt = (
        "Bạn là LLM Judge cho benchmark QA. "
        "Chấm theo rubric accuracy/completeness/safety/professionalism. "
        "Input gồm question, answer, ground_truth."
    )
    prompt_text = (
        f"{base_prompt}\nquestion: {question}\nanswer: {answer}\nground_truth: {ground_truth}"
    )
    output_text = json.dumps(judge_reasoning, ensure_ascii=False)

    prompt_tokens_one_call = _count_tokens(prompt_text, judge_model_name)
    completion_tokens_one_call = _count_tokens(output_text, judge_model_name)
    prompt_tokens = prompt_tokens_one_call * judge_call_count
    completion_tokens = completion_tokens_one_call * judge_call_count
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _compute_token_and_cost_metrics(results: List[Dict], default_model: str) -> Dict[str, Any]:
    default_cost_per_1m = float(os.getenv("TOKEN_COST_PER_1M", "1.0"))
    agent_input_cost_per_1m = float(os.getenv("AGENT_INPUT_COST_PER_1M", str(default_cost_per_1m)))
    agent_output_cost_per_1m = float(os.getenv("AGENT_OUTPUT_COST_PER_1M", str(default_cost_per_1m)))
    judge_input_cost_per_1m = float(os.getenv("JUDGE_INPUT_COST_PER_1M", str(default_cost_per_1m)))
    judge_output_cost_per_1m = float(os.getenv("JUDGE_OUTPUT_COST_PER_1M", str(default_cost_per_1m)))

    totals = {
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
    }
    estimated_cases = {"agent": 0, "judge": 0}

    for item in results:
        question = str(item.get("test_case", ""))
        answer = str(item.get("agent_response", ""))
        expected_answer = str(item.get("expected_answer", ""))

        agent_metadata = item.get("agent_metadata", {}) if isinstance(item.get("agent_metadata"), dict) else {}
        agent_model = str(agent_metadata.get("model", default_model))
        agent_usage = agent_metadata.get("token_usage", {}) if isinstance(agent_metadata.get("token_usage"), dict) else {}
        prompt_tokens = _safe_int(agent_usage.get("prompt_tokens"))
        completion_tokens = _safe_int(agent_usage.get("completion_tokens"))

        if prompt_tokens == 0 and completion_tokens == 0:
            contexts = item.get("agent_contexts", [])
            if not isinstance(contexts, list):
                contexts = []
            user_message = _build_agent_user_message(question, [str(c) for c in contexts])
            prompt_tokens = _count_tokens(AGENT_SYSTEM_PROMPT, agent_model) + _count_tokens(user_message, agent_model)
            completion_tokens = _count_tokens(answer, agent_model)
            estimated_cases["agent"] += 1

        totals["agent_input_tokens"] += prompt_tokens
        totals["agent_output_tokens"] += completion_tokens

        judge_info = item.get("judge", {}) if isinstance(item.get("judge"), dict) else {}
        judge_usage = judge_info.get("token_usage", {}) if isinstance(judge_info.get("token_usage"), dict) else {}
        judge_prompt_tokens = _safe_int(judge_usage.get("prompt_tokens"))
        judge_completion_tokens = _safe_int(judge_usage.get("completion_tokens"))

        if judge_prompt_tokens == 0 and judge_completion_tokens == 0:
            individual_scores = judge_info.get("individual_scores", {})
            judge_call_count = len(individual_scores) if isinstance(individual_scores, dict) and individual_scores else 2
            if judge_info.get("tie_breaker_used"):
                judge_call_count += 1

            judge_model_name = default_model
            if isinstance(individual_scores, dict) and individual_scores:
                judge_model_name = str(next(iter(individual_scores.keys())))

            judge_reasoning = judge_info.get("reasoning", {})
            if not isinstance(judge_reasoning, dict):
                judge_reasoning = {"reasoning": str(judge_reasoning)}
            estimated = _estimate_judge_tokens_from_text(
                question=question,
                answer=answer,
                ground_truth=expected_answer,
                judge_reasoning=judge_reasoning,
                judge_model_name=judge_model_name,
                judge_call_count=judge_call_count,
            )
            judge_prompt_tokens = estimated["prompt_tokens"]
            judge_completion_tokens = estimated["completion_tokens"]
            estimated_cases["judge"] += 1

        totals["judge_input_tokens"] += judge_prompt_tokens
        totals["judge_output_tokens"] += judge_completion_tokens

    totals["agent_total_tokens"] = totals["agent_input_tokens"] + totals["agent_output_tokens"]
    totals["judge_total_tokens"] = totals["judge_input_tokens"] + totals["judge_output_tokens"]
    totals["pipeline_total_tokens"] = totals["agent_total_tokens"] + totals["judge_total_tokens"]

    cost = {
        "agent_input_cost": round((totals["agent_input_tokens"] / 1_000_000) * agent_input_cost_per_1m, 6),
        "agent_output_cost": round((totals["agent_output_tokens"] / 1_000_000) * agent_output_cost_per_1m, 6),
        "judge_input_cost": round((totals["judge_input_tokens"] / 1_000_000) * judge_input_cost_per_1m, 6),
        "judge_output_cost": round((totals["judge_output_tokens"] / 1_000_000) * judge_output_cost_per_1m, 6),
    }
    cost["pipeline_total_cost"] = round(
        cost["agent_input_cost"]
        + cost["agent_output_cost"]
        + cost["judge_input_cost"]
        + cost["judge_output_cost"],
        6,
    )

    pricing = {
        "token_cost_per_1m_default": default_cost_per_1m,
        "agent_input_cost_per_1m": agent_input_cost_per_1m,
        "agent_output_cost_per_1m": agent_output_cost_per_1m,
        "judge_input_cost_per_1m": judge_input_cost_per_1m,
        "judge_output_cost_per_1m": judge_output_cost_per_1m,
    }

    return {
        "tokens": totals,
        "cost": cost,
        "pricing": pricing,
        "estimated_by_tiktoken_cases": estimated_cases,
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
    _require_tiktoken()

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    judge = LLMJudge()
    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(top_k=top_k), judge)
    results = await runner.run_all(dataset)

    total = len(results)
    token_cost = _compute_token_and_cost_metrics(
        results=results,
        default_model=os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
    )

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        },
        "token_usage": token_cost["tokens"],
        "cost_estimation": token_cost["cost"],
        "pricing": token_cost["pricing"],
        "estimated_by_tiktoken_cases": token_cost["estimated_by_tiktoken_cases"],
    }
    return results, summary


async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")

    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    print("\n💰 --- TOKEN & COST PIPELINE ---")
    v2_tokens = v2_summary.get("token_usage", {})
    v2_cost = v2_summary.get("cost_estimation", {})
    print(
        "Agent tokens (in/out): "
        f"{v2_tokens.get('agent_input_tokens', 0)} / {v2_tokens.get('agent_output_tokens', 0)}"
    )
    print(
        "Judge tokens (in/out): "
        f"{v2_tokens.get('judge_input_tokens', 0)} / {v2_tokens.get('judge_output_tokens', 0)}"
    )
    print(f"Pipeline total tokens: {v2_tokens.get('pipeline_total_tokens', 0)}")
    print(f"Estimated total cost: {v2_cost.get('pipeline_total_cost', 0)}")

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
