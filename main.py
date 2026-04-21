import asyncio
import json
import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

from agent.main_agent import MainAgent, SYSTEM_PROMPT as AGENT_SYSTEM_PROMPT
from engine.generator_eval import RAGASEvaluator
from engine.llm_judge import LLMJudge
from engine.runner import BenchmarkRunner


def _require_tiktoken() -> None:
    if tiktoken is None:
        raise ImportError("Thiếu package 'tiktoken'. Hãy cài: pip install tiktoken")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool_env(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
        f"[Tài liệu {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
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


def _resolve_agent_pricing_for_model(agent_model: str) -> Dict[str, Any]:
    default_cost_per_1m = float(os.getenv("TOKEN_COST_PER_1M", "1.0"))
    default_agent_input = float(os.getenv("AGENT_INPUT_COST_PER_1M", str(default_cost_per_1m)))
    default_agent_output = float(os.getenv("AGENT_OUTPUT_COST_PER_1M", str(default_cost_per_1m)))

    model_1 = str(os.getenv("AGENT_MODEL_1", "")).strip()
    model_2 = str(os.getenv("AGENT_MODEL_2", "")).strip()
    current = (agent_model or "").strip()

    if current and current == model_1:
        return {
            "agent_input_cost_per_1m": float(os.getenv("AGENT_MODEL_1_INPUT_COST_PER_1M", str(default_agent_input))),
            "agent_output_cost_per_1m": float(os.getenv("AGENT_MODEL_1_OUTPUT_COST_PER_1M", str(default_agent_output))),
            "pricing_source": "model_1_specific",
        }
    if current and current == model_2:
        return {
            "agent_input_cost_per_1m": float(os.getenv("AGENT_MODEL_2_INPUT_COST_PER_1M", str(default_agent_input))),
            "agent_output_cost_per_1m": float(os.getenv("AGENT_MODEL_2_OUTPUT_COST_PER_1M", str(default_agent_output))),
            "pricing_source": "model_2_specific",
        }
    return {
        "agent_input_cost_per_1m": default_agent_input,
        "agent_output_cost_per_1m": default_agent_output,
        "pricing_source": "global_default",
    }


def _compute_token_and_cost_metrics(results: List[Dict], default_model: str) -> Dict[str, Any]:
    default_cost_per_1m = float(os.getenv("TOKEN_COST_PER_1M", "1.0"))
    agent_pricing = _resolve_agent_pricing_for_model(default_model)
    agent_input_cost_per_1m = float(agent_pricing["agent_input_cost_per_1m"])
    agent_output_cost_per_1m = float(agent_pricing["agent_output_cost_per_1m"])
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

    agent_input_cost = round((totals["agent_input_tokens"] / 1_000_000) * agent_input_cost_per_1m, 6)
    agent_output_cost = round((totals["agent_output_tokens"] / 1_000_000) * agent_output_cost_per_1m, 6)
    agent_total_cost = round(agent_input_cost + agent_output_cost, 6)

    judge_eval_input_cost = round((totals["judge_input_tokens"] / 1_000_000) * judge_input_cost_per_1m, 6)
    judge_eval_output_cost = round((totals["judge_output_tokens"] / 1_000_000) * judge_output_cost_per_1m, 6)
    judge_eval_total_cost = round(judge_eval_input_cost + judge_eval_output_cost, 6)

    # Theo yêu cầu: cost dùng cho so sánh model chỉ tính AGENT input/output.
    cost = {
        "agent_input_cost": agent_input_cost,
        "agent_output_cost": agent_output_cost,
        "agent_total_cost": agent_total_cost,
        "pipeline_total_cost": agent_total_cost,
        # Judge cost chỉ để tham khảo eval pipeline, không dùng decision switch model.
        "judge_eval_input_cost": judge_eval_input_cost,
        "judge_eval_output_cost": judge_eval_output_cost,
        "judge_eval_total_cost": judge_eval_total_cost,
        "pipeline_total_cost_with_eval": round(agent_total_cost + judge_eval_total_cost, 6),
    }

    pricing = {
        "token_cost_per_1m_default": default_cost_per_1m,
        "agent_model_for_pricing": default_model,
        "agent_pricing_source": agent_pricing["pricing_source"],
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


def _avg(results: List[Dict], path: List[str]) -> float:
    if not results:
        return 0.0

    def _get(item: Dict, p: List[str]) -> float:
        cur: Any = item
        for key in p:
            if not isinstance(cur, dict):
                return 0.0
            cur = cur.get(key)
        return _safe_float(cur)

    return sum(_get(r, path) for r in results) / len(results)


def _pass_rate(results: List[Dict]) -> float:
    if not results:
        return 0.0
    passed = sum(1 for r in results if str(r.get("status", "")).lower() == "pass")
    return passed / len(results)


def _build_metrics(results: List[Dict]) -> Dict[str, float]:
    return {
        "avg_score": _avg(results, ["judge", "final_score"]),
        "agreement_rate": _avg(results, ["judge", "agreement_rate"]),
        "faithfulness": _avg(results, ["ragas", "faithfulness"]),
        "relevancy": _avg(results, ["ragas", "relevancy"]),
        "hit_rate": _avg(results, ["ragas", "retrieval", "hit_rate"]),
        "mrr": _avg(results, ["ragas", "retrieval", "mrr"]),
        "avg_latency_sec": _avg(results, ["latency"]),
        "pass_rate": _pass_rate(results),
    }


def _decide_model_switch(
    model_1_name: str,
    model_1_summary: Dict[str, Any],
    model_2_name: str,
    model_2_summary: Dict[str, Any],
) -> Dict[str, Any]:
    min_score_improvement = float(os.getenv("MODEL_SWITCH_MIN_SCORE_IMPROVEMENT", "0.05"))
    max_cost_increase_ratio = float(os.getenv("MODEL_SWITCH_MAX_COST_INCREASE_RATIO", "0.15"))
    require_non_decreasing_ragas = _to_bool_env(
        os.getenv("MODEL_SWITCH_REQUIRE_NON_DECREASING_RAGAS", "true"),
        default=True,
    )

    m1 = model_1_summary.get("metrics", {})
    m2 = model_2_summary.get("metrics", {})
    c1 = model_1_summary.get("cost_estimation", {})
    c2 = model_2_summary.get("cost_estimation", {})

    score_delta = _safe_float(m2.get("avg_score")) - _safe_float(m1.get("avg_score"))
    faithfulness_delta = _safe_float(m2.get("faithfulness")) - _safe_float(m1.get("faithfulness"))
    relevancy_delta = _safe_float(m2.get("relevancy")) - _safe_float(m1.get("relevancy"))
    agreement_delta = _safe_float(m2.get("agreement_rate")) - _safe_float(m1.get("agreement_rate"))
    model_1_cost = _safe_float(c1.get("agent_total_cost", c1.get("pipeline_total_cost")))
    model_2_cost = _safe_float(c2.get("agent_total_cost", c2.get("pipeline_total_cost")))
    cost_delta = model_2_cost - model_1_cost

    baseline_cost = max(model_1_cost, 1e-12)
    cost_increase_ratio = cost_delta / baseline_cost

    ragas_ok = True
    if require_non_decreasing_ragas:
        ragas_ok = faithfulness_delta >= 0 and relevancy_delta >= 0

    strong_quality_gain = score_delta >= min_score_improvement and ragas_ok
    affordable_cost = cost_increase_ratio <= max_cost_increase_ratio
    quality_not_worse = score_delta >= 0 and ragas_ok
    cost_better = cost_delta < 0

    should_switch = (strong_quality_gain and affordable_cost) or (quality_not_worse and cost_better)

    if should_switch:
        chosen_model = model_2_name
        reason = (
            "Model 2 đạt điều kiện đổi model: "
            "chất lượng tăng đủ ngưỡng và/hoặc chất lượng không giảm trong khi chi phí tốt hơn."
        )
    else:
        chosen_model = model_1_name
        reason = (
            "Giữ model 1 vì model 2 chưa vượt ngưỡng về chất lượng-chi phí theo policy hiện tại."
        )

    return {
        "policy": {
            "min_score_improvement": min_score_improvement,
            "max_cost_increase_ratio": max_cost_increase_ratio,
            "require_non_decreasing_ragas": require_non_decreasing_ragas,
        },
        "deltas": {
            "avg_score_delta": round(score_delta, 6),
            "faithfulness_delta": round(faithfulness_delta, 6),
            "relevancy_delta": round(relevancy_delta, 6),
            "agreement_rate_delta": round(agreement_delta, 6),
            "agent_total_cost_delta": round(cost_delta, 6),
            "agent_cost_increase_ratio": round(cost_increase_ratio, 6),
            # Backward-compat keys
            "pipeline_total_cost_delta": round(cost_delta, 6),
            "pipeline_cost_increase_ratio": round(cost_increase_ratio, 6),
        },
        "decision": {
            "should_switch_to_model_2": should_switch,
            "selected_model": chosen_model,
            "reason": reason,
        },
    }


async def run_benchmark_with_results(agent_version: str, agent_model: str) -> Tuple[List[Dict], Dict[str, Any]]:
    print(f"🚀 Khởi động Benchmark cho {agent_version} ({agent_model})...")

    if not os.path.exists("data/golden_set.jsonl"):
        raise FileNotFoundError("Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        raise ValueError("File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")

    load_dotenv()
    _require_tiktoken()

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    batch_size = int(os.getenv("BENCH_BATCH_SIZE", "5"))
    ragas_model = os.getenv("RAGAS_MODEL")

    judge = LLMJudge()
    evaluator = RAGASEvaluator(model=ragas_model, top_k=top_k)
    agent = MainAgent(model=agent_model)
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset, batch_size=batch_size)

    metrics = _build_metrics(results)
    token_cost = _compute_token_and_cost_metrics(results=results, default_model=agent_model)

    summary = {
        "metadata": {
            "version": agent_version,
            "agent_model": agent_model,
            "total": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
        "token_usage": token_cost["tokens"],
        "cost_estimation": token_cost["cost"],
        "pricing": token_cost["pricing"],
        "estimated_by_tiktoken_cases": token_cost["estimated_by_tiktoken_cases"],
    }
    return results, summary


async def main():
    load_dotenv()

    model_1 = os.getenv("AGENT_MODEL_1")
    model_2 = os.getenv("AGENT_MODEL_2")

    if not model_1 or not model_2:
        raise EnvironmentError(
            "Thiếu AGENT_MODEL_1 hoặc AGENT_MODEL_2 trong .env để chạy so sánh 2 mô hình."
        )

    model_1_results, model_1_summary = await run_benchmark_with_results(
        agent_version="Agent_Model_1",
        agent_model=model_1,
    )
    model_2_results, model_2_summary = await run_benchmark_with_results(
        agent_version="Agent_Model_2",
        agent_model=model_2,
    )

    comparison = _decide_model_switch(
        model_1_name=model_1,
        model_1_summary=model_1_summary,
        model_2_name=model_2,
        model_2_summary=model_2_summary,
    )

    should_switch = comparison["decision"]["should_switch_to_model_2"]
    selected_summary = model_2_summary if should_switch else model_1_summary

    print("\n📊 --- MODEL COMPARISON ---")
    print(f"Model 1: {model_1} | avg_score={model_1_summary['metrics']['avg_score']:.4f}")
    print(f"Model 2: {model_2} | avg_score={model_2_summary['metrics']['avg_score']:.4f}")
    print(f"Faithfulness delta: {comparison['deltas']['faithfulness_delta']:+.4f}")
    print(f"Relevancy delta: {comparison['deltas']['relevancy_delta']:+.4f}")
    print(f"Agent cost delta: {comparison['deltas']['agent_total_cost_delta']:+.6f}")
    print(f"Decision: {comparison['decision']['selected_model']}")
    print(f"Reason: {comparison['decision']['reason']}")

    summary_report = {
        "metadata": selected_summary["metadata"],
        "metrics": selected_summary["metrics"],
        "token_usage": selected_summary["token_usage"],
        "cost_estimation": selected_summary["cost_estimation"],
        "pricing": selected_summary["pricing"],
        "estimated_by_tiktoken_cases": selected_summary["estimated_by_tiktoken_cases"],
        "model_1": model_1_summary,
        "model_2": model_2_summary,
        "comparison": comparison,
    }
    results_report = {
        "model_1_results": model_1_results,
        "model_2_results": model_2_results,
        "comparison": comparison,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, ensure_ascii=False, indent=2)

    if should_switch:
        print("✅ QUYẾT ĐỊNH: ĐỔI SANG MODEL_2")
    else:
        print("✅ QUYẾT ĐỊNH: GIỮ MODEL_1")


if __name__ == "__main__":
    asyncio.run(main())