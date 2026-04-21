import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from engine.ragas_eval import RAGASEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent


async def test():
    with open("data/golden_set.jsonl") as f:
        case = json.loads(f.readline())

    print("Question :", case["question"][:100])
    print()

    agent = MainAgent()
    resp = await agent.query(case["question"])
    print("Answer   :", resp["answer"][:100])
    print()

    evaluator = RAGASEvaluator()
    result = await evaluator.score(case, resp)
    print(f"Faithfulness : {result['faithfulness']}  ({result['faithfulness_detail']['verified']}/{result['faithfulness_detail']['total']} statements)")
    for s in result["faithfulness_detail"]["statements"]:
        mark = "✅" if s["entailed"] else "❌"
        print(f"  {mark} {s['text']}")
    print()
    print(f"Relevancy    : {result['relevancy']}")
    print(f"  Generated Q: {result['relevancy_detail']['generated_question']}")
    print()
    print(f"Hit Rate     : {result['retrieval']['hit_rate']}")
    print(f"MRR          : {result['retrieval']['mrr']}")

    print()
    print("--- LLM Judge ---")
    judge = LLMJudge()
    judge_result = await judge.evaluate_multi_judge(
        case["question"], resp["answer"], case["expected_answer"]
    )
    print(f"Final Score  : {judge_result['final_score']}")
    print(f"Agreement    : {judge_result['agreement_rate']}")
    print(f"Tie-breaker  : {judge_result['tie_breaker_used']}")


asyncio.run(test())
