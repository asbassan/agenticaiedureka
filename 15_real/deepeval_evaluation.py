# pip install deepeval openai chromadb python-dotenv

import os
import sys
import csv
import random
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, ContextualPrecisionMetric
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
from support_ticketing_agent_hitl import retrieve, generate, openai_client  # evaluate the REAL RAG pipeline

# =====================================================================
# DeepEval evaluation of the SAME retrieve()/generate() pipeline that
# powers support_ticketing_agent_hitl.py, on the same real, database-
# sourced test cases as 15_real/ragas_evaluation.py - see that file for
# the full rationale (real FAQ ground truth + paraphrased questions,
# not hand-invented test cases).
#
# This file exists to contrast frameworks, not pipelines: RAGAS scores
# metrics itself via its own `.score()` calls; DeepEval instead hands
# an LLMTestCase to each metric's `.measure()` and reads `.score`/
# `.reason` back - one added benefit being a human-readable reason
# string per metric, not just a number.
#
# Same four metrics, same meaning as the RAGAS example:
#   Faithfulness             - does the answer only claim things that
#                               are actually IN the retrieved FAQs, or
#                               did the model add/invent something?
#   Answer Relevancy         - does the answer actually address what
#                               was asked (vs. a vague non-answer)?
#   Contextual Recall        - did retrieval find what the REAL FAQ
#                               answer needed? (expected_output = the
#                               database's own answer)
#   Contextual Precision     - of what was retrieved, how much was
#                               actually useful for the answer given?
# =====================================================================

EVAL_MODEL = "gpt-4o-mini"

faithfulness = FaithfulnessMetric(model=EVAL_MODEL)
answer_relevancy = AnswerRelevancyMetric(model=EVAL_MODEL)
contextual_recall = ContextualRecallMetric(model=EVAL_MODEL)
contextual_precision = ContextualPrecisionMetric(model=EVAL_MODEL)

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")


def paraphrase(question: str) -> str:
    """Rephrase a real FAQ question the way an actual user would type it - same
    meaning, different wording - so retrieval is tested on realistic input."""
    return openai_client.responses.create(
        model=EVAL_MODEL,
        instructions="Rephrase this customer support question naturally and casually, the "
                     "way a real user typing quickly would. Keep the same meaning. Return "
                     "ONLY the rephrased question, nothing else.",
        input=question,
    ).output_text.strip()


def build_test_cases_from_real_faqs(seed: int = 42) -> list[dict]:
    """Sample one real (question, answer) pair per category from the actual FAQ
    database - the SAME database the pipeline retrieves from - as the eval set."""
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    random.seed(seed)
    cases = []
    for category, entries in sorted(by_category.items()):
        entry = random.choice(entries)
        cases.append({
            "category": category,
            "original_question": entry["question"],
            "question": paraphrase(entry["question"]),
            "reference": entry["answer"],
        })
    return cases


TEST_CASES = build_test_cases_from_real_faqs()


def evaluate_case(question: str, reference: str) -> dict:
    faqs, _ = retrieve(question)
    retrieved_contexts = [f"Q: {faq['question']} A: {faq['answer']}" for faq in faqs]
    response = generate(question, faqs)

    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=reference,
        retrieval_context=retrieved_contexts,
    )

    faithfulness.measure(test_case)
    answer_relevancy.measure(test_case)
    contextual_recall.measure(test_case)
    contextual_precision.measure(test_case)

    return {
        "question": question,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "faithfulness": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
        "answer_relevancy": answer_relevancy.score,
        "answer_relevancy_reason": answer_relevancy.reason,
        "contextual_recall": contextual_recall.score,
        "contextual_precision": contextual_precision.score,
    }


if __name__ == "__main__":
    results = []

    for case in TEST_CASES:
        print("=" * 70)
        print(f"Category: {case['category']}")
        print(f"Original FAQ question: {case['original_question']}")
        print(f"Paraphrased as asked:  {case['question']}")

        result = evaluate_case(case["question"], case["reference"])
        results.append(result)

        print(f"Response: {result['response']}")
        print(f"Faithfulness:         {result['faithfulness']:.2f}  ({result['faithfulness_reason']})")
        print(f"Answer Relevancy:     {result['answer_relevancy']:.2f}  ({result['answer_relevancy_reason']})")
        print(f"Contextual Recall:    {result['contextual_recall']:.2f}")
        print(f"Contextual Precision: {result['contextual_precision']:.2f}")
        print()

    print("=" * 70)
    print("AVERAGES ACROSS ALL TEST CASES")
    for metric in ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision"]:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"  {metric}: {avg:.2f}")
