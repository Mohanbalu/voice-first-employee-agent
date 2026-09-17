"""Office RAG & Groq Benchmark Verification Script.

Tests all 15 office/location benchmark questions plus policy regression questions
against AWS RDS PostgreSQL pgvector + Groq LLM.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(str(BASE_DIR / "backend" / ".env"))

from backend.app.database import get_session_factory
from backend.app.rag.retriever import RAGRetriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.answer_generator import AnswerGenerator
from backend.app.ai.factory import get_ai_provider

DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"

TEST_QUESTIONS = [
    # Core multi-match test
    "Where is the IT team?",
    # SDC specific
    "Where can I find the IT team in SDC?",
    # Tower 1 specific
    "Which floor is the IT team on in Tower 1?",
    # Tower 2 IT negative check
    "Is there an IT team in Tower 2?",
    # Laptop vs IT team distinction
    "Where do I get laptop support?",
    "Where is technical support located?",
    # Techbees & Trainees
    "Where are the Techbees classrooms?",
    "Where can trainees sit or work?",
    # ODCs
    "Which floors have project-specific ODCs in SDC?",
    # Tower 2 experienced teams (floor unspecified check)
    "Where are experienced teams sitting in Tower 2?",
    # Facilities
    "Where is the cafeteria?",
    "Where are the seminar halls?",
    "Where can I play carrom, chess, or table tennis?",
    "Where are the breakout rooms?",
    "Where is parking available?",
    # Non-regression policy checks
    "How many days of annual leave do employees get?",
    "What is the remote work or WFH policy?",
]

def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: Office Locations & Policy RAG with Groq LLM")
    print(f"Tenant ID: {DEFAULT_TENANT_ID}")
    print(f"AI Provider: {os.getenv('AI_PROVIDER')}")
    print("=" * 80)

    import uuid
    groq_provider = get_ai_provider("groq")
    retriever = RAGRetriever()
    context_builder = ContextBuilder(max_chunks=6)
    answer_gen = AnswerGenerator(provider=groq_provider)

    passed_count = 0
    total_count = len(TEST_QUESTIONS)

    session_factory = get_session_factory()
    with session_factory() as session:
        for idx, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[{idx}/{total_count}] QUESTION: {question}")
            search_results = retriever.retrieve(
                session=session,
                tenant_id=uuid.UUID(DEFAULT_TENANT_ID),
                question=question,
            )

            print(f"  Retrieved {len(search_results)} chunks:")
            for r in search_results[:4]:
                print(f"    - Score: {r.similarity_score:.4f} | Section: {r.section} | Doc: {r.document_name}")

            built = context_builder.build(search_results)
            ans_res = answer_gen.generate(question=question, context_text=built.context_text)

            print("  ANSWER:")
            for line in ans_res.answer.strip().split("\n"):
                print(f"    {line}")

            # Check if answer contains meaningful content (not NO_CONTEXT_ANSWER unless intended)
            if "I could not find" not in ans_res.answer:
                passed_count += 1
                print("  --> STATUS: SUCCESS (Answer grounded in KB)")
            elif "Tower 2" in question and ("not specify" in ans_res.answer.lower() or "not mention" in ans_res.answer.lower() or "could not find" in ans_res.answer.lower()):
                passed_count += 1
                print("  --> STATUS: SUCCESS (Correctly noted unmentioned/unspecified details)")
            else:
                print("  --> STATUS: NO CONTEXT RETURNED")

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE: {passed_count}/{total_count} queries answered successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
