"""Live verification script for HCL Office Knowledge & Existing Policy RAG.
Runs real queries through RAGService connecting to AWS RDS pgvector and Groq LLM.
"""

import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir.parent))
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.app.rag.rag_service import RAGService
from backend.app.rag.retriever import RetrievalConfig
from backend.app.database import get_session_factory

TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

test_questions = [
    "Where is the cafeteria?",
    "Where is the IT team located in SDC?",
    "Where do I go for laptop issues and technical support?",
    "Which floor are the experienced teams on in Tower 2?",
    "Where can I play table tennis or chess?",
    "Where is parking located on campus?",
    "Where are Techbees classrooms located?",
    "Where are the seminar halls located?",
    "Where are the trainee ODCs located?",
    "What is the annual leave policy for employees?",
]

def main():
    print("=" * 70)
    print("LIVE HCL OFFICE KNOWLEDGE & POLICY RAG VERIFICATION")
    print("=" * 70)

    session_factory = get_session_factory()
    service = RAGService(retrieval_config=RetrievalConfig(top_k=3, min_similarity=0.4))

    with session_factory() as session:
        for idx, q in enumerate(test_questions, 1):
            print(f"\n[{idx}/{len(test_questions)}] QUESTION: {q}")
            resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question=q, session=session)
            print(f"CONFIDENCE: {resp.confidence}")
            print(f"SOURCES USED: {resp.used_context_count}")
            for src in resp.sources:
                print(f"  • {src.document_name} | {src.section} (score: {src.similarity_score:.3f})")
            print(f"ANSWER:\n{resp.answer}\n")
            print("-" * 70)

if __name__ == "__main__":
    main()
