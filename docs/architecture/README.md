# System Architecture

Architectural overview for the Voice-First Agentic AI Employee Workplace Assistant.

## High-Level Architecture
1. **Frontend Client**: React.js + Vite voice interface with audio capture and dashboard panels.
2. **Backend API**: FastAPI application managing WebSocket audio streams, REST endpoints, and authentication.
3. **Agent Orchestration**: Multi-agent system built on LangGraph (Knowledge, HR/IT Ticket Escalation, Navigation, Calendar, Tasks, Timesheet, Follow-up).
4. **Knowledge & RAG**: Document parsing, chunking, embedding generation, and pgvector-backed semantic retrieval.
5. **Database**: PostgreSQL with pgvector extension storing relational entities and vector embeddings.
