"""Voice Routes — Module 6.1 (Speech-to-Text).

Endpoints for audio upload, validation, and Groq Whisper transcription.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status

try:
    from backend.app.config import config
    from backend.app.database import get_db
    from backend.app.schemas.voice import (
        SynthesizeErrorResponse,
        SynthesizeRequest,
        TranscriptionErrorResponse,
        TranscriptionResponse,
        VoiceAgentResponse,
    )
    from backend.app.agents.orchestrator import AgentOrchestrator
    from backend.app.voice.speech_to_text import (
        SUPPORTED_AUDIO_EXTENSIONS,
        STTError,
        SpeechToTextProvider,
        get_stt_provider,
    )
    from backend.app.voice.text_to_speech import (
        TTSError,
        TTSSynthesisResult,
        TextToSpeechProvider,
        get_tts_provider,
    )
    from backend.app.voice.tts_service import TTSService
    from backend.app.voice.voice_agent_service import VoiceAgentService
    from backend.app.rag.answer_generator import clean_markdown_asterisks
    from backend.app.auth.dependencies import get_optional_current_user
    from backend.app.models.user import User
except ImportError:
    from app.config import config
    from app.database import get_db
    from app.schemas.voice import (
        SynthesizeErrorResponse,
        SynthesizeRequest,
        TranscriptionErrorResponse,
        TranscriptionResponse,
        VoiceAgentResponse,
    )
    from app.agents.orchestrator import AgentOrchestrator
    from app.voice.speech_to_text import (
        SUPPORTED_AUDIO_EXTENSIONS,
        STTError,
        SpeechToTextProvider,
        get_stt_provider,
    )
    from app.voice.text_to_speech import (
        TTSError,
        TTSSynthesisResult,
        TextToSpeechProvider,
        get_tts_provider,
    )
    from app.voice.tts_service import TTSService
    from app.voice.voice_agent_service import VoiceAgentService
    from app.rag.answer_generator import clean_markdown_asterisks
    from app.auth.dependencies import get_optional_current_user
    from app.models.user import User

logger = logging.getLogger("routes.voice")
router = APIRouter(prefix="/api/voice", tags=["voice"])

# Module-level provider resolver for dependency injection
_stt_provider: Optional[SpeechToTextProvider] = None
_voice_agent_service: Optional[VoiceAgentService] = None
_tts_service: Optional[TTSService] = None


def get_current_stt_provider() -> SpeechToTextProvider:
    """Dependency provider returning the configured STT provider instance."""
    global _stt_provider
    if _stt_provider is None:
        _stt_provider = get_stt_provider()
    return _stt_provider


def set_stt_provider(provider: Optional[SpeechToTextProvider]) -> None:
    """Sets the STT provider (used by unit tests for dependency injection)."""
    global _stt_provider
    _stt_provider = provider


def get_voice_agent_service(
    stt_provider: SpeechToTextProvider = Depends(get_current_stt_provider),
) -> VoiceAgentService:
    """Dependency provider returning the configured VoiceAgentService instance."""
    global _voice_agent_service
    if _voice_agent_service is not None:
        return _voice_agent_service
    return VoiceAgentService(stt_provider=stt_provider)


def set_voice_agent_service(service: Optional[VoiceAgentService]) -> None:
    """Sets the VoiceAgentService (used by unit tests for dependency injection)."""
    global _voice_agent_service
    _voice_agent_service = service


def get_tts_service() -> TTSService:
    """Dependency provider returning the configured TTSService instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


def set_tts_service(service: Optional[TTSService]) -> None:
    """Sets the TTSService (used by unit tests for dependency injection)."""
    global _tts_service
    _tts_service = service


def _resolve_tenant_id(tenant_id: Optional[str]) -> str:
    """Validates and resolves tenant UUID."""
    raw = tenant_id or config.tenant.default_id
    try:
        import uuid
        uuid.UUID(raw)
        return raw
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant_id: '{raw}'. Must be a valid UUID.",
        )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": TranscriptionErrorResponse, "description": "Invalid audio input or empty file"},
        413: {"model": TranscriptionErrorResponse, "description": "Audio file exceeds maximum upload size"},
        415: {"model": TranscriptionErrorResponse, "description": "Unsupported audio format"},
        502: {"model": TranscriptionErrorResponse, "description": "Upstream STT provider failure"},
        500: {"model": TranscriptionErrorResponse, "description": "Internal server error"},
    },
    summary="Transcribe audio to text",
    description=(
        "Uploads a voice recording (wav, mp3, m4a, webm, etc.) and returns the text transcript "
        "using Groq Whisper Large V3 Turbo."
    ),
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file binary (wav, mp3, m4a, webm, etc.)"),
    language: Optional[str] = Form(default=None, description="Optional ISO-639-1 language code (e.g. 'en', 'es')"),
    prompt: Optional[str] = Form(default=None, description="Optional context prompt to guide transcription"),
    provider: SpeechToTextProvider = Depends(get_current_stt_provider),
) -> TranscriptionResponse:
    """Handles multipart audio upload, performs validation, and executes transcription."""
    filename = audio.filename or "recording.wav"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 1. Validate extension
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning("Rejected upload with unsupported extension: '%s' (filename=%s)", ext, filename)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format: '.{ext}'. "
                f"Supported formats: {sorted(list(SUPPORTED_AUDIO_EXTENSIONS))}"
            ),
        )

    # 2. Read and validate size
    max_bytes = config.stt.max_upload_size_mb * 1024 * 1024
    content = await audio.read()
    file_size = len(content)

    if file_size == 0:
        logger.warning("Rejected upload with 0 bytes (filename=%s)", filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty (0 bytes received).",
        )

    if file_size > max_bytes:
        logger.warning(
            "Rejected oversized audio file: %d bytes (limit: %d bytes, filename=%s)",
            file_size,
            max_bytes,
            filename,
        )
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Audio file size ({file_size / (1024*1024):.1f}MB) exceeds maximum limit of {config.stt.max_upload_size_mb}MB.",
        )

    # 3. Temporary file lifecycle management
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        logger.info(
            "Dispatching transcription for '%s' (size=%d bytes, format=%s) to provider=%s",
            filename,
            file_size,
            ext,
            provider.model_name,
        )

        result = provider.transcribe(
            audio_file=tmp_path,
            filename=filename,
            language=language,
            prompt=prompt,
        )

        return TranscriptionResponse(
            success=True,
            text=result.text,
            language=result.language,
            duration=result.duration,
            segments=result.segments,
        )

    except STTError as exc:
        logger.error("STT provider error during transcription: %s", exc)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error during audio transcription: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the audio file.",
        ) from exc

    finally:
        # Guarantee deletion of temporary file in all conditions
        if tmp_path is not None and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError as del_exc:
                logger.warning("Failed to clean up temporary audio file %s: %s", tmp_path, del_exc)


@router.post(
    "/agent",
    response_model=VoiceAgentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": TranscriptionErrorResponse, "description": "Invalid audio input or empty file"},
        413: {"model": TranscriptionErrorResponse, "description": "Audio file exceeds maximum upload size"},
        415: {"model": TranscriptionErrorResponse, "description": "Unsupported audio format"},
        502: {"model": TranscriptionErrorResponse, "description": "Upstream STT or AI provider failure"},
        500: {"model": TranscriptionErrorResponse, "description": "Internal server error"},
    },
    summary="Voice Agent Orchestrator",
    description=(
        "End-to-end voice agent endpoint: accepts an audio recording, transcribes it via "
        "Groq Whisper, and routes the recognized request through the LangGraph Agent Orchestrator "
        "(RAG knowledge Q&A, tool intent, clarification, or decline) with tenant isolation."
    ),
)
async def voice_agent(
    audio: UploadFile = File(..., description="Audio file binary (wav, mp3, m4a, webm, etc.)"),
    tenant_id: Optional[str] = Form(default=None, description="Tenant UUID context (falls back to DEV_TENANT_ID)"),
    language: Optional[str] = Form(default=None, description="Optional ISO-639-1 language code"),
    prompt: Optional[str] = Form(default=None, description="Optional context prompt for STT"),
    conversation_id: Optional[str] = Form(default=None, description="Optional session/conversation ID"),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
    service: VoiceAgentService = Depends(get_voice_agent_service),
) -> VoiceAgentResponse:
    """Processes uploaded voice audio through STT and LangGraph agent orchestrator."""
    resolved_tenant_id = _resolve_tenant_id(tenant_id)
    if current_user and str(current_user.tenant_id):
        resolved_tenant_id = str(current_user.tenant_id)

    filename = audio.filename or "recording.wav"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 1. Validate audio extension
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        logger.warning("Rejected voice agent upload with unsupported extension: '%s' (filename=%s)", ext, filename)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format: '.{ext}'. "
                f"Supported formats: {sorted(list(SUPPORTED_AUDIO_EXTENSIONS))}"
            ),
        )

    # 2. Read and validate size
    max_bytes = config.stt.max_upload_size_mb * 1024 * 1024
    content = await audio.read()
    file_size = len(content)

    if file_size == 0:
        logger.warning("Rejected voice agent upload with 0 bytes (filename=%s)", filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty (0 bytes received).",
        )

    if file_size > max_bytes:
        logger.warning(
            "Rejected oversized audio file: %d bytes (limit: %d bytes, filename=%s)",
            file_size,
            max_bytes,
            filename,
        )
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Audio file size ({file_size / (1024*1024):.1f}MB) exceeds maximum limit of {config.stt.max_upload_size_mb}MB.",
        )

    # 3. Temporary file lifecycle management
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        logger.info(
            "Processing voice agent request for '%s' (size=%d bytes, tenant=%s, user=%s)",
            filename,
            file_size,
            resolved_tenant_id,
            getattr(current_user, "username", "anon"),
        )

        result: VoiceAgentResponse = service.process_voice_request(
            audio_file=tmp_path,
            filename=filename,
            tenant_id=resolved_tenant_id,
            language=language,
            prompt=prompt,
            conversation_id=conversation_id,
            current_user=current_user,
            db_session=db,
        )

        if result.response:
            result.response = clean_markdown_asterisks(result.response)

        return result

    except STTError as exc:
        logger.error("STT provider error in voice agent endpoint: %s", exc)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error in voice agent endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY if "orchestration" in str(exc).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc) if "orchestration" in str(exc).lower() else "An unexpected internal error occurred while processing the voice agent request.",
        ) from exc

    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError as del_exc:
                logger.warning("Failed to clean up temporary audio file %s: %s", tmp_path, del_exc)


@router.post(
    "/synthesize",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
            },
            "description": "Synthesized audio bytes in the requested format (default: audio/wav).",
        },
        400: {"model": SynthesizeErrorResponse, "description": "Invalid input text (empty or whitespace)"},
        413: {"model": SynthesizeErrorResponse, "description": "Text exceeds maximum allowed length"},
        422: {"model": SynthesizeErrorResponse, "description": "Validation error / unsupported format"},
        502: {"model": SynthesizeErrorResponse, "description": "Upstream TTS provider failure"},
        500: {"model": SynthesizeErrorResponse, "description": "Internal server error / unconfigured API key"},
    },
    summary="Synthesize text to speech audio",
    description=(
        "Converts a text response into speech audio (WAV by default) using Groq Orpheus TTS."
    ),
)
async def synthesize_speech(
    payload: SynthesizeRequest,
    service: TTSService = Depends(get_tts_service),
) -> Response:
    """Handles JSON synthesis request, validates text length and format, and returns raw audio bytes."""
    try:
        if payload.text:
            payload.text = clean_markdown_asterisks(payload.text)
        result: TTSSynthesisResult = service.synthesize_speech(payload)
        return Response(
            content=result.audio_bytes,
            media_type=result.content_type,
            headers={
                "Content-Disposition": f'inline; filename="synthesized.{result.format}"',
                "X-TTS-Provider": result.provider,
                "X-TTS-Model": result.model,
                "X-TTS-Voice": result.voice,
            },
        )
    except TTSError as exc:
        logger.error("TTS error during synthesis: %s (status=%d)", exc, exc.status_code)
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in synthesize endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while synthesizing speech.",
        ) from exc


