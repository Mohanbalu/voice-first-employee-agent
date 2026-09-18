import React, { useState, useRef, useEffect, useCallback } from 'react';
import { sendVoiceAgentAudio, synthesizeSpeech, sendTextAgentRequest, createTicket } from '../services/api';

/**
 * Sanitizes RAG citation sources to ensure NO raw filesystem paths are leaked.
 *
 * @param {Array} sources - Raw backend rag_sources array
 * @returns {Array} Sanitized source descriptors for display
 */
function sanitizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.map((src, idx) => {
    const docName = src.document_name || src.document_id || `Policy Reference ${idx + 1}`;
    let pages = '';
    if (src.page_start != null && src.page_end != null) {
      pages = src.page_start === src.page_end ? `p. ${src.page_start}` : `pp. ${src.page_start}–${src.page_end}`;
    } else if (src.page_start != null) {
      pages = `p. ${src.page_start}`;
    }
    return {
      id: src.chunk_id || `src-${idx}`,
      name: docName,
      pages: pages,
      score: src.similarity_score ? (src.similarity_score * 100).toFixed(0) + '%' : null,
    };
  });
}

/**
 * Strips all asterisk symbols from text for clean, human-readable display without markdown clutter.
 *
 * @param {string} text - Raw message text
 * @returns {string} Clean plain text without asterisks
 */
function removeAsterisks(text) {
  if (!text) return '';
  // Convert bullet points starting with * or • to -
  let clean = text.replace(/^\s*[*•]\s+/gm, '- ');
  // Remove all remaining asterisks (bold, italic, etc.)
  clean = clean.replace(/\*/g, '');
  return clean;
}

/**
 * Detects whether an assistant response indicates inability to answer
 * or requires support ticket escalation.
 */
function isUnableToAnswer(text, suggestTicket = false) {
  if (suggestTicket) return true;
  if (!text) return false;
  const lower = text.toLowerCase();
  const markers = [
    'could not find',
    'unable to find',
    'cannot find',
    'not find relevant information',
    'no relevant information',
    'not available in the company knowledge',
    'not mentioned in the available',
    'do not have information',
    "don't have information",
    'not found in the records',
    'not specified in current company records',
    'not specified in the provided records',
    'please raise a ticket',
    'contact it support',
    'contact hr',
    'please contact',
    'outside the scope of what i can help with',
    "i'd like to help, but i'm not sure what you're looking for",
  ];
  return markers.some((m) => lower.includes(m));
}

/**
 * Automatically infers support ticket category from query text.
 */
function detectTicketCategory(query) {
  const q = (query || '').toLowerCase();
  if (/laptop|wifi|network|software|hardware|it\b|vpn|mouse|keyboard|monitor|password|login|email|outlook|printer|techbee/.test(q)) {
    return 'IT';
  }
  if (/tower|floor|sdc|room|cafeteria|parking|seminar|breakout|desk|odc|play area|chess|carrom|table tennis|facility|facilities|location/.test(q)) {
    return 'FACILITIES';
  }
  if (/leave|holiday|casual|probation|salary|payroll|bonus|policy|insurance|medical|manager|resignation|notice|hr\b/.test(q)) {
    return 'HR';
  }
  return 'GENERAL';
}

/** Detects if assistant response confirms a scheduled reminder */
function isScheduleResponse(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  return (
    t.includes('scheduled for') ||
    t.includes('reminder created') ||
    t.includes('reminder set') ||
    t.includes('i have scheduled') ||
    t.includes('i have created a reminder') ||
    t.includes('i have set a reminder') ||
    t.includes('reminder has been scheduled')
  );
}

/** Detects if assistant response gives campus location/directions */
function isLocationResponse(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  return (
    (t.includes('tower 1') || t.includes('tower 2') || t.includes('floor') || t.includes('sdc') || t.includes('cafeteria') || t.includes('recreation') || t.includes('odc')) &&
    (t.includes('located') || t.includes('direction') || t.includes('campus') || t.includes('room') || t.includes('gps'))
  );
}

/**
 * VoiceAssistant Component — Module 6.4
 *
 * Reusable enterprise workplace assistant component with native browser MediaRecorder,
 * real-time status transitions, audio playback, and clean policy source rendering.
 */
export default function VoiceAssistant() {
  // Voice states: 'idle' | 'recording' | 'processing' | 'speaking' | 'completed' | 'error'
  const [voiceState, setVoiceState] = useState('idle');
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);
  const [ttsNotice, setTtsNotice] = useState(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  
  // Text chat input state
  const [inputText, setInputText] = useState('');

  // Conversation thread: Array of { id, role, text, sources, scheduleData, suggestTicket, queryText }
  const [messages, setMessages] = useState([]);

  // ── Raise-Ticket modal state ──────────────────────────────────────────────
  const [ticketModal, setTicketModal] = useState(null); // null | { queryText, prefillCategory, prefillSubject, prefillDesc }
  const [ticketForm, setTicketForm] = useState({ category: 'GENERAL', subject: '', description: '', priority: 'MEDIUM' });
  const [submittingTicket, setSubmittingTicket] = useState(false);
  const [ticketError, setTicketError] = useState(null);
  // Map of messageId -> ticket number after successful submission
  const [raisedTickets, setRaisedTickets] = useState({});
  const lastPromptRef = useRef(null); // tracks last prompt for retry

  // Native refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const currentAudioRef = useRef(null);
  const audioUrlRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, voiceState]);

  // Clean up media streams and audio resources on unmount
  useEffect(() => {
    return () => {
      stopMediaStream();
      cleanupAudio();
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
  }, []);

  /** Stop and release active microphone tracks */
  const stopMediaStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  /** Stop and cleanup playing audio elements and object URLs */
  const cleanupAudio = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    setIsPlayingAudio(false);
  }, []);

  /** Formats recording seconds into mm:ss */
  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const rem = secs % 60;
    return `${mins}:${rem < 10 ? '0' : ''}${rem}`;
  };

  /** Dynamically checks browser supported recording MIME types */
  const getSupportedMimeType = () => {
    const candidateTypes = [
      { mime: 'audio/webm;codecs=opus', ext: 'webm' },
      { mime: 'audio/webm', ext: 'webm' },
      { mime: 'audio/mp4', ext: 'mp4' },
      { mime: 'audio/ogg;codecs=opus', ext: 'ogg' },
      { mime: 'audio/wav', ext: 'wav' },
    ];
    for (const candidate of candidateTypes) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(candidate.mime)) {
        return candidate;
      }
    }
    return { mime: '', ext: 'wav' };
  };

  /** Starts microphone recording */
  const startRecording = async () => {
    setErrorMessage(null);
    setTtsNotice(null);
    cleanupAudio();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage('Audio recording is not supported on this browser.');
      setVoiceState('error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const { mime, ext } = getSupportedMimeType();
      const options = mime ? { mimeType: mime } : {};
      const recorder = new MediaRecorder(stream, options);

      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mime || 'audio/wav' });
        stopMediaStream();
        handleRecordedAudio(audioBlob, ext);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100); // 100ms timeslice for responsive chunking

      setVoiceState('recording');
      setRecordingSeconds(0);
      timerIntervalRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      stopMediaStream();
      console.error('Microphone access failed:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Microphone access was denied. Please allow microphone permissions in your browser to use voice input.');
      } else {
        setErrorMessage(`Microphone error: ${err.message || 'Unable to access audio input.'}`);
      }
      setVoiceState('error');
    }
  };

  /** Stops microphone recording */
  const stopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setVoiceState('processing');
    }
  };

  /** Handles the recorded audio blob and dispatches to /api/voice/agent */
  const handleRecordedAudio = async (audioBlob, ext) => {
    if (!audioBlob || audioBlob.size === 0) {
      setErrorMessage('No audio recorded. Please try speaking again.');
      setVoiceState('error');
      return;
    }

    const filename = `voice_input.${ext || 'webm'}`;

    try {
      // 1. Send audio to Voice Agent endpoint
      const result = await sendVoiceAgentAudio(audioBlob, { filename });

      const transcript = result.transcript || '(Unrecognized speech)';
      const assistantResponse = removeAsterisks(result.response || "I couldn't process that question.");
      const sanitizedSources = sanitizeSources(result.rag_sources);
      const backendSuggestTicket = !!result.suggest_ticket;
      const scheduleData = result.schedule_data || null;

      if (scheduleData) {
        window.dispatchEvent(new CustomEvent('schedule-created', { detail: scheduleData }));
      }

      // 2. Append User turn to conversation
      const userMessageId = `user-${Date.now()}`;
      const assistantMessageId = `asst-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        { id: userMessageId, role: 'user', text: transcript, timestamp: new Date() },
        {
          id: assistantMessageId,
          role: 'assistant',
          text: assistantResponse,
          sources: sanitizedSources,
          scheduleData: scheduleData,
          timestamp: new Date(),
          suggestTicket: isUnableToAnswer(assistantResponse, backendSuggestTicket),
          queryText: transcript,
        },
      ]);

      // 3. Trigger Text-to-Speech synthesis
      triggerTtsPlayback(assistantResponse, assistantMessageId);
      setVoiceState('completed');
    } catch (err) {
      console.error('Voice agent failed:', err);
      setErrorMessage(err.message || 'Failed to process voice request with backend agent.');
      setVoiceState('error');
    }
  };

  /**
   * Prepares clean, natural text for speech synthesis:
   * - Strips markdown symbols (**, ##, bullets, backticks)
   * - Removes citation references like (Source 1, page 3)
   * - Trims to sentence boundary within Groq's 1,200 TPM rate limit
   */
  const prepareSpokenText = (rawText, maxChars = 550) => {
    if (!rawText) return '';
    let clean = rawText
      .replace(/#+\s+/g, '')
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/`{1,3}(.*?)`{1,3}/g, '$1')
      .replace(/\(Source[^)]*\)/gi, '')
      .replace(/^\s*[-*+]\s+/gm, '')
      .replace(/\n+/g, ' ')
      .trim();

    if (clean.length <= maxChars) return clean;

    const truncated = clean.substring(0, maxChars);
    const lastSentenceEnd = Math.max(
      truncated.lastIndexOf('. '),
      truncated.lastIndexOf('! '),
      truncated.lastIndexOf('? ')
    );

    if (lastSentenceEnd > 120) {
      return truncated.substring(0, lastSentenceEnd + 1);
    }
    return truncated + '...';
  };

  /** Native Web Speech API fallback for when cloud TTS is rate limited or unavailable */
  const playNativeSpeech = (textToSpeak) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        // Try to pick a high quality English voice if available
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(
          (v) => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Microsoft') || v.name.includes('Samantha') || v.name.includes('Zira') || v.name.includes('David'))
        ) || voices.find((v) => v.lang.startsWith('en'));
        if (preferredVoice) {
          utterance.voice = preferredVoice;
        }

        utterance.onstart = () => {
          setIsPlayingAudio(true);
          setVoiceState('speaking');
        };
        utterance.onend = () => {
          setIsPlayingAudio(false);
          setVoiceState('completed');
        };
        utterance.onerror = (err) => {
          console.warn('Native speech playback error:', err);
          setIsPlayingAudio(false);
          setVoiceState('completed');
        };

        window.speechSynthesis.speak(utterance);
        return true;
      } catch (err) {
        console.warn('Native SpeechSynthesis failed:', err);
        return false;
      }
    }
    return false;
  };

  /** Calls /api/voice/synthesize and auto-plays response with native speech fallback */
  const triggerTtsPlayback = async (textToSpeak, messageId) => {
    const spokenText = prepareSpokenText(textToSpeak, 300);
    try {
      const audioBlob = await synthesizeSpeech(spokenText, { voice: 'autumn' });
      playAudioBlob(audioBlob, messageId);
      setTtsNotice(null);
    } catch (ttsErr) {
      console.info('Cloud TTS rate-limited or unavailable; falling back to browser speech synthesis:', ttsErr);
      const played = playNativeSpeech(spokenText);
      if (played) {
        // Speech is playing natively, clear any error notice
        setTtsNotice(null);
      } else {
        setTtsNotice('Voice playback is currently unavailable.');
      }
    }
  };

  /** Plays an audio blob via HTMLAudioElement */
  const playAudioBlob = (blob, messageId) => {
    cleanupAudio();

    try {
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;

      const audio = new Audio(url);
      currentAudioRef.current = audio;

      // Update message with audio URL for manual replay
      setMessages((prev) =>
        prev.map((msg) => (msg.id === messageId ? { ...msg, audioBlob: blob, audioUrl: url } : msg))
      );

      audio.onplay = () => {
        setIsPlayingAudio(true);
        setVoiceState('speaking');
      };

      audio.onended = () => {
        setIsPlayingAudio(false);
        setVoiceState('completed');
      };

      audio.onerror = (e) => {
        console.warn('Audio playback error:', e);
        setIsPlayingAudio(false);
        setVoiceState('completed');
      };

      audio.play().catch((playErr) => {
        console.warn('Browser blocked audio autoplay:', playErr);
        setIsPlayingAudio(false);
        setVoiceState('completed');
        setTtsNotice('Audio ready. Tap Play below to listen.');
      });
    } catch (err) {
      console.error('Failed creating audio object URL:', err);
    }
  };

  /** Manually replay audio from a message */
  const replayMessageAudio = (msg) => {
    if (msg.audioBlob) {
      playAudioBlob(msg.audioBlob, msg.id);
    } else if (msg.text) {
      const spoken = prepareSpokenText(msg.text, 300);
      const played = playNativeSpeech(spoken);
      if (!played) {
        triggerTtsPlayback(msg.text, msg.id);
      }
    }
  };

  /** Toggle active playback pause/resume */
  const togglePlayPause = () => {
    if (currentAudioRef.current) {
      if (isPlayingAudio) {
        currentAudioRef.current.pause();
        setIsPlayingAudio(false);
        setVoiceState('completed');
      } else {
        currentAudioRef.current.play();
        setIsPlayingAudio(true);
        setVoiceState('speaking');
      }
    } else if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      if (window.speechSynthesis.speaking) {
        if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
          setIsPlayingAudio(true);
          setVoiceState('speaking');
        } else {
          window.speechSynthesis.pause();
          setIsPlayingAudio(false);
          setVoiceState('completed');
        }
      }
    }
  };

  /** Handle quick prompt or text submission */
  const handleQuickPrompt = (promptText) => {
    lastPromptRef.current = promptText;
    setErrorMessage(null);
    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: 'user', text: promptText, timestamp: new Date() },
    ]);
    setVoiceState('processing');

    sendTextAgentRequest(promptText)
      .then((res) => {
        const assistantId = `asst-${Date.now()}`;
        const sanitizedSources = sanitizeSources(res.rag_sources);
        const cleanResponse = removeAsterisks(res.response || 'No response returned.');
        const backendSuggest = !!res.suggest_ticket;
        const scheduleData = res.schedule_data || null;
        if (scheduleData) {
          window.dispatchEvent(new CustomEvent('schedule-created', { detail: scheduleData }));
        }
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: 'assistant',
            text: cleanResponse,
            sources: sanitizedSources,
            scheduleData: scheduleData,
            timestamp: new Date(),
            suggestTicket: isUnableToAnswer(cleanResponse, backendSuggest),
            queryText: promptText,
          },
        ]);
        triggerTtsPlayback(cleanResponse, assistantId);
        setVoiceState('completed');
      })
      .catch((err) => {
        const errText = err.message || 'Request failed. Please try again.';
        // Show inline error bubble so the user is never left with a blank spinning state
        setMessages((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            role: 'assistant',
            text: `⚠️ ${errText}`,
            isError: true,
            timestamp: new Date(),
          },
        ]);
        setErrorMessage(errText);
        setVoiceState('idle'); // reset to idle so the user can retry immediately
      });
  };

  /** Handle keyboard text input submission */
  const handleTextSubmit = (e) => {
    if (e) e.preventDefault();
    const query = inputText.trim();
    if (!query) return;
    setInputText('');
    handleQuickPrompt(query);
  };

  const commandActions = [
    { label: 'Policy Inquiry', icon: '📖', query: 'What is the annual leave policy for employees?' },
    { label: 'Set Reminder', icon: '⏰', query: 'Remind me tomorrow at 10 AM to review project deliverable' },
    { label: 'Raise Ticket', icon: '🎫', isTicket: true },
    { label: 'Campus Location', icon: '📍', query: 'Where is the Cafeteria located on campus?' },
    { label: "Today's Schedule", icon: '📅', query: 'What are my upcoming reminders?' },
  ];

  const quickPrompts = [
    '📍 Where am I located in the office?',
    'What is the annual leave policy for employees?',
    'How many casual leaves can I take in a year?',
    'Remind me in 30 mins to submit timesheet',
    'Where is Conference Room B located?',
  ];

  return (
    <>
    <div className="assistant-container" role="region" aria-label="AI Voice Workplace Assistant">
      {/* ── Top Status Header ── */}
      <div className="assistant-header">
        <div className="assistant-info">
          <div className="assistant-avatar" aria-hidden="true">
            AI
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
              Workplace Command Center & Assistant
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Voice & Text Operations • AWS RDS pgvector Grounded • Time Scheduling
            </div>
          </div>
        </div>

        {/* Dynamic Status Indicator */}
        <div
          className={`assistant-status-tag status-${voiceState}`}
          aria-live="polite"
          aria-atomic="true"
        >
          <span className="status-dot" aria-hidden="true"></span>
          {voiceState === 'idle' && 'Ready'}
          {voiceState === 'recording' && `Listening (${formatTime(recordingSeconds)})`}
          {voiceState === 'processing' && 'Thinking & Searching Knowledge Base...'}
          {voiceState === 'speaking' && 'Speaking Response'}
          {voiceState === 'completed' && 'Ready'}
          {voiceState === 'error' && 'Action Required'}
        </div>
      </div>

      {/* ── Command Center Action Bar ── */}
      <div className="command-bar" role="toolbar" aria-label="Command Center Actions">
        <span className="command-bar-label">⚡ Quick Actions:</span>
        {commandActions.map((cmd, i) => (
          <button
            key={i}
            className="command-btn"
            onClick={() => {
              if (cmd.isTicket) {
                setTicketForm({
                  category: 'GENERAL',
                  subject: 'Workplace Inquiry',
                  description: 'Requesting workplace support.',
                  priority: 'MEDIUM',
                });
                setTicketError(null);
                setTicketModal({ messageId: `manual-${Date.now()}` });
              } else {
                handleQuickPrompt(cmd.query);
              }
            }}
            disabled={voiceState === 'recording' || voiceState === 'processing'}
          >
            <span>{cmd.icon}</span>
            <span>{cmd.label}</span>
          </button>
        ))}
      </div>

      {/* ── Error Banner ── */}
      {errorMessage && (
        <div className="alert-box alert-error" role="alert">
          <div>{errorMessage}</div>
          <button
            className="alert-dismiss"
            onClick={() => setErrorMessage(null)}
            aria-label="Dismiss error notice"
          >
            ✕
          </button>
        </div>
      )}

      {/* ── TTS Notice Banner ── */}
      {ttsNotice && (
        <div className="alert-box" style={{ background: '#eff6ff', border: '1px solid #bfdbfe', color: '#1e3a8a' }}>
          <div>{ttsNotice}</div>
          <button
            className="alert-dismiss"
            onClick={() => setTtsNotice(null)}
            aria-label="Dismiss TTS notice"
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Conversation Scroll Thread ── */}
      <div className="conversation-thread" tabIndex="0" aria-label="Conversation History">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon" aria-hidden="true">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
              </svg>
            </div>
            <h3>Enterprise AI Workplace Assistant</h3>
            <p>
              Speak naturally using the microphone or type below. Ask about verified policies, set time reminders, locate campus amenities, or escalate support tickets.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.role}`}>
              <div className="message-avatar" aria-hidden="true">
                {msg.role === 'user' ? 'You' : 'AI'}
              </div>
              <div className="message-bubble">
                <div className="message-sender">
                  {msg.role === 'user' ? 'Employee' : 'AI Workplace Assistant'}
                </div>
                <div className="message-text">{removeAsterisks(msg.text)}</div>

                {/* 1. Policy References & Citations (Zero Internal Path Exposure) */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="response-card response-card-policy">
                    <div className="response-card-header">
                      <span>📚 Verified Policy Grounding</span>
                      <span style={{ fontSize: '11px', opacity: 0.8 }}>RDS pgvector</span>
                    </div>
                    <div className="sources-list">
                      {msg.sources.map((src) => (
                        <span key={src.id} className="source-badge" title={`Confidence: ${src.score || 'Grounded'}`}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                          </svg>
                          <span>{src.name}</span>
                          {src.pages && <span style={{ opacity: 0.7 }}>({src.pages})</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* 2. Scheduled Reminder Card */}
                {msg.role === 'assistant' && (msg.scheduleData || isScheduleResponse(msg.text)) && (
                  <div className="response-card response-card-schedule">
                    <div className="response-card-header">
                      <span>⏰ Workplace Reminder Scheduled</span>
                      <span className="schedule-pill">Active</span>
                    </div>
                    <div className="schedule-card-row">
                      <div>
                        <strong>{msg.scheduleData?.title || 'Reminder Scheduled'}</strong>
                        {msg.scheduleData?.scheduled_at && (
                          <div style={{ fontSize: '12px', marginTop: '3px', color: '#581c87' }}>
                            📅 {new Date(msg.scheduleData.scheduled_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true })}
                            {msg.scheduleData.recurrence_type && msg.scheduleData.recurrence_type !== 'NONE' && (
                              <span> • 🔄 {msg.scheduleData.recurrence_type}</span>
                            )}
                          </div>
                        )}
                        <div style={{ marginTop: '8px' }}>
                          <button
                            type="button"
                            onClick={() => {
                              window.history.pushState({}, '', '/schedules');
                              window.dispatchEvent(new PopStateEvent('popstate'));
                            }}
                            style={{
                              background: '#7e22ce',
                              color: '#ffffff',
                              border: 'none',
                              padding: '4px 10px',
                              borderRadius: '4px',
                              fontSize: '12px',
                              fontWeight: '600',
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                          >
                            <span>View in Schedules Tab</span>
                            <span>→</span>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 3. Campus Facility & Directions Card */}
                {msg.role === 'assistant' && isLocationResponse(msg.text) && (
                  <div className="response-card response-card-location">
                    <div className="response-card-header">
                      <span>📍 Campus Directions & Navigation</span>
                      <span className="location-pill">HCL Campus Grounding</span>
                    </div>
                    <div style={{ fontSize: '12px' }}>
                      Navigate via designated campus walkways. Check floor directories at the central lobby.
                    </div>
                  </div>
                )}

                {/* Replay Audio Control for Assistant Messages */}
                {msg.role === 'assistant' && (
                  <div className="audio-controls-bar">
                    <button
                      className="play-audio-btn"
                      onClick={() => replayMessageAudio(msg)}
                      aria-label="Replay audio for this response"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                      </svg>
                      Listen
                    </button>
                    {isPlayingAudio && currentAudioRef.current && (
                      <button
                        className="play-audio-btn"
                        onClick={togglePlayPause}
                        aria-label="Pause audio playback"
                      >
                        ⏸ Pause
                      </button>
                    )}
                  </div>
                )}

                {/* ── 4. Raise Support Ticket Banner (shown on unanswered queries) ── */}
                {msg.role === 'assistant' && msg.suggestTicket && (
                  <div className="ticket-prompt-banner" role="region" aria-label="Unable to answer – raise ticket option">
                    <div className="ticket-prompt-header">
                      <div className="ticket-prompt-icon-badge" aria-hidden="true">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M15 5v2M15 11v2M15 17v2M5 5h14a2 2 0 0 1 2 2v3a2 2 0 0 0 0 4v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3a2 2 0 0 0 0-4V7a2 2 0 0 1 2-2z"></path>
                        </svg>
                      </div>
                      <div className="ticket-prompt-info">
                        <div className="ticket-prompt-title">Need additional assistance?</div>
                        <div className="ticket-prompt-desc">
                          Raise a workplace support ticket and our IT / HR operations team will resolve this within 24 hours.
                        </div>
                      </div>
                    </div>
                    {raisedTickets[msg.id] ? (
                      <div className="ticket-raised-success-badge">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        Ticket Raised: {raisedTickets[msg.id]}
                      </div>
                    ) : (
                      <button
                        className="raise-ticket-btn"
                        onClick={() => {
                          const q = msg.queryText || '';
                          const cat = detectTicketCategory(q);
                          setTicketForm({
                            category: cat,
                            subject: q ? `Inquiry: ${q.slice(0, 200)}` : 'Support Inquiry',
                            description: q
                              ? `Employee Query: ${q}\n\nAssistant Status: Information not found in company knowledge base.\nRequesting support follow-up.`
                              : 'Requesting support for unanswered workplace inquiry.',
                            priority: 'MEDIUM',
                          });
                          setTicketError(null);
                          setTicketModal({ messageId: msg.id });
                        }}
                        aria-label="Open support ticket form"
                      >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                          <path d="M15 5v2M15 11v2M15 17v2M5 5h14a2 2 0 0 1 2 2v3a2 2 0 0 0 0 4v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3a2 2 0 0 0 0-4V7a2 2 0 0 1 2-2z"></path>
                        </svg>
                        Raise Support Ticket
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Quick Prompts Bar ── */}
      <div className="quick-prompts" aria-label="Suggested Workplace Questions">
        <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 700 }}>Suggestions:</span>
        {quickPrompts.map((prompt, i) => (
          <button
            key={i}
            className="prompt-chip"
            onClick={() => handleQuickPrompt(prompt)}
            disabled={voiceState === 'recording' || voiceState === 'processing'}
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* ── Text Input Row (Keyboard Accessibility) ── */}
      <form className="chat-input-row" onSubmit={handleTextSubmit}>
        <input
          type="text"
          className="chat-text-input"
          placeholder="Ask policy question, set reminder ('Remind me tomorrow at 10 AM...'), or type query..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={voiceState === 'processing'}
        />
        <button
          type="submit"
          className="chat-send-btn"
          disabled={!inputText.trim() || voiceState === 'processing'}
          aria-label="Send message"
          title="Send query"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </form>

      {/* ── Voice Control Deck ── */}
      <div className="voice-deck">
        <div className="mic-button-wrapper">
          <button
            className={`mic-button ${voiceState === 'recording' ? 'recording' : ''}`}
            onClick={voiceState === 'recording' ? stopRecording : startRecording}
            disabled={voiceState === 'processing'}
            aria-label={voiceState === 'recording' ? 'Stop voice recording' : 'Start voice recording'}
            title={voiceState === 'recording' ? 'Click to stop recording' : 'Click to start speaking'}
          >
            {voiceState === 'recording' ? (
              <svg width="26" height="26" viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
              </svg>
            )}
          </button>
          <div className="mic-pulse-ring" aria-hidden="true" />
        </div>

        {/* Dynamic Status / Waveform */}
        <div className="deck-hint">
          {voiceState === 'recording' ? (
            <div className="waveform-container" aria-hidden="true">
              <span className="waveform-bar"></span>
              <span className="waveform-bar"></span>
              <span className="waveform-bar"></span>
              <span className="waveform-bar"></span>
              <span className="waveform-bar"></span>
              <span style={{ marginLeft: 8, color: 'var(--record-active)', fontWeight: 600 }}>
                Listening... Click to finish
              </span>
            </div>
          ) : voiceState === 'processing' ? (
            <span>Processing your voice recording...</span>
          ) : voiceState === 'speaking' ? (
            <span style={{ color: 'var(--status-success)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
              </svg>
              Speaking answer...
            </span>
          ) : (
            <span>Click microphone to speak</span>
          )}
        </div>
      </div>
    </div>

    {/* ── Support Ticket Modal ── */}
    {ticketModal && (
      <div
        className="ticket-modal-overlay"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ticket-modal-title"
        onClick={(e) => { if (e.target === e.currentTarget && !submittingTicket) setTicketModal(null); }}
      >
        <div className="ticket-modal-card">
          {/* Header */}
          <div className="ticket-modal-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div className="ticket-modal-icon" aria-hidden="true">🎫</div>
              <div>
                <div id="ticket-modal-title" style={{ fontWeight: 700, fontSize: '16px', color: 'var(--text-primary)' }}>
                  Raise Support Ticket
                </div>
                <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Our team will follow up on your inquiry within 24 hours.
                </div>
              </div>
            </div>
            <button
              className="modal-close-btn"
              onClick={() => { if (!submittingTicket) setTicketModal(null); }}
              aria-label="Close ticket modal"
            >
              ✕
            </button>
          </div>

          {/* Error Banner */}
          {ticketError && (
            <div className="modal-alert-error" role="alert">{ticketError}</div>
          )}

          {/* Form */}
          <form
            className="ticket-modal-form"
            onSubmit={async (e) => {
              e.preventDefault();
              setSubmittingTicket(true);
              setTicketError(null);
              try {
                const created = await createTicket({
                  category: ticketForm.category,
                  subject: ticketForm.subject.trim(),
                  description: ticketForm.description.trim(),
                  priority: ticketForm.priority,
                });
                const ticketNum = created.ticket_number || 'SUBMITTED';
                setRaisedTickets((prev) => ({ ...prev, [ticketModal.messageId]: ticketNum }));
                setTicketModal(null);
                // Append a confirmation message in the conversation
                setMessages((prev) => [
                  ...prev,
                  {
                    id: `ticket-${Date.now()}`,
                    role: 'assistant',
                    text: `Your support ticket has been raised successfully.\nTicket Number: ${ticketNum}\nOur team will follow up within 24 hours.`,
                    timestamp: new Date(),
                    suggestTicket: false,
                  },
                ]);
              } catch (err) {
                setTicketError(err.message || 'Failed to raise ticket. Please try again.');
              } finally {
                setSubmittingTicket(false);
              }
            }}
          >
            <div className="form-row-2col">
              <div className="form-group">
                <label htmlFor="ticket-category">Category</label>
                <select
                  id="ticket-category"
                  className="ticket-form-input"
                  value={ticketForm.category}
                  onChange={(e) => setTicketForm({ ...ticketForm, category: e.target.value })}
                  disabled={submittingTicket}
                >
                  <option value="IT">IT Support</option>
                  <option value="HR">HR</option>
                  <option value="FACILITIES">Facilities</option>
                  <option value="GENERAL">General</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="ticket-priority">Priority</label>
                <select
                  id="ticket-priority"
                  className="ticket-form-input"
                  value={ticketForm.priority}
                  onChange={(e) => setTicketForm({ ...ticketForm, priority: e.target.value })}
                  disabled={submittingTicket}
                >
                  <option value="LOW">Low</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="HIGH">High</option>
                  <option value="URGENT">Urgent</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="ticket-subject">Subject *</label>
              <input
                id="ticket-subject"
                type="text"
                required
                className="ticket-form-input"
                placeholder="Brief summary of your issue or request"
                value={ticketForm.subject}
                onChange={(e) => setTicketForm({ ...ticketForm, subject: e.target.value })}
                disabled={submittingTicket}
                maxLength={255}
              />
            </div>

            <div className="form-group">
              <label htmlFor="ticket-description">Description *</label>
              <textarea
                id="ticket-description"
                required
                className="ticket-form-textarea"
                placeholder="Provide all relevant details (e.g. Tower/Floor, error messages, screenshots)"
                value={ticketForm.description}
                onChange={(e) => setTicketForm({ ...ticketForm, description: e.target.value })}
                disabled={submittingTicket}
                rows={4}
              />
            </div>

            <div className="ticket-modal-actions">
              <button
                type="button"
                className="btn-cancel"
                onClick={() => setTicketModal(null)}
                disabled={submittingTicket}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn-submit-ticket"
                disabled={submittingTicket}
              >
                {submittingTicket ? 'Submitting…' : '🎫 Submit Ticket'}
              </button>
            </div>
          </form>
        </div>
      </div>
    )}
    </>
  );
}
