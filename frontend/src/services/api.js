/**
 * API Service Module — Module 6.4 (React Voice UI).
 *
 * Provides typed HTTP communication with backend voice and agent endpoints:
 *   POST /api/voice/agent (multipart/form-data)
 *   POST /api/voice/synthesize (application/json)
 *   POST /api/agent (application/json fallback)
 */

/**
 * Normalizes backend base URL ensuring clean, canonical API endpoints
 * without duplicate prefixes, trailing slashes, or path fragments.
 */
function resolveApiBaseUrl() {
  const rawUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  if (!rawUrl) return '';
  return rawUrl
    .trim()
    .replace(/\/+$/, '')
    .replace(/\/api\/v1\/?$/, '')
    .replace(/\/v1\/?$/, '')
    .replace(/\/api\/?$/, '');
}

const API_BASE_URL = resolveApiBaseUrl();

const DEV_TENANT_ID = '00000000-0000-0000-0000-000000000001';

/**
 * Sends a recorded audio blob to the STT -> LangGraph Voice Agent endpoint.
 *
 * @param {Blob} audioBlob - Binary audio blob from MediaRecorder
 * @param {Object} [options] - Optional parameters
 * @param {string} [options.tenantId] - Tenant UUID
 * @param {string} [options.language] - Optional ISO language code
 * @param {string} [options.prompt] - Optional context prompt
 * @param {string} [options.conversationId] - Optional session ID
 * @param {string} [options.filename] - Custom audio filename (e.g. recording.webm)
 * @returns {Promise<Object>} VoiceAgentResponse JSON payload
 */
export async function sendVoiceAgentAudio(audioBlob, options = {}) {
  const formData = new FormData();
  const filename = options.filename || 'recording.webm';

  formData.append('audio', audioBlob, filename);
  formData.append('tenant_id', options.tenantId || DEV_TENANT_ID);

  if (options.language) {
    formData.append('language', options.language);
  }
  if (options.prompt) {
    formData.append('prompt', options.prompt);
  }
  if (options.conversationId) {
    formData.append('conversation_id', options.conversationId);
  }

  const endpoint = `${API_BASE_URL}/api/voice/agent`;
  if (import.meta.env.DEV) {
    console.debug(`[API Service] Calling: ${endpoint}`);
  }

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      const errorMsg = response.status === 404
        ? `API endpoint not found (404) at ${endpoint}. Please check backend routes.`
        : (data?.detail || data?.error || `Voice agent failed with status ${response.status}`);
      const err = new Error(errorMsg);
      err.status = response.status;
      err.data = data;
      throw err;
    }

    return data;
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to the backend server. Please verify the API is running on port 8000.');
    }
    throw error;
  }
}

/**
 * Synthesizes text into speech audio bytes.
 *
 * @param {string} text - The text string to synthesize
 * @param {Object} [options] - Optional synthesis parameters
 * @param {string} [options.voice] - Voice persona (autumn, diana, etc.)
 * @param {string} [options.format] - Audio format (default: 'wav')
 * @param {number} [options.speed] - Speech rate multiplier (e.g. 1.0)
 * @returns {Promise<Blob>} Binary audio Blob (audio/wav)
 */
export async function synthesizeSpeech(text, options = {}) {
  if (!text || !text.trim()) {
    throw new Error('Text must not be empty.');
  }

  const payload = {
    text: text.trim(),
    voice: options.voice || 'autumn',
    format: options.format || 'wav',
  };

  if (options.speed) {
    payload.speed = options.speed;
  }

  const endpoint = `${API_BASE_URL}/api/voice/synthesize`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let errorDetail = `TTS failed with status ${response.status}`;
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorJson.error || errorDetail;
      } catch (_) {
        // Fall back to status text if response is not JSON
      }
      const err = new Error(errorDetail);
      err.status = response.status;
      throw err;
    }

    return await response.blob();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Unable to connect to TTS backend service.');
    }
    throw error;
  }
}

/**
 * Sends a text request directly to the LangGraph Agent endpoint (text fallback).
 *
 * @param {string} requestText - User question
 * @param {string} [tenantId] - Tenant UUID
 * @returns {Promise<Object>} Agent response object
 */
export async function sendTextAgentRequest(requestText, tenantId = DEV_TENANT_ID) {
  const endpoint = `${API_BASE_URL}/api/agent`;
  if (import.meta.env.DEV) {
    console.debug(`[API Service] Calling: ${endpoint}`);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: getAuthHeaders({
        'Content-Type': 'application/json',
      }),
      body: JSON.stringify({
        request: requestText,
        tenant_id: tenantId,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) {
      const errorMsg = response.status === 404
        ? `API endpoint not found (404) at ${endpoint}. Please check backend routes.`
        : (data?.detail || data?.error || `Agent request failed with status ${response.status}`);
      throw new Error(errorMsg);
    }
    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('The assistant took too long to respond (30s). Please try again.');
    }
    throw error;
  }
}

// ── Authentication & Token Helpers ──────────────────────────────────────────

export function getStoredToken() {
  return localStorage.getItem('auth_token') || '';
}

function getAuthHeaders(extra = {}) {
  const token = getStoredToken();
  const headers = { ...extra };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export async function loginUser(username, password) {
  const endpoint = `${API_BASE_URL}/api/auth/login`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Authentication failed. Please check your credentials.');
  }
  return data;
}

export async function getCurrentUser() {
  const endpoint = `${API_BASE_URL}/api/auth/me`;
  const response = await fetch(endpoint, {
    method: 'GET',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to retrieve profile.');
  }
  return data;
}

// ── HR Employee Management ──────────────────────────────────────────────────

export async function getEmployees() {
  const endpoint = `${API_BASE_URL}/api/hr/employees`;
  const response = await fetch(endpoint, {
    method: 'GET',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to fetch employees list.');
  }
  return data;
}

export async function createEmployee(employeeData) {
  const endpoint = `${API_BASE_URL}/api/hr/employees`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(employeeData),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to provision employee.');
  }
  return data;
}

export async function updateEmployeeStatus(employeeId, is_active) {
  const endpoint = `${API_BASE_URL}/api/hr/employees/${employeeId}/status`;
  const response = await fetch(endpoint, {
    method: 'PATCH',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ is_active }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to update employee status.');
  }
  return data;
}

// ── Ticket System ───────────────────────────────────────────────────────────

export async function createTicket(ticketData) {
  const endpoint = `${API_BASE_URL}/api/tickets`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(ticketData),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to create support ticket.');
  }
  return data;
}

export async function getMyTickets() {
  const endpoint = `${API_BASE_URL}/api/tickets/me`;
  const response = await fetch(endpoint, {
    method: 'GET',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to load tickets.');
  }
  return data;
}

export async function getTickets(statusFilter = '') {
  // Accept either a plain string or { status: 'OPEN' } object shape
  const statusValue = (typeof statusFilter === 'object' && statusFilter !== null)
    ? statusFilter.status
    : statusFilter;
  const query = statusValue ? `?status=${encodeURIComponent(statusValue)}` : '';
  const endpoint = `${API_BASE_URL}/api/tickets${query}`;
  const response = await fetch(endpoint, {
    method: 'GET',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to fetch organizational tickets.');
  }
  return data;
}

export async function updateTicketStatus(ticketId, status, resolution_notes = '') {
  const endpoint = `${API_BASE_URL}/api/tickets/${ticketId}/status`;
  const response = await fetch(endpoint, {
    method: 'PATCH',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ status, resolution_notes }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to update ticket status.');
  }
  return data;
}

/**
 * HR submits an official answer for an open ticket.
 * The answer is saved on the ticket AND ingested into the RAG
 * knowledge base so the AI assistant can use it for future queries.
 *
 * @param {string} ticketId - UUID of the ticket
 * @param {string} answer   - HR's official answer text
 * @param {boolean} [ingestToKb=true] - Whether to add to RAG knowledge base
 */
export async function answerTicket(ticketId, answer, ingestToKb = true) {
  const endpoint = `${API_BASE_URL}/api/tickets/${ticketId}/answer`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ answer, ingest_to_kb: ingestToKb }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to submit answer.');
  }
  return data;
}

// ── Scheduling & Reminders ──────────────────────────────────────────────────

export async function getSchedules(params = {}) {
  const queryParts = [];
  if (params.status) queryParts.push(`status=${encodeURIComponent(params.status)}`);
  if (params.upcoming_only) queryParts.push('upcoming_only=true');
  if (params.today_only) queryParts.push('today_only=true');
  if (params.limit) queryParts.push(`limit=${params.limit}`);
  if (params.offset) queryParts.push(`offset=${params.offset}`);

  const qs = queryParts.length > 0 ? `?${queryParts.join('&')}` : '';
  const endpoint = `${API_BASE_URL}/api/schedules${qs}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail || 'Failed to retrieve schedules.');
    }
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      console.warn('[getSchedules] Request timed out after 8s, returning empty fallback.');
      return { total: 0, items: [] };
    }
    throw err;
  }
}

export async function createSchedule(scheduleData) {
  const endpoint = `${API_BASE_URL}/api/schedules`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(scheduleData),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to create schedule.');
  }
  return data;
}

export async function updateSchedule(scheduleId, updateData) {
  const endpoint = `${API_BASE_URL}/api/schedules/${scheduleId}`;
  const response = await fetch(endpoint, {
    method: 'PATCH',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(updateData),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to update schedule.');
  }
  return data;
}

export async function completeSchedule(scheduleId) {
  const endpoint = `${API_BASE_URL}/api/schedules/${scheduleId}/complete`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to complete schedule.');
  }
  return data;
}

export async function cancelSchedule(scheduleId) {
  const endpoint = `${API_BASE_URL}/api/schedules/${scheduleId}/cancel`;
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail || 'Failed to cancel schedule.');
  }
  return data;
}

export async function deleteSchedule(scheduleId) {
  const endpoint = `${API_BASE_URL}/api/schedules/${scheduleId}`;
  const response = await fetch(endpoint, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  if (!response.ok && response.status !== 204) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data?.detail || 'Failed to delete schedule.');
  }
  return true;
}

