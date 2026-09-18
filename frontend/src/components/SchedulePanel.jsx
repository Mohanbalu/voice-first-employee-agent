import React, { useState, useEffect, useCallback } from 'react';
import {
  getSchedules,
  createSchedule,
  completeSchedule,
  cancelSchedule,
  deleteSchedule,
} from '../services/api';

/**
 * Formats an ISO date string into human-friendly relative and absolute time.
 */
function formatScheduleTime(isoString) {
  if (!isoString) return 'Unspecified';
  try {
    const d = new Date(isoString);
    const now = new Date();
    const diffMs = d.getTime() - now.getTime();
    const diffMins = Math.round(diffMs / (1000 * 60));
    const diffHours = Math.round(diffMins / 60);
    const diffDays = Math.round(diffHours / 24);

    let relative = '';
    if (Math.abs(diffMins) < 2) {
      relative = 'Just now';
    } else if (diffMins > 0 && diffMins < 60) {
      relative = `in ${diffMins} min${diffMins === 1 ? '' : 's'}`;
    } else if (diffMins > 0 && diffHours < 24) {
      relative = `in ${diffHours} hr${diffHours === 1 ? '' : 's'}`;
    } else if (diffDays === 1) {
      relative = 'Tomorrow';
    } else if (diffDays > 1 && diffDays < 7) {
      relative = `in ${diffDays} days`;
    } else if (diffMins < 0 && Math.abs(diffHours) < 24) {
      relative = `${Math.abs(diffHours)} hr${Math.abs(diffHours) === 1 ? '' : 's'} ago`;
    }

    const formatted = d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });

    return relative ? `${formatted} (${relative})` : formatted;
  } catch {
    return isoString;
  }
}

export default function SchedulePanel({ onOpenVoiceAssistant }) {
  const [schedules, setSchedules] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [activeTab, setActiveTab] = useState('upcoming'); // 'upcoming' | 'today' | 'recurring' | 'completed'

  // New Schedule Modal
  const [showModal, setShowModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [newForm, setNewForm] = useState({
    title: '',
    description: '',
    scheduled_at: '',
    recurrence_type: 'NONE',
    reminder_type: 'NOTIFICATION',
    duration_minutes: 30,
  });

  const fetchSchedules = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSchedules();
      const list = data?.schedules || data?.items || [];
      setSchedules(list);
    } catch (err) {
      console.warn('[SchedulePanel] fetchSchedules error:', err);
      setError(err.message || 'Failed to load schedules.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSchedules();

    // Auto-refresh when schedule is created via voice or text
    const handleScheduleCreated = () => {
      fetchSchedules();
    };
    window.addEventListener('schedule-created', handleScheduleCreated);
    return () => window.removeEventListener('schedule-created', handleScheduleCreated);
  }, [fetchSchedules]);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newForm.title.trim() || !newForm.scheduled_at) {
      setError('Please provide a title and date/time.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const payload = {
        title: newForm.title.trim(),
        description: newForm.description.trim() || null,
        scheduled_at: new Date(newForm.scheduled_at).toISOString(),
        recurrence_type: newForm.recurrence_type,
        reminder_type: newForm.reminder_type,
        duration_minutes: Number(newForm.duration_minutes) || 30,
      };
      await createSchedule(payload);
      setSuccess('Reminder scheduled successfully!');
      setShowModal(false);
      setNewForm({
        title: '',
        description: '',
        scheduled_at: '',
        recurrence_type: 'NONE',
        reminder_type: 'NOTIFICATION',
        duration_minutes: 30,
      });
      fetchSchedules();
    } catch (err) {
      setError(err.message || 'Failed to create schedule.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleComplete = async (id) => {
    try {
      await completeSchedule(id);
      setSuccess('Marked schedule as complete.');
      fetchSchedules();
    } catch (err) {
      setError(err.message || 'Failed to complete schedule.');
    }
  };

  const handleCancel = async (id) => {
    try {
      await cancelSchedule(id);
      setSuccess('Schedule cancelled.');
      fetchSchedules();
    } catch (err) {
      setError(err.message || 'Failed to cancel schedule.');
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this reminder?')) return;
    try {
      await deleteSchedule(id);
      setSuccess('Schedule deleted.');
      fetchSchedules();
    } catch (err) {
      setError(err.message || 'Failed to delete schedule.');
    }
  };

  // Filter items based on active tab
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const todayEnd = todayStart + 24 * 60 * 60 * 1000;

  const filteredSchedules = schedules.filter((s) => {
    const sTime = new Date(s.scheduled_at).getTime();
    if (activeTab === 'upcoming') {
      return s.status === 'PENDING' && sTime >= Date.now() - 3600000;
    }
    if (activeTab === 'today') {
      return sTime >= todayStart && sTime < todayEnd;
    }
    if (activeTab === 'recurring') {
      return s.recurrence_type && s.recurrence_type !== 'NONE';
    }
    if (activeTab === 'completed') {
      return s.status === 'COMPLETED' || s.status === 'CANCELLED';
    }
    return true;
  });

  return (
    <div className="schedules-container" role="region" aria-label="Schedules and Time Reminders">
      {/* Header & Actions */}
      <div className="schedules-header">
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
            Workplace Schedules & Time Reminders
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)' }}>
            Track scheduled tasks, meeting reminders, recurring deadlines, and voice-prompted alerts.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={() => fetchSchedules()}
            className="command-btn"
            style={{ padding: '8px 12px' }}
            title="Refresh schedules list"
            disabled={loading}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }}
            >
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
            </svg>
            <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
          </button>

          {onOpenVoiceAssistant && (
            <button
              onClick={onOpenVoiceAssistant}
              className="command-btn"
              style={{ padding: '8px 14px' }}
              title="Schedule using natural voice or text command"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              </svg>
              <span>Speak Reminder</span>
            </button>
          )}

          <button
            onClick={() => {
              // Pre-set scheduled_at to 1 hour from now
              const d = new Date(Date.now() + 60 * 60 * 1000);
              const pad = (n) => String(n).padStart(2, '0');
              const localIso = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
              setNewForm((prev) => ({ ...prev, scheduled_at: localIso }));
              setShowModal(true);
            }}
            className="hero-cta"
            style={{ padding: '8px 16px', fontSize: '13px' }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Reminder</span>
          </button>
        </div>
      </div>

      {/* Status Banners */}
      {error && (
        <div className="alert-box alert-error" role="alert">
          <div>{error}</div>
          <button className="alert-dismiss" onClick={() => setError(null)}>✕</button>
        </div>
      )}
      {success && (
        <div className="alert-box" style={{ background: '#ecfdf5', border: '1px solid #a7f3d0', color: '#065f46' }}>
          <div>{success}</div>
          <button className="alert-dismiss" onClick={() => setSuccess(null)}>✕</button>
        </div>
      )}

      {/* Tabs */}
      <div className="schedules-tabs" role="tablist">
        <button
          className={`tab-btn ${activeTab === 'upcoming' ? 'active' : ''}`}
          onClick={() => setActiveTab('upcoming')}
          role="tab"
          aria-selected={activeTab === 'upcoming'}
        >
          Upcoming ({schedules.filter((s) => s.status === 'PENDING').length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'today' ? 'active' : ''}`}
          onClick={() => setActiveTab('today')}
          role="tab"
          aria-selected={activeTab === 'today'}
        >
          Today
        </button>
        <button
          className={`tab-btn ${activeTab === 'recurring' ? 'active' : ''}`}
          onClick={() => setActiveTab('recurring')}
          role="tab"
          aria-selected={activeTab === 'recurring'}
        >
          Recurring ({schedules.filter((s) => s.recurrence_type && s.recurrence_type !== 'NONE').length})
        </button>
        <button
          className={`tab-btn ${activeTab === 'completed' ? 'active' : ''}`}
          onClick={() => setActiveTab('completed')}
          role="tab"
          aria-selected={activeTab === 'completed'}
        >
          Completed / Past
        </button>
      </div>

      {/* Schedules Grid */}
      {loading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
          Loading your workplace schedules...
        </div>
      ) : filteredSchedules.length === 0 ? (
        <div className="empty-state" style={{ padding: '60px 20px', background: '#ffffff', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
          <div className="empty-state-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="16" y1="2" x2="16" y2="6"></line>
              <line x1="8" y1="2" x2="8" y2="6"></line>
              <line x1="3" y1="10" x2="21" y2="10"></line>
            </svg>
          </div>
          <h3>No {activeTab} schedules found</h3>
          <p>
            You have no items in this view. You can create a new reminder with the button above or tell the Voice Assistant: "Remind me tomorrow at 10 AM to submit timesheet".
          </p>
        </div>
      ) : (
        <div className="schedules-grid">
          {filteredSchedules.map((item) => (
            <div key={item.id} className="schedule-card">
              <div className="schedule-card-top">
                <div style={{ flex: 1 }}>
                  <div className="schedule-title">{item.title}</div>
                  {item.description && (
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      {item.description}
                    </p>
                  )}
                </div>
                <span className={`schedule-badge ${item.status.toLowerCase()}`}>
                  {item.status}
                </span>
              </div>

              {/* Time & Recurrence Info */}
              <div className="schedule-time-row">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span>{formatScheduleTime(item.scheduled_at)}</span>
              </div>

              {/* Badges: Recurrence & Alert Type */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                {item.recurrence_type && item.recurrence_type !== 'NONE' && (
                  <span className="schedule-badge recurring">
                    🔄 {item.recurrence_type}
                  </span>
                )}
                <span className="schedule-badge" style={{ background: '#f8fafc', border: '1px solid #cbd5e1', color: '#475569' }}>
                  🔔 {item.reminder_type || 'NOTIFICATION'}
                </span>
                {item.duration_minutes && (
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    ({item.duration_minutes} mins)
                  </span>
                )}
              </div>

              {/* Actions */}
              <div className="schedule-card-actions">
                {item.status === 'PENDING' && (
                  <>
                    <button
                      className="schedule-action-btn complete"
                      onClick={() => handleComplete(item.id)}
                      title="Mark as Done"
                    >
                      ✓ Complete
                    </button>
                    <button
                      className="schedule-action-btn cancel"
                      onClick={() => handleCancel(item.id)}
                      title="Cancel this reminder"
                    >
                      Cancel
                    </button>
                  </>
                )}
                <button
                  className="schedule-action-btn delete"
                  onClick={() => handleDelete(item.id)}
                  title="Permanently remove"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── New Reminder Modal ── */}
      {showModal && (
        <div
          className="ticket-modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => { if (e.target === e.currentTarget && !submitting) setShowModal(false); }}
        >
          <div className="ticket-modal-card">
            <div className="ticket-modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div className="ticket-modal-icon">⏰</div>
                <div>
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
                    Schedule Workplace Reminder
                  </h3>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                    Set a time-based task, meeting alert, or recurring workplace reminder.
                  </p>
                </div>
              </div>
              <button
                className="modal-close-btn"
                onClick={() => { if (!submitting) setShowModal(false); }}
              >
                ✕
              </button>
            </div>

            <form className="ticket-modal-form" onSubmit={handleCreate}>
              <div className="form-group">
                <label>Title / Task Description *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Review Q3 deliverable with manager"
                  value={newForm.title}
                  onChange={(e) => setNewForm({ ...newForm, title: e.target.value })}
                  className="ticket-form-input"
                  disabled={submitting}
                />
              </div>

              <div className="form-row-2col">
                <div className="form-group">
                  <label>Scheduled Date & Time *</label>
                  <input
                    type="datetime-local"
                    required
                    value={newForm.scheduled_at}
                    onChange={(e) => setNewForm({ ...newForm, scheduled_at: e.target.value })}
                    className="ticket-form-input"
                    disabled={submitting}
                  />
                </div>
                <div className="form-group">
                  <label>Duration (Minutes)</label>
                  <input
                    type="number"
                    min="5"
                    max="480"
                    value={newForm.duration_minutes}
                    onChange={(e) => setNewForm({ ...newForm, duration_minutes: e.target.value })}
                    className="ticket-form-input"
                    disabled={submitting}
                  />
                </div>
              </div>

              <div className="form-row-2col">
                <div className="form-group">
                  <label>Recurrence</label>
                  <select
                    value={newForm.recurrence_type}
                    onChange={(e) => setNewForm({ ...newForm, recurrence_type: e.target.value })}
                    className="ticket-form-input"
                    disabled={submitting}
                  >
                    <option value="NONE">Does not repeat (One-time)</option>
                    <option value="DAILY">Daily</option>
                    <option value="WEEKLY">Weekly</option>
                    <option value="WEEKDAYS">Every Weekday (Mon–Fri)</option>
                    <option value="MONTHLY">Monthly</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Reminder Type</label>
                  <select
                    value={newForm.reminder_type}
                    onChange={(e) => setNewForm({ ...newForm, reminder_type: e.target.value })}
                    className="ticket-form-input"
                    disabled={submitting}
                  >
                    <option value="NOTIFICATION">Notification Banner</option>
                    <option value="VOICE">Voice Announcement</option>
                    <option value="POPUP">Modal Pop-up</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label>Additional Notes / Agenda</label>
                <textarea
                  rows={3}
                  placeholder="Optional details, links, or attendees"
                  value={newForm.description}
                  onChange={(e) => setNewForm({ ...newForm, description: e.target.value })}
                  className="ticket-form-textarea"
                  disabled={submitting}
                />
              </div>

              <div className="ticket-modal-actions">
                <button
                  type="button"
                  className="btn-cancel"
                  onClick={() => setShowModal(false)}
                  disabled={submitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-submit-ticket"
                  disabled={submitting}
                >
                  {submitting ? 'Scheduling...' : 'Save Reminder'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
