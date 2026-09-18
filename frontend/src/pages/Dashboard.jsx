import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { getMyTickets, createTicket } from '../services/api';

export default function Dashboard({ onNavigateToAssistant, onNavigateToSchedules }) {
  const { user } = useAuth();

  // Tickets state
  const [tickets, setTickets] = useState([]);
  const [loadingTickets, setLoadingTickets] = useState(false);
  const [ticketError, setTicketError] = useState(null);
  const [ticketSuccess, setTicketSuccess] = useState(null);

  // New ticket modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [newTicket, setNewTicket] = useState({
    subject: '',
    description: '',
    category: 'IT',
    priority: 'MEDIUM',
  });

  const fetchMyTickets = useCallback(async () => {
    if (!user) return;
    setLoadingTickets(true);
    try {
      const data = await getMyTickets();
      // TicketListResponse returns { total, tickets: [...] }
      setTickets(data.tickets || data.items || []);
    } catch (err) {
      console.error('Failed to load tickets', err);
    } finally {
      setLoadingTickets(false);
    }
  }, [user]);

  useEffect(() => {
    fetchMyTickets();
  }, [fetchMyTickets]);

  const handleCreateTicket = async (e) => {
    e.preventDefault();
    if (!newTicket.subject.trim() || !newTicket.description.trim()) return;

    setSubmitting(true);
    setTicketError(null);
    setTicketSuccess(null);
    try {
      const created = await createTicket(newTicket);
      setTicketSuccess(`Ticket ${created.ticket_number} raised successfully!`);
      setShowCreateModal(false);
      setNewTicket({ subject: '', description: '', category: 'IT', priority: 'MEDIUM' });
      fetchMyTickets();
    } catch (err) {
      setTicketError(err.message || 'Failed to raise ticket');
    } finally {
      setSubmitting(false);
    }
  };

  const cards = [
    {
      title: 'Workplace Time Scheduling',
      desc: 'Natural language time parsing ("in 30 mins", "tomorrow at 10 AM") and recurring task alerts.',
      icon: '⏰',
      badge: 'Scheduling',
      onClick: onNavigateToSchedules,
    },
    {
      title: 'Company Policy Q&A',
      desc: 'Retrieval-Augmented Generation across employee handbooks and policy chunks via pgvector.',
      icon: '📚',
      badge: 'RAG Grounded',
      onClick: onNavigateToAssistant,
    },
    {
      title: 'HR & Leave Management',
      desc: 'Inquire about leave entitlements, casual leaves, carryover limits, and submit inquiries.',
      icon: '🏖️',
      badge: 'Integrated',
      onClick: onNavigateToAssistant,
    },
    {
      title: 'Office Facilities & Navigation',
      desc: 'SDC campus directions, Tower 1 & 2 facilities, cafeteria, breakout rooms, and play areas.',
      icon: '🗺️',
      badge: 'HCL Campus',
      onClick: onNavigateToAssistant,
    },
  ];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
      {/* Hero Banner with Personalized Greeting */}
      <div className="dashboard-hero">
        <div className="hero-text">
          <h1>
            Welcome, {user?.name || 'Employee'}{' '}
            {user?.sap_id && (
              <span style={{ fontSize: '18px', fontWeight: '400', color: 'var(--text-secondary)' }}>
                (SAP ID: {user.sap_id})
              </span>
            )}
          </h1>
          <p>
            Department: <strong>{user?.department || 'Operations'}</strong> | Role:{' '}
            <strong style={{ color: 'var(--accent-primary)' }}>{user?.role || 'EMPLOYEE'}</strong>
            <br />
            Access verified workplace policies, schedule reminders, submit IT/HR requests, and interact hands-free with your AI Agent.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {onNavigateToSchedules && (
            <button
              onClick={onNavigateToSchedules}
              className="command-btn"
              style={{ padding: '10px 18px', background: '#ffffff', border: '1px solid #bfdbfe', color: '#1d4ed8', fontWeight: 600 }}
              title="Open Workplace Schedules"
            >
              ⏰ Schedules
            </button>
          )}
          <button
            className="hero-cta"
            onClick={onNavigateToAssistant}
            aria-label="Open Voice Assistant"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="23"></line>
              <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            <span>Voice Assistant</span>
          </button>
        </div>
      </div>

      {/* Workplace Overview Grid */}
      <div className="dashboard-grid" style={{ marginBottom: '32px' }}>
        {cards.map((card, idx) => (
          <div
            key={idx}
            className="dashboard-card"
            onClick={card.onClick}
            style={{ cursor: card.onClick ? 'pointer' : 'default' }}
            title={card.onClick ? `Open ${card.title}` : undefined}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="card-icon">{card.icon}</div>
              <span className="brand-badge">{card.badge}</span>
            </div>
            <div className="card-title">{card.title}</div>
            <div className="card-desc">{card.desc}</div>
          </div>
        ))}
      </div>

      {/* Employee Support Tickets Section */}
      <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)', padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
              My Workplace Support Tickets
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Track technical, HR, and facility requests created by you or raised on your behalf by the AI Agent.
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 16px',
              backgroundColor: 'var(--accent-primary)',
              color: '#fff',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontWeight: '600',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            Raise New Ticket
          </button>
        </div>

        {ticketSuccess && (
          <div style={{ padding: '10px 14px', backgroundColor: 'rgba(16, 185, 129, 0.15)', border: '1px solid var(--status-success)', borderRadius: 'var(--radius-md)', color: '#6ee7b7', marginBottom: '16px', fontSize: '13px' }}>
            {ticketSuccess}
          </div>
        )}
        {ticketError && (
          <div style={{ padding: '10px 14px', backgroundColor: 'rgba(244, 63, 94, 0.15)', border: '1px solid var(--status-error)', borderRadius: 'var(--radius-md)', color: '#fca5a5', marginBottom: '16px', fontSize: '13px' }}>
            {ticketError}
          </div>
        )}

        {loadingTickets ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading your tickets...</div>
        ) : tickets.length === 0 ? (
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>📋</div>
            <p>No support tickets raised yet.</p>
            <span style={{ fontSize: '12px' }}>You can raise a ticket here or ask the AI Assistant: &quot;Please raise an IT ticket for my monitor&quot;.</span>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Ticket #</th>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Subject</th>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Category</th>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Priority</th>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Status</th>
                  <th style={{ padding: '10px 12px', fontWeight: '600' }}>Created At</th>
                </tr>
              </thead>
              <tbody>
                {tickets.map((t) => (
                  <tr key={t.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '12px', fontFamily: 'monospace', fontWeight: '600', color: 'var(--accent-primary)' }}>
                      {t.ticket_number}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <div style={{ fontWeight: '500', color: 'var(--text-primary)' }}>{t.subject}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t.description}</div>
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>{t.category}</td>
                    <td style={{ padding: '12px' }}>
                      <span
                        style={{
                          padding: '3px 8px',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: '11px',
                          fontWeight: '600',
                          backgroundColor:
                            t.priority === 'URGENT'
                              ? 'rgba(244, 63, 94, 0.2)'
                              : t.priority === 'HIGH'
                              ? 'rgba(245, 158, 11, 0.2)'
                              : 'rgba(59, 130, 246, 0.2)',
                          color:
                            t.priority === 'URGENT'
                              ? 'var(--status-error)'
                              : t.priority === 'HIGH'
                              ? 'var(--status-warning)'
                              : 'var(--accent-primary)',
                        }}
                      >
                        {t.priority}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span
                        style={{
                          padding: '3px 8px',
                          borderRadius: 'var(--radius-full)',
                          fontSize: '11px',
                          fontWeight: '600',
                          backgroundColor:
                            t.status === 'OPEN'
                              ? 'rgba(59, 130, 246, 0.15)'
                              : t.status === 'IN_PROGRESS'
                              ? 'rgba(245, 158, 11, 0.15)'
                              : 'rgba(16, 185, 129, 0.15)',
                          color:
                            t.status === 'OPEN'
                              ? 'var(--accent-primary)'
                              : t.status === 'IN_PROGRESS'
                              ? 'var(--status-warning)'
                              : 'var(--status-success)',
                        }}
                      >
                        {t.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)', fontSize: '12px' }}>
                      {new Date(t.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Raise Ticket Modal */}
      {showCreateModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '16px',
          }}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-lg)',
              maxWidth: '500px',
              width: '100%',
              padding: '24px',
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
                Raise Support Ticket
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '20px' }}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateTicket}>
              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Category *
                </label>
                <select
                  value={newTicket.category}
                  onChange={(e) => setNewTicket({ ...newTicket, category: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="IT">IT Support (Hardware / Network / VPN / Access)</option>
                  <option value="HR">Human Resources (Leave / Benefits / Payroll)</option>
                  <option value="FACILITY">Facility & Admin (ODC Access / Parking / Maintenance)</option>
                  <option value="GENERAL">General Workplace Inquiry</option>
                </select>
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Priority
                </label>
                <select
                  value={newTicket.priority}
                  onChange={(e) => setNewTicket({ ...newTicket, priority: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="LOW">Low</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="HIGH">High</option>
                  <option value="URGENT">Urgent</option>
                </select>
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Subject *
                </label>
                <input
                  type="text"
                  required
                  placeholder="Brief summary of your issue or request"
                  value={newTicket.subject}
                  onChange={(e) => setNewTicket({ ...newTicket, subject: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                  }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Description *
                </label>
                <textarea
                  required
                  rows={4}
                  placeholder="Provide all relevant details (e.g. Tower/Floor, error messages, asset tags)"
                  value={newTicket.description}
                  onChange={(e) => setNewTicket({ ...newTicket, description: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                    fontFamily: 'inherit',
                  }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  style={{
                    padding: '10px 16px',
                    backgroundColor: 'transparent',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: 'var(--accent-primary)',
                    border: 'none',
                    borderRadius: 'var(--radius-md)',
                    color: '#fff',
                    fontWeight: '600',
                    cursor: submitting ? 'not-allowed' : 'pointer',
                  }}
                >
                  {submitting ? 'Submitting...' : 'Submit Ticket'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
