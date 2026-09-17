import React, { useState, useEffect, useCallback } from 'react';
import {
  getEmployees,
  createEmployee,
  updateEmployeeStatus,
  getTickets,
  updateTicketStatus,
  answerTicket,
} from '../services/api';

export default function AdminDashboard() {
  // Tabs: 'employees' | 'tickets'
  const [activeTab, setActiveTab] = useState('employees');

  // Employee State
  const [employees, setEmployees] = useState([]);
  const [empLoading, setEmpLoading] = useState(false);
  const [empError, setEmpError] = useState(null);
  const [empSuccess, setEmpSuccess] = useState(null);
  const [empSearch, setEmpSearch] = useState('');

  // Add Employee Modal State
  const [showAddEmpModal, setShowAddEmpModal] = useState(false);
  const [newEmpForm, setNewEmpForm] = useState({
    sap_id: '',
    full_name: '',
    email: '',
    department: 'Engineering',
    password: '',
  });
  const [sapValidationErr, setSapValidationErr] = useState('');
  const [submittingEmp, setSubmittingEmp] = useState(false);

  // Ticket State
  const [tickets, setTickets] = useState([]);
  const [ticketsLoading, setTicketsLoading] = useState(false);
  const [ticketError, setTicketError] = useState(null);
  const [ticketSuccess, setTicketSuccess] = useState(null);
  const [ticketStatusFilter, setTicketStatusFilter] = useState('ALL');

  // HR Answer State
  // answerDraft: { [ticketId]: string }  — tracks textarea content per ticket
  const [answerDraft, setAnswerDraft] = useState({});
  // answerPanelOpen: { [ticketId]: bool } — controls expanded answer panel
  const [answerPanelOpen, setAnswerPanelOpen] = useState({});
  // answerSubmitting: { [ticketId]: bool }
  const [answerSubmitting, setAnswerSubmitting] = useState({});
  // answerKb: { [ticketId]: bool } — per-ticket KB toggle (default true)
  const [answerKb, setAnswerKb] = useState({});

  // Load Employees
  const fetchEmployees = useCallback(async () => {
    setEmpLoading(true);
    setEmpError(null);
    try {
      const data = await getEmployees();
      setEmployees(data || []);
    } catch (err) {
      setEmpError(err.message || 'Failed to load employees');
    } finally {
      setEmpLoading(false);
    }
  }, []);

  // Load Tickets
  const fetchTickets = useCallback(async () => {
    setTicketsLoading(true);
    setTicketError(null);
    try {
      const statusParam = ticketStatusFilter === 'ALL' ? null : ticketStatusFilter;
      const data = await getTickets({ status: statusParam });
      // TicketListResponse returns { total, tickets: [...] }
      setTickets(data.tickets || data.items || []);
    } catch (err) {
      setTicketError(err.message || 'Failed to load tickets');
    } finally {
      setTicketsLoading(false);
    }
  }, [ticketStatusFilter]);

  useEffect(() => {
    fetchEmployees();
  }, [fetchEmployees]);

  useEffect(() => {
    fetchTickets();
  }, [fetchTickets]);

  // Auto-refresh tickets every 30 seconds so HR sees new tickets without reload
  useEffect(() => {
    const interval = setInterval(() => {
      fetchTickets();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchTickets]);

  // Handle SAP ID change with real-time validation
  const handleSapIdChange = (e) => {
    const val = e.target.value.trim();
    setNewEmpForm((prev) => ({ ...prev, sap_id: val }));
    if (!val) {
      setSapValidationErr('');
    } else if (!/^560\d{0,5}$/.test(val)) {
      setSapValidationErr('SAP ID must start with 560 and contain only digits');
    } else if (val.length !== 8) {
      setSapValidationErr(`Must be exactly 8 digits (current: ${val.length}/8)`);
    } else {
      setSapValidationErr('');
    }
  };

  const handleCreateEmployee = async (e) => {
    e.preventDefault();
    if (!/^560\d{5}$/.test(newEmpForm.sap_id)) {
      setSapValidationErr('SAP ID must be exactly 8 digits starting with 560 (e.g. 56031439)');
      return;
    }

    setSubmittingEmp(true);
    setEmpError(null);
    setEmpSuccess(null);
    try {
      const payload = {
        sap_id: newEmpForm.sap_id,
        full_name: newEmpForm.full_name,
        email: newEmpForm.email,
        department: newEmpForm.department,
        password: newEmpForm.password || `HclEmp@${newEmpForm.sap_id}`,
      };
      await createEmployee(payload);
      setEmpSuccess(`Employee ${payload.full_name} (${payload.sap_id}) provisioned successfully!`);
      setShowAddEmpModal(false);
      setNewEmpForm({
        sap_id: '',
        full_name: '',
        email: '',
        department: 'Engineering',
        password: '',
      });
      fetchEmployees();
    } catch (err) {
      setEmpError(err.message || 'Failed to create employee');
    } finally {
      setSubmittingEmp(false);
    }
  };

  const handleToggleEmpStatus = async (emp) => {
    const newStatus = !emp.is_active;
    const confirmMsg = newStatus
      ? `Reactivate employee ${emp.full_name} (${emp.sap_id})?`
      : `Deactivate employee ${emp.full_name} (${emp.sap_id})? User will no longer be able to log in.`;
    if (!window.confirm(confirmMsg)) return;

    try {
      await updateEmployeeStatus(emp.id, newStatus);
      setEmpSuccess(`Updated status for ${emp.full_name} to ${newStatus ? 'Active' : 'Inactive'}`);
      fetchEmployees();
    } catch (err) {
      setEmpError(err.message || 'Failed to update employee status');
    }
  };

  const handleUpdateTicketStatus = async (ticketId, newStatus) => {
    try {
      let notes = undefined;
      if (newStatus === 'RESOLVED' || newStatus === 'CLOSED') {
        notes = window.prompt('Optional resolution notes:', 'Resolved via HR portal.') || undefined;
      }
      await updateTicketStatus(ticketId, newStatus, notes);
      setTicketSuccess(`Ticket updated to ${newStatus}`);
      fetchTickets();
    } catch (err) {
      setTicketError(err.message || 'Failed to update ticket status');
    }
  };

  /** HR submits an official answer → saves on ticket + ingests to RAG KB */
  const handleAnswerTicket = async (ticketId) => {
    const answer = (answerDraft[ticketId] || '').trim();
    if (!answer || answer.length < 10) {
      setTicketError('Answer must be at least 10 characters.');
      return;
    }
    setAnswerSubmitting((prev) => ({ ...prev, [ticketId]: true }));
    setTicketError(null);
    try {
      const ingestToKb = answerKb[ticketId] !== false; // default true
      await answerTicket(ticketId, answer, ingestToKb);
      setTicketSuccess(
        ingestToKb
          ? '✅ Answer submitted and added to the AI knowledge base!'
          : '✅ Answer submitted (not added to knowledge base).'
      );
      // Close panel, clear draft
      setAnswerPanelOpen((prev) => ({ ...prev, [ticketId]: false }));
      setAnswerDraft((prev) => ({ ...prev, [ticketId]: '' }));
      fetchTickets();
    } catch (err) {
      setTicketError(err.message || 'Failed to submit answer.');
    } finally {
      setAnswerSubmitting((prev) => ({ ...prev, [ticketId]: false }));
    }
  };

  // Stats
  const totalEmployees = employees.length;
  const activeEmployees = employees.filter((e) => e.is_active).length;
  const openTicketsCount = tickets.filter((t) => t.status === 'OPEN').length;
  const pendingTicketsCount = tickets.filter((t) => t.status === 'IN_PROGRESS').length;

  const filteredEmployees = employees.filter((emp) => {
    if (!empSearch) return true;
    const s = empSearch.toLowerCase();
    return (
      (emp.sap_id && emp.sap_id.toLowerCase().includes(s)) ||
      (emp.full_name && emp.full_name.toLowerCase().includes(s)) ||
      (emp.email && emp.email.toLowerCase().includes(s)) ||
      (emp.department && emp.department.toLowerCase().includes(s))
    );
  });

  return (
    <div className="admin-container" style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px 16px' }}>
      {/* Header Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '4px' }}>
            HR Administration & Operations
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
            Manage workforce provisioning, SAP ID access control, and workplace service tickets.
          </p>
        </div>
        {activeTab === 'employees' && (
          <button
            onClick={() => setShowAddEmpModal(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 18px',
              backgroundColor: 'var(--accent-primary)',
              color: '#fff',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontWeight: '600',
              cursor: 'pointer',
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            Provision Employee
          </button>
        )}
      </div>

      {/* KPI Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '28px' }}>
        <div className="dashboard-card" style={{ padding: '16px 20px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Total Workforce
          </span>
          <div style={{ fontSize: '28px', fontWeight: '700', color: 'var(--text-primary)', marginTop: '6px' }}>
            {totalEmployees}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Registered employees</span>
        </div>

        <div className="dashboard-card" style={{ padding: '16px 20px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Active Accounts
          </span>
          <div style={{ fontSize: '28px', fontWeight: '700', color: 'var(--status-success)', marginTop: '6px' }}>
            {activeEmployees}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Login-enabled profiles</span>
        </div>

        <div className="dashboard-card" style={{ padding: '16px 20px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Open Service Tickets
          </span>
          <div style={{ fontSize: '28px', fontWeight: '700', color: 'var(--status-error)', marginTop: '6px' }}>
            {openTicketsCount}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Awaiting action</span>
        </div>

        <div className="dashboard-card" style={{ padding: '16px 20px' }}>
          <span style={{ fontSize: '13px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            In Progress Tickets
          </span>
          <div style={{ fontSize: '28px', fontWeight: '700', color: 'var(--status-warning)', marginTop: '6px' }}>
            {pendingTicketsCount}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Currently being handled</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', marginBottom: '20px', gap: '8px' }}>
        <button
          onClick={() => setActiveTab('employees')}
          style={{
            padding: '12px 20px',
            backgroundColor: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'employees' ? '2px solid var(--accent-primary)' : '2px solid transparent',
            color: activeTab === 'employees' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: activeTab === 'employees' ? '600' : '400',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          Employee Management ({employees.length})
        </button>
        <button
          onClick={() => setActiveTab('tickets')}
          style={{
            padding: '12px 20px',
            backgroundColor: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'tickets' ? '2px solid var(--accent-primary)' : '2px solid transparent',
            color: activeTab === 'tickets' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: activeTab === 'tickets' ? '600' : '400',
            cursor: 'pointer',
            fontSize: '15px',
          }}
        >
          Workplace Support Tickets ({tickets.length})
        </button>
      </div>

      {/* Feedback Alerts */}
      {(empError || ticketError) && (
        <div style={{ padding: '12px 16px', backgroundColor: 'rgba(244, 63, 94, 0.15)', border: '1px solid var(--status-error)', borderRadius: 'var(--radius-md)', color: '#fca5a5', marginBottom: '16px' }}>
          {empError || ticketError}
        </div>
      )}
      {(empSuccess || ticketSuccess) && (
        <div style={{ padding: '12px 16px', backgroundColor: 'rgba(16, 185, 129, 0.15)', border: '1px solid var(--status-success)', borderRadius: 'var(--radius-md)', color: '#6ee7b7', marginBottom: '16px' }}>
          {empSuccess || ticketSuccess}
        </div>
      )}

      {/* ===================== TAB 1: EMPLOYEE MANAGEMENT ===================== */}
      {activeTab === 'employees' && (
        <div>
          {/* Search bar */}
          <div style={{ marginBottom: '16px' }}>
            <input
              type="text"
              placeholder="Search by SAP ID, name, email, or department..."
              value={empSearch}
              onChange={(e) => setEmpSearch(e.target.value)}
              style={{
                width: '100%',
                maxWidth: '450px',
                padding: '10px 14px',
                backgroundColor: 'var(--bg-input)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontSize: '14px',
              }}
            />
          </div>

          {empLoading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading employees...</div>
          ) : (
            <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-surface)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>SAP ID</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Full Name</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Email</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Department</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Status</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEmployees.length === 0 ? (
                    <tr>
                      <td colSpan="6" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                        No employees found matching criteria.
                      </td>
                    </tr>
                  ) : (
                    filteredEmployees.map((emp) => (
                      <tr key={emp.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                        <td style={{ padding: '12px 16px', fontFamily: 'monospace', fontWeight: '600', color: 'var(--accent-primary)' }}>
                          {emp.sap_id}
                        </td>
                        <td style={{ padding: '12px 16px', fontWeight: '500', color: 'var(--text-primary)' }}>
                          {emp.full_name}
                        </td>
                        <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>
                          {emp.email}
                        </td>
                        <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>
                          {emp.department || 'General'}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <span
                            style={{
                              padding: '4px 10px',
                              borderRadius: 'var(--radius-full)',
                              fontSize: '12px',
                              fontWeight: '600',
                              backgroundColor: emp.is_active ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                              color: emp.is_active ? 'var(--status-success)' : 'var(--status-error)',
                            }}
                          >
                            {emp.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <button
                            onClick={() => handleToggleEmpStatus(emp)}
                            style={{
                              padding: '6px 12px',
                              borderRadius: 'var(--radius-sm)',
                              border: '1px solid var(--border-subtle)',
                              backgroundColor: 'var(--bg-surface)',
                              color: emp.is_active ? 'var(--status-error)' : 'var(--status-success)',
                              fontSize: '12px',
                              fontWeight: '500',
                              cursor: 'pointer',
                            }}
                          >
                            {emp.is_active ? 'Deactivate' : 'Reactivate'}
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ===================== TAB 2: TICKETS ===================== */}
      {activeTab === 'tickets' && (
        <div>
          {/* Status Filter buttons + Refresh */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
            {['ALL', 'OPEN', 'IN_PROGRESS', 'RESOLVED', 'CLOSED'].map((st) => (
              <button
                key={st}
                onClick={() => setTicketStatusFilter(st)}
                style={{
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-full)',
                  border: '1px solid var(--border-subtle)',
                  backgroundColor: ticketStatusFilter === st ? 'var(--accent-primary)' : 'var(--bg-surface)',
                  color: ticketStatusFilter === st ? '#fff' : 'var(--text-secondary)',
                  fontSize: '12px',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                {st.replace('_', ' ')}
              </button>
            ))}
            <button
              onClick={fetchTickets}
              disabled={ticketsLoading}
              style={{
                marginLeft: 'auto',
                padding: '6px 14px',
                borderRadius: 'var(--radius-full)',
                border: '1px solid var(--border-subtle)',
                backgroundColor: 'var(--bg-surface)',
                color: 'var(--text-secondary)',
                fontSize: '12px',
                fontWeight: '600',
                cursor: ticketsLoading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
              {ticketsLoading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>

          {ticketsLoading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>Loading tickets...</div>
          ) : (
            <div style={{ backgroundColor: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '14px' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-surface)', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Ticket ID</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Raised By</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Subject</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Category</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Priority</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Status</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Update Status</th>
                    <th style={{ padding: '12px 16px', fontWeight: '600' }}>Answer</th>
                  </tr>
                </thead>
                <tbody>
                  {tickets.length === 0 ? (
                    <tr>
                      <td colSpan="8" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                        No tickets found.
                      </td>
                    </tr>
                  ) : (
                    tickets.map((t) => (
                      <React.Fragment key={t.id}>
                        <tr style={{ borderBottom: answerPanelOpen[t.id] ? 'none' : '1px solid var(--border-subtle)' }}>
                        <td style={{ padding: '12px 16px', fontFamily: 'monospace', fontWeight: '600', color: 'var(--accent-primary)', whiteSpace: 'nowrap' }}>
                          {t.ticket_number}
                        </td>
                        {/* Raised By */}
                        <td style={{ padding: '12px 16px', whiteSpace: 'nowrap' }}>
                          <div style={{ fontWeight: '500', color: 'var(--text-primary)', fontSize: '13px' }}>
                            {t.creator_name || 'Employee'}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            {t.created_at ? new Date(t.created_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : ''}
                          </div>
                        </td>
                        <td style={{ padding: '12px 16px', maxWidth: '250px' }}>
                          <div style={{ fontWeight: '500', color: 'var(--text-primary)' }}>{t.subject}</div>
                          <div style={{ fontSize: '12px', color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {t.description}
                          </div>
                        </td>
                        <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>
                          {t.category}
                        </td>
                        <td style={{ padding: '12px 16px' }}>
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
                        <td style={{ padding: '12px 16px' }}>
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
                        <td style={{ padding: '12px 16px' }}>
                          <select
                            value={t.status}
                            onChange={(e) => handleUpdateTicketStatus(t.id, e.target.value)}
                            style={{
                              padding: '6px 10px',
                              backgroundColor: 'var(--bg-input)',
                              border: '1px solid var(--border-subtle)',
                              borderRadius: 'var(--radius-sm)',
                              color: 'var(--text-primary)',
                              fontSize: '12px',
                              cursor: 'pointer',
                            }}
                          >
                            <option value="OPEN">OPEN</option>
                            <option value="IN_PROGRESS">IN_PROGRESS</option>
                            <option value="RESOLVED">RESOLVED</option>
                            <option value="CLOSED">CLOSED</option>
                          </select>
                        </td>
                        {/* Answer action cell */}
                        <td style={{ padding: '8px 16px' }}>
                          {t.hr_answer ? (
                            <span style={{ fontSize: '11px', color: 'var(--status-success)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                              KB Indexed
                            </span>
                          ) : (t.status === 'OPEN' || t.status === 'IN_PROGRESS') ? (
                            <button
                              onClick={() => setAnswerPanelOpen((prev) => ({ ...prev, [t.id]: !prev[t.id] }))}
                              style={{
                                padding: '5px 12px',
                                borderRadius: 'var(--radius-sm)',
                                border: '1px solid rgba(99, 102, 241, 0.5)',
                                backgroundColor: answerPanelOpen[t.id] ? 'rgba(99,102,241,0.15)' : 'transparent',
                                color: 'var(--accent-primary)',
                                fontSize: '12px',
                                fontWeight: '600',
                                cursor: 'pointer',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {answerPanelOpen[t.id] ? '▲ Hide' : '✏️ Answer'}
                            </button>
                          ) : null}
                        </td>
                      </tr>
                      {/* ── Expandable HR Answer Panel ── */}
                      {answerPanelOpen[t.id] && (
                        <tr key={`answer-${t.id}`}>
                          <td colSpan="8" style={{ padding: '0 16px 16px 16px', backgroundColor: 'rgba(99,102,241,0.04)', borderBottom: '1px solid var(--border-subtle)' }}>
                            <div style={{ padding: '14px 16px', backgroundColor: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(99,102,241,0.25)', marginTop: '8px' }}>
                              <div style={{ fontWeight: '600', fontSize: '13px', color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                                Provide HR Answer
                                <span style={{ fontSize: '11px', fontWeight: '400', color: 'var(--text-secondary)', marginLeft: '4px' }}>
                                  for ticket <strong>{t.ticket_number}</strong>
                                </span>
                              </div>

                              {/* Employee's original question context */}
                              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '10px', padding: '8px 12px', backgroundColor: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent-primary)' }}>
                                <strong>Question:</strong> {t.subject}
                                {t.description && t.description !== t.subject && (
                                  <div style={{ marginTop: '4px', opacity: 0.8 }}>{t.description}</div>
                                )}
                              </div>

                              <textarea
                                id={`answer-text-${t.id}`}
                                placeholder="Type the official answer here. This will resolve the ticket and be available to the AI assistant for future similar questions..."
                                value={answerDraft[t.id] || ''}
                                onChange={(e) => setAnswerDraft((prev) => ({ ...prev, [t.id]: e.target.value }))}
                                disabled={answerSubmitting[t.id]}
                                rows={4}
                                style={{
                                  width: '100%',
                                  padding: '10px 12px',
                                  backgroundColor: 'var(--bg-input)',
                                  border: '1px solid var(--border-subtle)',
                                  borderRadius: 'var(--radius-sm)',
                                  color: 'var(--text-primary)',
                                  fontSize: '13px',
                                  resize: 'vertical',
                                  boxSizing: 'border-box',
                                  fontFamily: 'inherit',
                                  lineHeight: '1.5',
                                }}
                              />

                              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '10px', flexWrap: 'wrap', gap: '10px' }}>
                                {/* KB Toggle */}
                                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '12px', color: 'var(--text-secondary)' }}>
                                  <input
                                    type="checkbox"
                                    checked={answerKb[t.id] !== false}
                                    onChange={(e) => setAnswerKb((prev) => ({ ...prev, [t.id]: e.target.checked }))}
                                    disabled={answerSubmitting[t.id]}
                                    style={{ accentColor: 'var(--accent-primary)', width: '14px', height: '14px' }}
                                  />
                                  <span>
                                    <strong style={{ color: 'var(--accent-primary)' }}>Add to AI Knowledge Base</strong>
                                    {' '}— future employees asking the same question will get this answer
                                  </span>
                                </label>

                                {/* Submit */}
                                <button
                                  onClick={() => handleAnswerTicket(t.id)}
                                  disabled={answerSubmitting[t.id] || !(answerDraft[t.id] || '').trim()}
                                  style={{
                                    padding: '8px 20px',
                                    borderRadius: 'var(--radius-sm)',
                                    border: 'none',
                                    backgroundColor: 'var(--accent-primary)',
                                    color: '#fff',
                                    fontSize: '13px',
                                    fontWeight: '700',
                                    cursor: (answerSubmitting[t.id] || !(answerDraft[t.id] || '').trim()) ? 'not-allowed' : 'pointer',
                                    opacity: (answerSubmitting[t.id] || !(answerDraft[t.id] || '').trim()) ? 0.6 : 1,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    whiteSpace: 'nowrap',
                                  }}
                                >
                                  {answerSubmitting[t.id] ? (
                                    'Submitting…'
                                  ) : (
                                    <>
                                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                      Submit Answer {answerKb[t.id] !== false ? '& Publish to KB' : ''}
                                    </>
                                  )}
                                </button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                      </React.Fragment>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ===================== PROVISION EMPLOYEE MODAL ===================== */}
      {showAddEmpModal && (
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
              maxWidth: '520px',
              width: '100%',
              padding: '24px',
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>
                Provision New Employee
              </h2>
              <button
                onClick={() => setShowAddEmpModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '20px' }}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateEmployee}>
              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  SAP ID (Employee ID) *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. 56031499"
                  maxLength={8}
                  value={newEmpForm.sap_id}
                  onChange={handleSapIdChange}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: `1px solid ${sapValidationErr ? 'var(--status-error)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                    fontFamily: 'monospace',
                  }}
                />
                {sapValidationErr ? (
                  <div style={{ color: 'var(--status-error)', fontSize: '12px', marginTop: '4px' }}>
                    {sapValidationErr}
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '4px' }}>
                    Must be exactly 8 digits starting with 560
                  </div>
                )}
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Full Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Ananya Sharma"
                  value={newEmpForm.full_name}
                  onChange={(e) => setNewEmpForm({ ...newEmpForm, full_name: e.target.value })}
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

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Corporate Email *
                </label>
                <input
                  type="email"
                  required
                  placeholder="e.g. ananya.sharma@hcl.com"
                  value={newEmpForm.email}
                  onChange={(e) => setNewEmpForm({ ...newEmpForm, email: e.target.value })}
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

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Department
                </label>
                <select
                  value={newEmpForm.department}
                  onChange={(e) => setNewEmpForm({ ...newEmpForm, department: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                  }}
                >
                  <option value="Engineering">Engineering</option>
                  <option value="IT Operations">IT Operations</option>
                  <option value="Human Resources">Human Resources</option>
                  <option value="Finance">Finance</option>
                  <option value="Facility & Admin">Facility & Admin</option>
                  <option value="Legal">Legal</option>
                </select>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: '500', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Initial Password (Optional)
                </label>
                <input
                  type="password"
                  placeholder="Defaults to HclEmp@{SAP_ID}"
                  value={newEmpForm.password}
                  onChange={(e) => setNewEmpForm({ ...newEmpForm, password: e.target.value })}
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

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  onClick={() => setShowAddEmpModal(false)}
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
                  disabled={submittingEmp || !!sapValidationErr}
                  style={{
                    padding: '10px 20px',
                    backgroundColor: 'var(--accent-primary)',
                    border: 'none',
                    borderRadius: 'var(--radius-md)',
                    color: '#fff',
                    fontWeight: '600',
                    cursor: submittingEmp || sapValidationErr ? 'not-allowed' : 'pointer',
                    opacity: submittingEmp || sapValidationErr ? 0.6 : 1,
                  }}
                >
                  {submittingEmp ? 'Provisioning...' : 'Provision Employee'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
