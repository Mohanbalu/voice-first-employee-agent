import React from 'react';

export default function MeetingCard({ meeting, onJoin }) {
  if (!meeting) return null;
  return (
    <div className="dashboard-card" style={{ padding: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{
          fontSize: '11px',
          fontWeight: '700',
          padding: '2px 8px',
          borderRadius: 'var(--radius-full)',
          backgroundColor: '#eff6ff',
          color: '#2563eb',
        }}>
          📅 {meeting.time || 'Upcoming'}
        </span>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {meeting.room || 'Conference Room A'}
        </span>
      </div>
      <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', margin: '8px 0 4px' }}>
        {meeting.title}
      </h4>
      {meeting.agenda && (
        <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
          {meeting.agenda}
        </p>
      )}
      {onJoin && (
        <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={() => onJoin(meeting.id)}
            className="hero-cta"
            style={{ padding: '6px 12px', fontSize: '12px' }}
          >
            Join Meeting
          </button>
        </div>
      )}
    </div>
  );
}
