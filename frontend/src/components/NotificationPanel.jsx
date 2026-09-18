import React from 'react';

export default function NotificationPanel({ notifications = [], onDismiss }) {
  if (!notifications.length) {
    return (
      <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
        No pending notifications
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {notifications.map((n, i) => (
        <div key={n.id || i} className="alert-box" style={{ margin: '4px 0', background: '#eff6ff', border: '1px solid #bfdbfe', color: '#1e3a8a' }}>
          <div>
            <div style={{ fontWeight: '600', fontSize: '13px' }}>{n.title}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{n.message}</div>
          </div>
          {onDismiss && (
            <button className="alert-dismiss" onClick={() => onDismiss(n.id || i)}>✕</button>
          )}
        </div>
      ))}
    </div>
  );
}
