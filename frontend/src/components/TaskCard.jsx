import React from 'react';

export default function TaskCard({ task, onToggleStatus }) {
  if (!task) return null;
  return (
    <div className="dashboard-card" style={{ padding: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{
          fontSize: '11px',
          fontWeight: '700',
          padding: '2px 8px',
          borderRadius: 'var(--radius-full)',
          backgroundColor: task.priority === 'HIGH' ? '#fee2e2' : '#eff6ff',
          color: task.priority === 'HIGH' ? '#dc2626' : '#2563eb',
        }}>
          {task.priority || 'MEDIUM'}
        </span>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {task.due_date ? new Date(task.due_date).toLocaleDateString() : 'No due date'}
        </span>
      </div>
      <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-primary)', margin: '8px 0 4px' }}>
        {task.title}
      </h4>
      {task.description && (
        <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
          {task.description}
        </p>
      )}
      {onToggleStatus && (
        <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={() => onToggleStatus(task.id)}
            className="schedule-action-btn complete"
          >
            {task.status === 'COMPLETED' ? '✓ Done' : 'Mark Done'}
          </button>
        </div>
      )}
    </div>
  );
}
