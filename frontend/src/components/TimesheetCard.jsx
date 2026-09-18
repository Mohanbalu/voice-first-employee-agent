import React from 'react';

export default function TimesheetCard({ timesheet, onEdit }) {
  if (!timesheet) return null;
  return (
    <div className="dashboard-card" style={{ padding: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>
          🗓️ {timesheet.work_date || 'Today'}
        </span>
        <span className={`schedule-badge ${timesheet.status === 'APPROVED' ? 'completed' : 'pending'}`}>
          {timesheet.status || 'DRAFT'}
        </span>
      </div>
      <div style={{ margin: '8px 0', fontSize: '16px', fontWeight: '700', color: 'var(--accent-primary)' }}>
        {timesheet.hours_worked || 8} hrs logged
      </div>
      {timesheet.task_description && (
        <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
          {timesheet.task_description}
        </p>
      )}
    </div>
  );
}
