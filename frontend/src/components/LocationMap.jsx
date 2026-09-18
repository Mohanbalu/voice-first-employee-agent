import React from 'react';

export default function LocationMap({ location }) {
  const campusLocations = [
    { name: 'Tower 1 — Ground Floor', desc: 'Cafeteria, HR Helpdesk, Main Reception', coordinates: '12.9716° N, 80.2458° E' },
    { name: 'Tower 1 — 3rd Floor', desc: 'SDC Engineering Labs, ODC-3, Breakout Zone', coordinates: '12.9718° N, 80.2460° E' },
    { name: 'Tower 2 — 1st Floor', desc: 'Conference Hall B, Recreation Area (Table Tennis, Chess)', coordinates: '12.9721° N, 80.2455° E' },
    { name: 'Tower 2 — 4th Floor', desc: 'IT Asset Provisioning Desk, Hardware Repair', coordinates: '12.9723° N, 80.2452° E' },
  ];

  return (
    <div className="dashboard-card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
        <span style={{ fontSize: '20px' }}>📍</span>
        <h3 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)' }}>
          HCL Workplace Campus Directory
        </h3>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '10px' }}>
        {campusLocations.map((loc, i) => (
          <div key={i} style={{ padding: '10px 12px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: '600', fontSize: '13px', color: 'var(--text-primary)' }}>{loc.name}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>{loc.desc}</div>
            <div style={{ fontSize: '11px', color: 'var(--accent-primary)', marginTop: '4px', fontFamily: 'monospace' }}>
              GPS: {loc.coordinates}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
