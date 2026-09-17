import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import Dashboard from './pages/Dashboard';
import Assistant from './pages/Assistant';
import AdminDashboard from './admin/AdminDashboard';
import Login from './pages/Login';

function AppContent() {
  const { user, loading, logout, isAuthenticated } = useAuth();
  const [currentView, setCurrentView] = useState('dashboard');

  // Sync initial view from window URL path or hash
  useEffect(() => {
    const handleLocationChange = () => {
      const path = window.location.pathname.toLowerCase();
      const hash = window.location.hash.toLowerCase();
      if (path.includes('admin') || hash.includes('admin')) {
        setCurrentView('admin');
      } else if (path.includes('assistant') || hash.includes('assistant')) {
        setCurrentView('assistant');
      } else {
        setCurrentView('dashboard');
      }
    };

    handleLocationChange();
    window.addEventListener('popstate', handleLocationChange);
    return () => window.removeEventListener('popstate', handleLocationChange);
  }, []);

  const navigateTo = (view) => {
    setCurrentView(view);
    const targetUrl =
      view === 'admin' ? '/admin' : view === 'assistant' ? '/assistant' : '/dashboard';
    window.history.pushState({}, '', targetUrl);
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', color: 'var(--text-secondary)' }}>
        Authenticating session...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Login onLoginSuccess={(target) => navigateTo(target)} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* ── Top Navigation Bar ── */}
      <header className="app-header" role="banner">
        <div className="brand-container">
          <div className="brand-logo" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="23"></line>
              <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <span className="brand-title">Employee AI Assistant</span>
            <span className="brand-badge">SaaS v0.7.0</span>
          </div>
        </div>

        <nav className="nav-links" role="navigation" aria-label="Main Navigation">
          <button
            className={`nav-button ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => navigateTo('dashboard')}
            aria-current={currentView === 'dashboard' ? 'page' : undefined}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7"></rect>
              <rect x="14" y="3" width="7" height="7"></rect>
              <rect x="14" y="14" width="7" height="7"></rect>
              <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
            <span>Dashboard</span>
          </button>

          <button
            className={`nav-button ${currentView === 'assistant' ? 'active' : ''}`}
            onClick={() => navigateTo('assistant')}
            aria-current={currentView === 'assistant' ? 'page' : undefined}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="23"></line>
              <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            <span>Voice Assistant</span>
          </button>

          {user?.role === 'HR' && (
            <button
              className={`nav-button ${currentView === 'admin' ? 'active' : ''}`}
              onClick={() => navigateTo('admin')}
              aria-current={currentView === 'admin' ? 'page' : undefined}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
              </svg>
              <span>HR Admin</span>
            </button>
          )}
        </nav>

        {/* User Badge & Logout */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginLeft: 'auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', fontSize: '12px' }}>
            <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
              {user?.name || user?.username}
            </span>
            <span
              style={{
                fontSize: '10px',
                fontWeight: '700',
                padding: '1px 6px',
                borderRadius: 'var(--radius-full)',
                backgroundColor: user?.role === 'HR' ? 'rgba(168, 85, 247, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                color: user?.role === 'HR' ? '#c084fc' : 'var(--accent-primary)',
              }}
            >
              {user?.role} {user?.sap_id ? `• ${user.sap_id}` : ''}
            </span>
          </div>

          <button
            onClick={logout}
            style={{
              padding: '6px 12px',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: '500',
              cursor: 'pointer',
            }}
            title="Sign out of system"
          >
            Logout
          </button>
        </div>
      </header>

      {/* ── Main View Container ── */}
      <main className="main-content" role="main">
        {currentView === 'dashboard' ? (
          <Dashboard onNavigateToAssistant={() => navigateTo('assistant')} />
        ) : currentView === 'assistant' ? (
          <Assistant />
        ) : currentView === 'admin' ? (
          user?.role === 'HR' ? (
            <AdminDashboard />
          ) : (
            <div style={{ maxWidth: '600px', margin: '60px auto', padding: '32px', backgroundColor: 'var(--bg-card)', border: '1px solid var(--status-error)', borderRadius: 'var(--radius-lg)', textAlign: 'center' }}>
              <div style={{ fontSize: '40px', marginBottom: '16px' }}>🚫</div>
              <h2 style={{ color: 'var(--status-error)', marginBottom: '12px' }}>403 Access Denied</h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '24px', fontSize: '14px' }}>
                You do not have HR Administrator privileges to view this section. This incident is recorded in audit logs.
              </p>
              <button
                onClick={() => navigateTo('dashboard')}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                Return to Dashboard
              </button>
            </div>
          )
        ) : null}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
