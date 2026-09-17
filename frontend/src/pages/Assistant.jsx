import React from 'react';
import VoiceAssistant from '../components/VoiceAssistant';

/**
 * Assistant Page — Module 6.4
 *
 * Dedicated full-screen workplace voice assistant view.
 */
export default function Assistant() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%' }}>
      <div style={{ marginBottom: '4px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
          AI Workplace Assistant
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
          Ask questions naturally using your voice. The assistant queries verified company policies and responds in real time.
        </p>
      </div>

      <VoiceAssistant />
    </div>
  );
}
