// src/pages/Dashboard.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './dashboard-react.module.css';

export default function Dashboard() {
  const [explanations, setExplanations] = useState({});
  const [status, setStatus] = useState('loading'); // 'loading' | 'error' | 'ready'
  const navigate = useNavigate();

  const vizPath = '/visualizations/';
  const jsonUrl = '/visualizations/plot_explanations.json';

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(jsonUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const data = await res.json();
        setExplanations(data);
        setStatus('ready');
      } catch (err) {
        console.error('Failed to load explanations JSON:', err);
        setStatus('error');
      }
    })();
  }, []);

  if (status === 'loading') {
    return (
      <div className={styles.container}>
        <div className={styles.grid}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} className={styles.card} style={{ opacity: 0.5 }}>
              <div
                style={{
                  width: '100%',
                  height: '200px',
                  background: 'linear-gradient(90deg, #2a2a3e 25%, #3a3a50 50%, #2a2a3e 75%)',
                  backgroundSize: '200% 100%',
                  animation: 'shimmer 1.5s infinite',
                  borderRadius: '8px',
                }}
              />
              <p className={styles.explanation} style={{ opacity: 0.3 }}>
                Loading visualization...
              </p>
            </div>
          ))}
        </div>
        <style>{`
          @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
          }
        `}</style>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className={styles.container}>
        <p className={styles.error}>Unable to load plot explanations.</p>
      </div>
    );
  }

  const entries = Object.entries(explanations);
  if (entries.length === 0) {
    return (
      <div className={styles.container}>
        <p className={styles.loading}>No visualizations found.</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.grid}>
        {entries.map(([filename, explanation]) => (
          <div key={filename} className={styles.card}>
            <img
              src={`${vizPath}${filename}`}
              alt={filename}
              className={styles.image}
              loading="lazy"
            />
            <p className={styles.explanation}>{explanation}</p>
          </div>
        ))}
      </div>

      {/* Next-stage button → goes to your new Chat page */}
      <div className={styles.nextWrapper}>
        <button
          className={styles.nextBtn}
          onClick={() => navigate('/chat')}
        >
          Go to Chat Stage
        </button>
      </div>
    </div>
  );
}
