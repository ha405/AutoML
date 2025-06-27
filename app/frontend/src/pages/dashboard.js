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
    return <p className={styles.loading}>Loading visualizations…</p>;
  }
  if (status === 'error') {
    return <p className={styles.error}>Unable to load plot explanations.</p>;
  }

  const entries = Object.entries(explanations);
  if (entries.length === 0) {
    return <p className={styles.loading}>No explanations found.</p>;
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
