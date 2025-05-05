import React from 'react';
import Conversation from './Conversation';      // adjust path as needed
import Dashboard from './dashboard';            // adjust path as needed
import styles from './AppLayout.module.css';

export default function AppLayout() {
  return (
    <div className={styles.appWrapper}>
      <aside className={styles.sidebar}>
        <Conversation />
      </aside>
      <main className={styles.mainContent}>
        <Dashboard />
      </main>
    </div>
  );
}
