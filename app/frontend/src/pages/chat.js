// src/pages/Chat.js
import React, { useEffect, useState } from 'react';
import styles from './chat.module.css';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('loading'); // 'loading' | 'error' | 'ready'
  const [errorMsg, setErrorMsg] = useState('');

  // Initialize
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/chatlm', { method: 'POST' });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || `HTTP ${res.status}`);
        }
        const { suggestion } = await res.json();
        setMessages([{ role: 'assistant', content: suggestion }]);
        setStatus('ready');
      } catch (err) {
        console.error('Chat init failed:', err);
        setErrorMsg(err.message);
        setStatus('error');
      }
    })();
  }, []);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    setMessages(m => [...m, { role: 'user', content: text }]);
    setInput('');
    setStatus('loading');

    try {
      const res = await fetch('/api/chatlm/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_message: text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const { response } = await res.json();
      setMessages(m => [...m, { role: 'assistant', content: response }]);
      setStatus('ready');
    } catch (err) {
      console.error('Chat send failed:', err);
      setErrorMsg(err.message);
      setStatus('error');
    }
  };

  if (status === 'loading') {
    return <p className={styles.loading}>Loading chat…</p>;
  }
  if (status === 'error') {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.error}>Unable to load chat.</p>
        <p className={styles.errorDetails}>{errorMsg}</p>
      </div>
    );
  }

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messageList}>
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === 'user' ? styles.userMsg : styles.botMsg}
          >
            {m.content}
          </div>
        ))}
      </div>
      <div className={styles.inputBar}>
        <input
          className={styles.input}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message…"
        />
        <button
          className={styles.sendBtn}
          onClick={sendMessage}
          disabled={status === 'loading'}
        >
          Send
        </button>
      </div>
    </div>
  );
}
