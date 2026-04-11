// src/pages/Chat.js
import React, { useEffect, useState, useRef } from 'react';
import styles from './chat.module.css';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('loading'); // 'loading' | 'error' | 'ready'
  const [errorMsg, setErrorMsg] = useState('');
  const messageEndRef = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, status]);

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
    if (!text || status === 'loading') return;

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
      setMessages(m => [...m, { role: 'assistant', content: `⚠️ Error: ${err.message}` }]);
      setStatus('ready'); // Allow retry
    }
  };

  // Fatal init error — only show full-page error if we have 0 messages
  if (status === 'error' && messages.length === 0) {
    return (
      <div className={styles.chatContainer}>
        <div className={styles.errorContainer}>
          <p className={styles.error}>Unable to initialize chat.</p>
          <p className={styles.errorDetails}>{errorMsg}</p>
        </div>
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
        {status === 'loading' && messages.length > 0 && (
          <div className={styles.botMsg} style={{ opacity: 0.6 }}>
            Thinking...
          </div>
        )}
        {status === 'loading' && messages.length === 0 && (
          <div className={styles.botMsg} style={{ opacity: 0.6 }}>
            Initializing chat...
          </div>
        )}
        <div ref={messageEndRef} />
      </div>
      <div className={styles.inputBar}>
        <input
          className={styles.input}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message…"
          disabled={status === 'loading'}
        />
        <button
          className={styles.sendBtn}
          onClick={sendMessage}
          disabled={status === 'loading' || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
