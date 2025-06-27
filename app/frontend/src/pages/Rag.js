import React, { useState, useRef, useEffect } from 'react';
import styles from './chat.module.css';
import { callApi } from '../api/client';
import { useLocation } from 'react-router-dom';

export default function Rag() {
  const { search } = useLocation();
  const [queryInput, setQueryInput] = useState('');
  const [messages, setMessages] = useState([]); // { sender:'user'|'bot', text }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  // ref to ensure we only fetch once on mount
  const initialFetchRef = useRef(false);

  // On mount: fetch initial query if present
  useEffect(() => {
    const params = new URLSearchParams(search);
    const q = params.get('query')?.trim();
    if (q && !initialFetchRef.current) {
      initialFetchRef.current = true;
      setQueryInput(q);
      setMessages([{ sender: 'user', text: q }]);
      console.log('🏁 Initial fetchAnswer:', q);
      fetchAnswer(q);
    }
  }, [search]);

  // Auto-scroll on new messages
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const fetchAnswer = async (prompt) => {
    setLoading(true);
    setError('');
    try {
      console.log('🔍 fetchAnswer called for:', prompt);
      const { answer } = await callApi('rag', 'POST', { query: prompt });
      setMessages((m) => [...m, { sender: 'bot', text: answer }]);
    } catch (err) {
      console.error(err);
      setError('Error fetching response.');
    } finally {
      setLoading(false);
    }
  };

  const submitQuery = (e) => {
    if (e) e.preventDefault();
    const prompt = queryInput.trim();
    if (!prompt || loading) return;

    setMessages((m) => [...m, { sender: 'user', text: prompt }]);
    setQueryInput('');
    inputRef.current?.focus();
    fetchAnswer(prompt);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery(e);
    }
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messageList} ref={containerRef}>
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === 'user' ? styles.userMsg : styles.botMsg}
          >
            {m.text}
          </div>
        ))}

        {loading && <div className={styles.loading}>Loading...</div>}
        {error && <div className={styles.error}>{error}</div>}
      </div>

      <form onSubmit={submitQuery} className={styles.inputBar}>
        <input
          type="text"
          ref={inputRef}
          className={styles.input}
          placeholder="Type your message..."
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
          disabled={loading}
        />
        <button
          type="submit"
          className={styles.sendBtn}
          disabled={!queryInput.trim() || loading}
          title="Send"
        >
          ➤
        </button>
      </form>
    </div>
  );
}
