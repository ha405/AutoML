import React, { useEffect, useState, useRef } from 'react';
// 1. Import the renamed CSS Module
import styles from './conversation_style.module.css';
import { useLocation, useNavigate } from 'react-router-dom';
import { callApi } from '../api/client'; // Adjust path if necessary

// Keep parseMessage function
const parseMessage = (msg) => {
  if (typeof msg !== 'string') return { sender: 'unknown', content: String(msg) };
  const parts = msg.split(":", 1);
  const sender = parts[0].toLowerCase();
  let content = msg.substring(parts[0].length + 1).trim();
  if (sender === 'ai' && content.match(/^Question\s*\d+\)/i)) {
    content = content.substring(content.indexOf(')') + 1).trim();
  }
  return { sender, content };
};


export default function Conversation() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const [conversation, setConversation] = useState(state?.conversation || []);
  const [userResponse, setUserResponse] = useState('');
  const [isFinal, setIsFinal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const conversationEndRef = useRef(null);

  // Keep useEffect for scrolling
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation]);

  // Keep handleSubmit function
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userResponse.trim() || isLoading) return;
    setIsLoading(true);
    const messageToSend = userResponse;
    setUserResponse('');
    setConversation(prev => [...prev, `User: ${messageToSend}`]);
    try {
      const data = await callApi('conversation', 'POST', { user_response: messageToSend });
      setConversation(data.conversation || []);
      if (data.final_problem) { setIsFinal(true); }
    } catch (error) {
      console.error("API call failed:", error);
      setConversation(prev => prev.slice(0, -1));
      setUserResponse(messageToSend);
    } finally {
       setIsLoading(false);
    }
  };

  return (
    // 2. Add the outer wrapper div for page background and centering
    <div className={styles.conversationPageWrapper}>
      {/* 3. Use the new class name for the actual chat window */}
      <div className={styles.chatWindow}>
        <div className={styles.header}>
          <h1>Getting On Track...</h1>
        </div>

        <div className={styles.conversation}>
          {conversation.map((msg, i) => {
            const { sender, content } = parseMessage(msg);
            const isAI = sender === 'ai';
            const bubbleClass = isAI ? styles.aiBubble : styles.userBubble;
            return (
              <div key={i} className={bubbleClass}>
                <p>{content}</p>
              </div>
            );
          })}
          <div ref={conversationEndRef} />
        </div>

        {isFinal ? (
          <div className={styles.savedMessage}>
            <p>Your final business problem has been saved for later use.</p>
          </div>
        ) : (
          <form className={styles.responseForm} onSubmit={handleSubmit}>
            <input
              type="text"
              placeholder="Type your answer here..."
              value={userResponse}
              onChange={(e) => setUserResponse(e.target.value)}
              required
              disabled={isLoading}
            />
            <button
              type="submit"
              className={styles.submitButton}
              disabled={isLoading || !userResponse.trim()}
            >
              Send
            </button>
          </form>
        )}
      </div> {/* End chatWindow */}
    </div> // End conversationPageWrapper
  );
}