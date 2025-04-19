import React, { useState, useRef, useEffect } from 'react';
// 1. Ensure the CSS Module is imported correctly
import styles from './index-react.module.css';
import { useNavigate } from 'react-router-dom';
// Keep Icon components
import AttachFileIcon from '@mui/icons-material/AttachFile';
import MicIcon from '@mui/icons-material/Mic';
import { callApi } from '../api/client'; // Adjust path if necessary

export default function Home() {
  const [queryInput, setQueryInput] = useState('');
  const [file, setFile] = useState(null);
  const navigate = useNavigate();
  const inputRef = useRef(null);
  const [theme] = useState('dark'); // Keep theme state (though CSS module doesn't explicitly use .dark)

  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);
  const [micError, setMicError] = useState('');
  const [micReady, setMicReady] = useState(false);

  // useEffect and other functions remain the same
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setMicError("Voice input not supported by this browser.");
      setMicReady(false); return;
    }
    if (navigator.permissions && typeof navigator.permissions.query === 'function') {
      navigator.permissions.query({ name: 'microphone' }).then((ps) => {
        if (ps.state === 'denied') { setMicError("Mic permission denied."); setMicReady(false); }
        ps.onchange = () => { if (ps.state === 'denied') { setMicError("Mic permission denied."); setMicReady(false); recognitionRef.current?.stop(); } else { setMicError(''); } };
      }).catch(err => console.error("Mic permission query error:", err));
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US"; recognition.interimResults = false; recognition.maxAlternatives = 1;
    recognition.onresult = (e) => { setQueryInput(e.results[0][0].transcript); };
    recognition.onerror = (e) => {
      let msg = `Speech error: ${e.error}`;
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { msg = "Mic permission denied."; }
      else if (e.error === 'no-speech') { msg = "No speech detected."; }
      else if (e.error === 'audio-capture') { msg = "Mic hardware error."; }
      else if (e.error === 'network') { msg = "Network error."; }
      setMicError(msg); setIsListening(false);
    };
    recognition.onstart = () => { setIsListening(true); setMicError(''); };
    recognition.onend = () => { setIsListening(false); };
    recognitionRef.current = recognition; setMicReady(true); setMicError('');
    return () => { recognitionRef.current?.abort(); };
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) { setFile(e.target.files[0]); setMicError(''); }
    else { setFile(null); }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isListening && recognitionRef.current) { recognitionRef.current.stop(); }
    const formData = new FormData();
    formData.append('queryInput', queryInput);
    if (file) { formData.append('file', file); }
    try {
      const data = await callApi('home', 'POST', formData);
      if (data && data.conversation) { navigate('/conversation', { state: { conversation: data.conversation } }); }
      else { console.error("API response missing data."); setMicError("Failed to process request."); }
    } catch (error) { console.error("API call failed:", error); setMicError("Network/server error."); }
  };

  const handleVoiceInput = () => {
    if (!micReady || !recognitionRef.current) { if (!micError) setMicError("Voice unavailable."); return; }
    if (!isListening) {
      try { setQueryInput(''); setFile(null); setMicError(''); recognitionRef.current.start(); }
      catch (error) { console.error("Mic start error:", error); setMicError("Could not start mic."); setIsListening(false); }
    } else { recognitionRef.current.stop(); }
  };

  const micButtonDisabled = !micReady;

  // Construct className strings using the imported styles object
  // Assuming your CSS module uses camelCase or converts kebab-case to camelCase
  const autoMLClassName = `${styles.autoML || ''} ${isListening ? styles.listening : ''}`.trim();
  const micButtonClassName = `${styles.iconButton || ''} ${isListening ? styles.active : ''}`.trim();

  return (
    // 2. Apply classNames using the 'styles' object
    <div className={autoMLClassName}>
      <div className={styles.container}>
        <div className={styles.header}>
          <p>Bonjour</p>
          <h1>How Can We Help You?</h1>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Make sure CSS module defines .inputArea */}
          <div className={styles.inputArea}>
            {/* Make sure CSS module defines .inputWrapper */}
            <div className={styles.inputWrapper}>
              <input
                type="text"
                ref={inputRef}
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder={isListening ? "Listening..." : "Describe Your Problem"}
                required
                disabled={isListening}
                // Input field often doesn't need its own specific class from modules
                // if styled via parent (.inputWrapper input)
              />
              {/* Make sure CSS module defines .inputIcons */}
              <div className={styles.inputIcons}>
                <label
                  htmlFor="file-upload"
                  className={styles.iconButton} // Use styles object
                  aria-disabled={isListening}
                  title="Attach file"
                  style={isListening ? { pointerEvents: 'none', opacity: 0.5 } : {}}
                >
                  <AttachFileIcon fontSize="inherit" />
                </label>
                <input
                  id="file-upload"
                  type="file"
                  hidden
                  onChange={handleFileChange}
                  disabled={isListening}
                />
                <button
                  type="button"
                  onClick={handleVoiceInput}
                  className={micButtonClassName} // Use combined className
                  disabled={micButtonDisabled || isListening}
                  title={micError ? micError : !micReady ? "Voice input unavailable" : isListening ? "Stop Listening" : "Start Voice Input"}
                >
                  <MicIcon fontSize="inherit" />
                </button>
              </div>
            </div>

            {/* Use styles object for status messages */}
            {file && !micError && !isListening && (
              <p className={styles.fileDisplay}>File: {file.name}</p>
            )}
            {isListening && (
              <p className={`${styles.statusDisplay || ''} ${styles.listeningIndicator || ''}`.trim()}>🎙️ Listening...</p>
            )}
            {micError && (
              <p className={`${styles.statusDisplay || ''} ${styles.errorMessage || ''}`.trim()}>⚠️ {micError}</p>
            )}
          </div>

          {/* Use styles object for the action button */}
          <button type="submit" className={styles.actionButton} disabled={isListening}>
            Start Analysis
          </button>
        </form>
      </div>
    </div>
  );
}