import React, { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import styles from './conversation_style.module.css';
import { callApi } from '../api/client'; // Ensure this points to your API client function

// --- Define Backend Prompt Strings as JS Constants ---
// Extract the core message, trim, and lower-case for easier comparison later
// IMPORTANT: Update these strings if your Python feedback.py prompts change!
const BACKEND_SYSTEM_PROMPT_INITIAL_CORE = `
You are an AI assisting with business problem definition.
The user will provide a brief business problem statement.
Your task is to:
1. Analyze the user's problem statement.
2. Decide if you need more information to understand the problem deeply enough to formulate a well-defined business problem statement.
3. If you need more information, ask ONE clarifying question.  The question should be specific and help you understand a crucial aspect of the problem.
4. If you believe you have enough information to formulate a problem statement, your response MUST START with the EXACT phrase: "Ready to formulate problem." and nothing else before it. Do not ask a question in this case.
Your response should follow one of these formats:
Format 1: Asking a Clarifying Question (if more info needed):
<Your clarifying question>
Format 2: Ready to formulate problem (if enough info):
Ready to formulate problem.  (Your response MUST START with this EXACT phrase)
Example:
What specific metrics are you currently tracking to measure this problem?
Start the process now.
`.trim().toLowerCase();

const BACKEND_SYSTEM_PROMPT_NEXT_ITERATION_CORE = `
You are continuing to assist with business problem definition.
The user has responded to your previous question.
Your task is to:
1. Review the entire conversation so far, including the initial problem statement and the user's responses to your questions.
2. Decide if you now have enough information to formulate a well-defined business problem statement.
3. If you still need more information, ask ONE more clarifying question.
4. If you believe you have enough information NOW to formulate a problem statement, your response MUST START with the EXACT phrase: "Ready to formulate problem." and nothing else before it.
Your response should follow one of these formats:
Format 1: Asking a Clarifying Question:
<Your clarifying question>
Format 2: Ready to formulate problem:
Ready to formulate problem. (Your response MUST START with this EXACT phrase)
Continue the process.
`.trim().toLowerCase();
// --- End Backend Prompt Strings ---


export default function Conversation() {
  const { state } = useLocation(); // Initial state from /api/home via navigation
  const [conversation, setConversation] = useState([]);
  const [userResponse, setUserResponse] = useState('');
  const [isFinal, setIsFinal] = useState(false);
  const [finalProblemText, setFinalProblemText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle, conversation, conversation_complete, processing_eda, processing_plan, processing_ml, processing_viz_plan, processing_viz, complete, error
  const [pipelineMessage, setPipelineMessage] = useState(''); // User-facing status message
  const conversationEndRef = useRef(null);

  // --- Effect 1: Initial Load and Setup ---
  useEffect(() => {
    const locationState = state;
    let initialConvo = [];
    let initialFinalProblem = null;

    if (locationState) {
        initialConvo = locationState.conversation || [];
        initialFinalProblem = locationState.final_problem || null;
    }

    // Filter out non-display messages (FileDetails, initial User prompt if present)
    const filteredConvo = initialConvo.filter(msg =>
        !msg.startsWith('FileDetails:') &&
        !(msg.startsWith('User:') && initialConvo.indexOf(msg) === 0)
    );
    setConversation(filteredConvo);

    if (initialFinalProblem) {
        setFinalProblemText(initialFinalProblem);
        setIsFinal(true);
        setPipelineStatus('conversation_complete');
    } else if (filteredConvo.length > 0) {
        // If initialConvo had content (likely the first AI question from /api/home)
        setPipelineStatus('conversation');
    } else {
        // This might happen if /api/home failed or returned empty convo
        setPipelineStatus('error');
        setPipelineMessage('Failed to initialize conversation.');
    }

  }, [state]); // Depend only on state from navigation


  // --- Effect 2: Scroll Conversation ---
  useEffect(() => {
    if (pipelineStatus === 'conversation') {
      conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation, pipelineStatus]);


  // --- Effect 3: Trigger Pipeline Steps ---
  useEffect(() => {
    const runPipelineStep = async (step) => {
      let nextStep = '';
      let success = false;
      let errorMessage = '';
      let apiEndpoint = '';
      let apiMethod = 'POST';
      let resultData = null; // To store response data if needed

      try {
        switch (step) {
          case 'processing_eda':
            setPipelineMessage('Generating & Running EDA...');
            apiEndpoint = 'dataanalysis';
            resultData = await callApi(apiEndpoint, apiMethod);
            // Adjust success check based on actual backend response structure
            success = resultData?.status?.includes('success') || resultData?.execution_successful;
            if (!success) errorMessage = resultData?.error || 'EDA execution failed.';
            else nextStep = 'processing_plan';
            break;

          case 'processing_plan':
             setPipelineMessage('Generating ML Plan...');
             apiEndpoint = 'superllm';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'plan_generated';
             if (!success) errorMessage = resultData?.error || 'ML plan generation failed.';
             else nextStep = 'processing_ml';
            break;

          case 'processing_ml':
             setPipelineMessage('Generating & Running ML Code...');
             apiEndpoint = 'ml';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'success';
             if (!success) errorMessage = resultData?.error || resultData?.message || 'ML execution failed.';
             else nextStep = 'processing_viz_plan';
            break;

          case 'processing_viz_plan':
             setPipelineMessage('Generating Visualization Plan...');
             apiEndpoint = 'visualizationplanning';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'plan_generated';
             if (!success) errorMessage = resultData?.error || 'Visualization plan generation failed.';
             else nextStep = 'processing_viz';
            break;

          case 'processing_viz':
             setPipelineMessage('Generating & Running Visualizations...');
             apiEndpoint = 'visualizations';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'success';
             if (!success) errorMessage = resultData?.error || resultData?.message || 'Visualization execution failed.';
             else nextStep = 'complete';
            break;

          default:
            console.warn("Unknown pipeline step:", step);
            return;
        }

        if (success) {
          setPipelineStatus(nextStep);
        } else {
          setPipelineStatus('error');
          setPipelineMessage(`Error during ${step.replace('processing_', '')}: ${errorMessage}`);
        }

      } catch (err) {
        console.error(`API call failed for ${step} (${apiEndpoint}):`, err);
        setPipelineStatus('error');
        setPipelineMessage(`Error during ${step.replace('processing_', '')}: ${err.message || 'Network or API error'}`);
      }
    };

    if (pipelineStatus === 'conversation_complete') {
      setPipelineStatus('processing_eda');
    } else if (pipelineStatus.startsWith('processing_')) {
      runPipelineStep(pipelineStatus);
    } else if (pipelineStatus === 'complete') {
        setPipelineMessage('Processing Complete! Visualizations generated.');
    }

  }, [pipelineStatus]); // Run when pipelineStatus changes


  // --- Handle User Submission ---
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userResponse.trim() || isLoading || isFinal || pipelineStatus !== 'conversation') return;

    const answer = userResponse.trim();
    const currentConvoSnapshot = [...conversation]; // Snapshot before optimistic update
    const updatedConvoUI = [...currentConvoSnapshot, `User: ${answer}`];
    setConversation(updatedConvoUI);
    setUserResponse('');
    setIsLoading(true);

    try {
      const data = await callApi('conversation', 'POST', { user_response: answer });

      const newConvo = data.conversation || [];
      // Filter response from backend before setting state
      const finalFilteredConvo = newConvo.filter(msg =>
          !msg.startsWith('FileDetails:') &&
          !(msg.startsWith('User:') && newConvo.indexOf(msg) === 0) // Filter initial user prompt IF backend includes it
      );

      setConversation(finalFilteredConvo);

      if (data.final_problem) {
        setFinalProblemText(data.final_problem);
        setIsFinal(true);
        setPipelineStatus('conversation_complete'); // Trigger pipeline
      }
    } catch (err) {
      console.error('API call failed:', err);
      setConversation(currentConvoSnapshot); // Revert on error
      setUserResponse(answer);
      setPipelineStatus('error');
      setPipelineMessage(`Failed to get response: ${err.message || 'API error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to clean messages
  const cleanMessage = (msg, prefixToRemove) => {
    let cleaned = msg.startsWith(prefixToRemove)
      ? msg.substring(prefixToRemove.length).trim()
      : msg.trim();
    cleaned = cleaned.replace(/^question\s*\d+\)\s*/i, '').trim(); // Remove "Question X)" prefix
    cleaned = cleaned.replace(/^ai:\s*/i, '').trim(); // Remove potential "AI: " prefix if backend adds it unexpectedly
    return cleaned;
  };

  // Filter system messages
  const shouldDisplaySystemMessage = (msg) => {
    // Only check messages starting with "System:"
    if (!msg.startsWith('System:')) return true; // Display non-system messages

    const cleanedMsg = msg.substring('System:'.length).trim();
    const content = cleanedMsg.toLowerCase();

    // Check for exact match against the core prompt text (already lowercased)
    if (content === BACKEND_SYSTEM_PROMPT_INITIAL_CORE ||
        content === BACKEND_SYSTEM_PROMPT_NEXT_ITERATION_CORE) {
        return false; // Don't display the exact system prompts
    }

    // Check for other generic hidden phrases
    const hiddenPhrases = [
        "you are ", "your task is", "format 1:", "format 2:",
        "decide if you now have enough information", "continue the process",
        "structure a well-defined business problem statement",
        "your response should only include the final business problem statement",
        "based on the entire conversation.*final business problem statement" // Basic check
    ];

    // If content includes any of these, hide it
    return !hiddenPhrases.some(phrase => content.includes(phrase.replace(/\.\*/g, '')));
  };


  return (
    <div className={styles.conversationPageWrapper}>
      <div className={styles.chatWindow}>
        <div className={styles.header}>
          <h1>
            {pipelineStatus === 'conversation' && !isFinal && 'Defining the Business Problem...'}
            {isFinal && pipelineStatus !== 'error' && !pipelineStatus.startsWith('processing_') && 'Business Problem Defined'}
            {pipelineStatus.startsWith('processing_') && 'Processing Pipeline...'}
            {pipelineStatus === 'complete' && 'Processing Complete'}
            {pipelineStatus === 'error' && 'Error Occurred'}
          </h1>
        </div>

        {isFinal && !pipelineStatus.startsWith('processing_') && pipelineStatus !== 'conversation' ? (
          <div className={styles.finalProblemDisplay}>
            <h2>Final Business Problem:</h2>
            <p className={styles.finalProblemText}>{finalProblemText}</p>
            {pipelineStatus === 'complete' && (
                <div className={styles.pipelineMessage}>✅ {pipelineMessage}</div>
            )}
             {pipelineStatus === 'error' && (
                <div className={`${styles.pipelineMessage} ${styles.errorMessage}`}>❌ {pipelineMessage}</div>
            )}
             {pipelineStatus === 'conversation_complete' && (
                <div className={styles.pipelineMessage}>⏳ Preparing Automated Analysis...</div>
            )}
          </div>
        ) : (
          <>
            <div className={styles.conversation}>
              {conversation.map((msg, idx) => {
                 const isUser = msg.startsWith('User:');
                 const isSystemOrAI = msg.startsWith('System:') || msg.startsWith('AI:'); // Backend now uses "AI:"
                 const prefix = isUser ? 'User:' : (msg.startsWith('AI:') ? 'AI:' : 'System:');
                 const bubbleClass = isUser ? styles.userBubble : styles.aiBubble;
                 const cleanedMsg = cleanMessage(msg, prefix);

                 // Determine if the message should be displayed
                 const display = isUser || (isSystemOrAI && shouldDisplaySystemMessage(msg));

                 if (cleanedMsg && display) {
                     return (
                        <div key={idx} className={bubbleClass}>
                          {/* Render potentially multi-line messages correctly */}
                          <p style={{ whiteSpace: 'pre-wrap' }}>{cleanedMsg}</p>
                        </div>
                     );
                 }
                 return null;
              })}

              {(isLoading || pipelineStatus.startsWith('processing_')) && (
                <div className={styles.loadingIndicator}>
                  <p>{pipelineStatus.startsWith('processing_') ? pipelineMessage : 'Thinking...'}</p>
                </div>
              )}
               {pipelineStatus === 'error' && pipelineStatus !== 'conversation_complete' && !isFinal && (
                  <div className={`${styles.pipelineMessage} ${styles.errorMessage}`}>❌ {pipelineMessage}</div>
              )}
              <div ref={conversationEndRef} />
            </div>

            {pipelineStatus === 'conversation' && !isFinal && (
              <form className={styles.responseForm} onSubmit={handleSubmit}>
                <input
                  type="text"
                  placeholder="Type your answer here..."
                  value={userResponse}
                  onChange={e => setUserResponse(e.target.value)}
                  required
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  className={styles.submitButton}
                  disabled={isLoading || !userResponse.trim()}
                >
                  {isLoading ? 'Sending...' : 'Send'}
                </button>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  );
}