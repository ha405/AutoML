import React, { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import styles from './conversation_style.module.css';
import { callApi } from '../api/client'; // Ensure this points to your API client function

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


export default function Conversation() {
  const { state } = useLocation();
  const [conversation, setConversation] = useState([]);
  const [userResponse, setUserResponse] = useState('');
  const [isFinal, setIsFinal] = useState(false);
  const [finalProblemText, setFinalProblemText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  // Updated Pipeline Status states reflecting the new order
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle, initializing, conversation, conversation_complete, processing_plan, processing_eda, processing_ml, processing_viz_plan, processing_viz, complete, error
  const [pipelineMessage, setPipelineMessage] = useState('');
  const conversationEndRef = useRef(null);

  const processAndSetConversation = (convoData, finalProblemData) => {
    const filteredConvo = (convoData || []).filter(msg =>
        !msg.startsWith('FileDetails:') &&
        !(msg.startsWith('User:') && (convoData || []).indexOf(msg) === 0)
    );
    setConversation(filteredConvo);

    if (finalProblemData) {
        setFinalProblemText(finalProblemData);
        setIsFinal(true);
        setPipelineStatus('conversation_complete');
    } else if (filteredConvo.length > 0) {
        setPipelineStatus('conversation');
    } else {
        setPipelineStatus('error');
        setPipelineMessage('Conversation initialization failed or empty.');
    }
  };

  useEffect(() => {
    const locationState = state;
    let initialConvoFromState = [];
    let initialFinalProblemFromState = null;

    if (locationState) {
        initialConvoFromState = locationState.conversation || [];
        initialFinalProblemFromState = locationState.final_problem || null;
    }

    setIsLoading(true);
    setPipelineStatus('initializing');
    setPipelineMessage('Initializing conversation...');

    if (initialFinalProblemFromState) {
        processAndSetConversation(initialConvoFromState, initialFinalProblemFromState);
        setIsLoading(false);
        setPipelineMessage('');
    } else {
        callApi('conversation', 'GET')
            .then(data => {
                if (data && data.conversation) {
                    processAndSetConversation(data.conversation, data.final_problem);
                } else {
                     console.error("GET /api/conversation response missing data:", data);
                     setPipelineStatus('error');
                     setPipelineMessage('Failed to get initial conversation state from server.');
                }
            })
            .catch(err => {
                console.error("Initial GET /api/conversation failed:", err);
                setPipelineStatus('error');
                setPipelineMessage(`Failed to connect: ${err.message || 'Network error'}`);
            })
            .finally(() => {
                setIsLoading(false);
                 if (pipelineStatus !== 'error') {
                     setPipelineMessage('');
                 }
            });
    }
  }, [state]); // Removed processAndSetConversation from deps


  useEffect(() => {
    if (pipelineStatus === 'conversation' || pipelineStatus === 'initializing') {
       conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation, pipelineStatus, isLoading]);


  // --- Effect 3: Trigger Pipeline Steps (Updated Order) ---
  useEffect(() => {
    const runPipelineStep = async (step) => {
      let nextStep = '';
      let success = false;
      let errorMessage = '';
      let apiEndpoint = '';
      let apiMethod = 'POST';
      let resultData = null;

      setIsLoading(true);

      try {
        switch (step) {
          // --- Step 1: Generate Unified Plan (SuperLLM) ---
          case 'processing_plan':
             setPipelineMessage('Generating Unified Analysis & ML Plan...');
             apiEndpoint = 'superllm'; // Call the endpoint that triggers the planner
             resultData = await callApi(apiEndpoint, apiMethod);
             // Adjust success check based on actual backend response
             success = resultData?.status === 'plan_generated' || resultData?.status === 'ml_plan_exists'; // Allow checking existing plan too
             if (!success) errorMessage = resultData?.error || resultData?.details || 'Unified plan generation failed.';
             else nextStep = 'processing_eda'; // Next step is EDA
             break;

          // --- Step 2: Run EDA (Guided by Plan) ---
          case 'processing_eda':
            setPipelineMessage('Generating & Running Guided EDA...');
            apiEndpoint = 'dataanalysis';
            resultData = await callApi(apiEndpoint, apiMethod);
            success = resultData?.status === 'success';
            if (!success) errorMessage = resultData?.error || resultData?.details || 'Guided EDA execution failed.';
            else nextStep = 'processing_ml'; // Next step is ML
            break;

          // --- Step 3: Run ML (Based on Plan & EDA Output) ---
          case 'processing_ml':
             setPipelineMessage('Generating & Running ML Code...');
             apiEndpoint = 'ml';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'success';
             if (!success) errorMessage = resultData?.error || resultData?.details || 'ML execution failed.';
             else nextStep = 'processing_viz_plan'; // Next step is Viz Plan
            break;

          // --- Step 4: Generate Visualization Plan ---
          case 'processing_viz_plan':
             setPipelineMessage('Generating Visualization Plan...');
             apiEndpoint = 'visualizationplanning';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'plan_generated';
             if (!success) errorMessage = resultData?.error || resultData?.details || 'Visualization plan generation failed.';
             else nextStep = 'processing_viz';
            break;

          // --- Step 5: Run Visualizations ---
          case 'processing_viz':
             setPipelineMessage('Generating & Running Visualizations...');
             apiEndpoint = 'visualizations';
             resultData = await callApi(apiEndpoint, apiMethod);
             success = resultData?.status === 'success';
             if (!success) errorMessage = resultData?.error || resultData?.message || 'Visualization execution failed.';
             else nextStep = 'complete'; // Pipeline finishes
            break;

          default:
            console.warn("Unknown pipeline step:", step);
            setIsLoading(false);
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
      } finally {
          if (!success || nextStep === 'complete') {
             setIsLoading(false);
          }
      }
    };

    // --- Pipeline Trigger Logic (Updated Start) ---
    if (pipelineStatus === 'conversation_complete') {
      const timer = setTimeout(() => {
         // Start with the Plan generation now
         setPipelineStatus('processing_plan');
      }, 500);
      return () => clearTimeout(timer);
    } else if (pipelineStatus.startsWith('processing_')) {
      runPipelineStep(pipelineStatus);
    } else if (pipelineStatus === 'complete') {
        setPipelineMessage('Processing Complete! Visualizations generated.');
        setIsLoading(false);
    } else if (pipelineStatus === 'error') {
        setIsLoading(false);
    }

  }, [pipelineStatus]); // Keep dependency


  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userResponse.trim() || isLoading || isFinal || pipelineStatus !== 'conversation') return;

    const answer = userResponse.trim();
    const currentConvoSnapshot = [...conversation];
    const updatedConvoUI = [...currentConvoSnapshot, `User: ${answer}`];
    setConversation(updatedConvoUI);
    setUserResponse('');
    setIsLoading(true);
    setPipelineMessage('');

    try {
      const data = await callApi('conversation', 'POST', { user_response: answer });
      const newConvo = data.conversation || [];
      const finalProblemData = data.final_problem || null;
      processAndSetConversation(newConvo, finalProblemData); // Use helper
    } catch (err) {
      console.error('API call failed:', err);
      setConversation(currentConvoSnapshot);
      setUserResponse(answer);
      setPipelineStatus('error');
      setPipelineMessage(`Failed to get response: ${err.message || 'API error'}`);
    } finally {
      // Re-evaluate isLoading based on whether the conversation phase is complete
       const problemIsNowFinal = !!(finalProblemText || state?.final_problem); // Check if problem is determined now
       if (!problemIsNowFinal && pipelineStatus !== 'error') {
           setIsLoading(false);
       }
       // If problem became final, isLoading will be handled by the pipeline effect
    }
  };

  const cleanMessage = (msg, prefixToRemove) => {
    let cleaned = msg.startsWith(prefixToRemove)
      ? msg.substring(prefixToRemove.length).trim()
      : msg.trim();
    cleaned = cleaned.replace(/^question\s*\d+\)\s*/i, '').trim();
    cleaned = cleaned.replace(/^ai:\s*/i, '').trim();
    return cleaned;
  };

  const shouldDisplaySystemMessage = (msg) => {
    if (!msg.startsWith('System:')) return true;
    const cleanedMsg = msg.substring('System:'.length).trim();
    const content = cleanedMsg.toLowerCase();
    if (content === BACKEND_SYSTEM_PROMPT_INITIAL_CORE ||
        content === BACKEND_SYSTEM_PROMPT_NEXT_ITERATION_CORE) {
        return false;
    }
    const hiddenPhrases = [
        "you are ", "your task is", "format 1:", "format 2:",
        "decide if you now have enough information", "continue the process",
        "structure a well-defined business problem statement",
        "your response should only include the final business problem statement",
        "based on the entire conversation.*final business problem statement"
    ];
    return !hiddenPhrases.some(phrase => content.includes(phrase.replace(/\.\*/g, '')));
  };

  return (
    <div className={styles.conversationPageWrapper}>
      <div className={styles.chatWindow}>
        <div className={styles.header}>
          <h1>
            {pipelineStatus === 'initializing' && 'Initializing...'}
            {pipelineStatus === 'conversation' && !isFinal && 'Defining the Business Problem...'}
            {isFinal && pipelineStatus === 'conversation_complete' && 'Business Problem Defined'}
            {pipelineStatus.startsWith('processing_') && 'Processing Pipeline...'}
            {pipelineStatus === 'complete' && 'Processing Complete'}
            {pipelineStatus === 'error' && 'Error Occurred'}
          </h1>
        </div>

        {isFinal && pipelineStatus !== 'conversation' ? (
          <div className={styles.finalProblemDisplay}>
            <h2>Final Business Problem:</h2>
            <p className={styles.finalProblemText}>{finalProblemText}</p>
            {(isLoading || pipelineStatus.startsWith('processing_') || pipelineStatus === 'conversation_complete') && (
                 <div className={styles.pipelineMessage}>⏳ {pipelineMessage || 'Starting Pipeline...'}</div>
            )}
            {pipelineStatus === 'complete' && !isLoading && (
                <div className={styles.pipelineMessage}>✅ {pipelineMessage}</div>
            )}
             {pipelineStatus === 'error' && !isLoading && (
                <div className={`${styles.pipelineMessage} ${styles.errorMessage}`}>❌ {pipelineMessage}</div>
            )}
          </div>
        ) : (
          <>
            <div className={styles.conversation}>
              {conversation.map((msg, idx) => {
                 const isUser = msg.startsWith('User:');
                 const isSystemOrAI = msg.startsWith('System:') || msg.startsWith('AI:');
                 const prefix = isUser ? 'User:' : (msg.startsWith('AI:') ? 'AI:' : 'System:');
                 const bubbleClass = isUser ? styles.userBubble : styles.aiBubble;
                 const cleanedMsg = cleanMessage(msg, prefix);
                 const display = isUser || (isSystemOrAI && shouldDisplaySystemMessage(msg));

                 if (cleanedMsg && display) {
                     return (
                        <div key={idx} className={bubbleClass}>
                          <p style={{ whiteSpace: 'pre-wrap' }}>{cleanedMsg}</p>
                        </div>
                     );
                 }
                 return null;
              })}

              {isLoading && (
                <div className={styles.loadingIndicator}>
                  <p>{pipelineMessage || 'Thinking...'}</p>
                </div>
              )}
               {pipelineStatus === 'error' && !isLoading && !isFinal && (
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