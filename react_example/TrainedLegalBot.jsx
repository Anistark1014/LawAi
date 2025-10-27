/**
 * React Component for Trained ML Legal Model
 * This component calls your trained ML model API
 */

import React, { useState, useEffect } from 'react';

const TrainedLegalBot = () => {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const [history, setHistory] = useState([]);

  // Your trained model API endpoint
  const API_BASE_URL = 'http://localhost:5000'; // Change this to your deployed URL

  useEffect(() => {
    // Check if model is loaded on component mount
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/health`);
      const data = await res.json();
      setModelStatus(data);
    } catch (error) {
      console.error('Error checking model status:', error);
      setModelStatus({ status: 'error', model_loaded: false });
    }
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/legal-advice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          max_length: 300,  // Adjust response length
          temperature: 0.7  // Adjust creativity (0.1 = conservative, 1.0 = creative)
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      
      // Update conversation history
      const newHistory = [...history, 
        { role: 'user', content: message, timestamp: new Date().toISOString() },
        { role: 'assistant', content: data.response, timestamp: data.timestamp }
      ];
      setHistory(newHistory);
      
      setResponse(data);
      setMessage('');

    } catch (error) {
      console.error('Error calling trained legal model:', error);
      setResponse({
        response: 'Sorry, there was an error connecting to the legal model. Please ensure the API server is running.',
        status: 'error',
        error: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    setResponse(null);
  };

  return (
    <div className="trained-legal-bot">
      <div className="header">
        <h2>🏛️ AI Justice Bot - Trained Legal Model</h2>
        <p>Ask questions about cyber law based on your trained legal documents</p>
        
        {/* Model Status Indicator */}
        <div className={`status-indicator ${modelStatus?.model_loaded ? 'loaded' : 'not-loaded'}`}>
          {modelStatus?.model_loaded ? (
            <span>✅ Legal Model Ready</span>
          ) : (
            <span>❌ Model Not Loaded - Please train the model first</span>
          )}
        </div>
      </div>

      {/* Conversation History */}
      {history.length > 0 && (
        <div className="conversation-history">
          <div className="history-header">
            <h3>Conversation History</h3>
            <button onClick={clearHistory} className="clear-btn">Clear</button>
          </div>
          
          {history.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <div className="message-content">
                <strong>{msg.role === 'user' ? 'You' : 'Legal AI'}:</strong>
                <p>{msg.content}</p>
              </div>
              <div className="timestamp">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Input Section */}
      <div className="input-section">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about cyber law... Example: 'Someone has hacked my social media account and is posting fake content. What legal action can I take?'"
          rows={4}
          className="message-input"
          disabled={!modelStatus?.model_loaded}
        />
        
        <div className="input-controls">
          <button 
            onClick={sendMessage} 
            disabled={loading || !message.trim() || !modelStatus?.model_loaded}
            className="send-button"
          >
            {loading ? 'Getting Legal Advice...' : 'Ask Legal AI'}
          </button>
          
          <div className="input-info">
            <small>
              Powered by your trained legal documents • 
              Response time: ~2-5 seconds
            </small>
          </div>
        </div>
      </div>

      {/* Current Response */}
      {response && (
        <div className="response-section">
          <div className="response-header">
            <h3>Legal Guidance</h3>
            {response.model && (
              <span className="model-badge">
                📚 From: {response.model}
              </span>
            )}
          </div>
          
          <div className="response-content">
            {response.status === 'error' ? (
              <div className="error-response">
                <p>❌ {response.response}</p>
                {response.error && <small>Error: {response.error}</small>}
              </div>
            ) : (
              <div className="success-response">
                <p>{response.response}</p>
              </div>
            )}
          </div>
          
          {response.timestamp && (
            <div className="response-footer">
              Generated at {new Date(response.timestamp).toLocaleString()}
            </div>
          )}
        </div>
      )}

      {/* Example Queries */}
      <div className="example-queries">
        <h4>Example Legal Questions:</h4>
        <div className="examples">
          {[
            "Someone hacked my email and is sending spam from my account",
            "I was cheated in an online payment, what legal recourse do I have?",
            "My personal data was leaked from a website, what are my rights?",
            "Someone is cyberbullying me on social media, what can I do legally?"
          ].map((example, index) => (
            <button
              key={index}
              onClick={() => setMessage(example)}
              className="example-btn"
              disabled={loading || !modelStatus?.model_loaded}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <style jsx>{`
        .trained-legal-bot {
          max-width: 1000px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .header {
          text-align: center;
          margin-bottom: 30px;
        }
        
        .status-indicator {
          display: inline-block;
          padding: 8px 16px;
          border-radius: 20px;
          font-size: 14px;
          font-weight: bold;
          margin-top: 10px;
        }
        
        .status-indicator.loaded {
          background: #d4edda;
          color: #155724;
        }
        
        .status-indicator.not-loaded {
          background: #f8d7da;
          color: #721c24;
        }
        
        .conversation-history {
          background: #f8f9fa;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 20px;
          max-height: 400px;
          overflow-y: auto;
        }
        
        .history-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }
        
        .clear-btn {
          background: #dc3545;
          color: white;
          border: none;
          padding: 5px 12px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }
        
        .message {
          margin-bottom: 15px;
          padding: 10px;
          border-radius: 8px;
        }
        
        .message.user {
          background: #e3f2fd;
          margin-left: 20%;
        }
        
        .message.assistant {
          background: #f3e5f5;
          margin-right: 20%;
        }
        
        .message-content p {
          margin: 5px 0 0 0;
          white-space: pre-wrap;
        }
        
        .timestamp {
          font-size: 11px;
          color: #666;
          text-align: right;
          margin-top: 5px;
        }
        
        .input-section {
          margin: 20px 0;
        }
        
        .message-input {
          width: 100%;
          padding: 15px;
          border: 2px solid #e1e5e9;
          border-radius: 12px;
          font-size: 16px;
          resize: vertical;
          font-family: inherit;
        }
        
        .message-input:disabled {
          background: #f5f5f5;
          cursor: not-allowed;
        }
        
        .input-controls {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 10px;
        }
        
        .send-button {
          background: #007bff;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 16px;
          font-weight: bold;
        }
        
        .send-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }
        
        .input-info {
          color: #666;
          font-size: 12px;
        }
        
        .response-section {
          background: white;
          border: 2px solid #e1e5e9;
          border-radius: 12px;
          padding: 20px;
          margin: 20px 0;
        }
        
        .response-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }
        
        .model-badge {
          background: #f8f9fa;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
          color: #666;
        }
        
        .success-response p {
          line-height: 1.6;
          white-space: pre-wrap;
        }
        
        .error-response {
          color: #dc3545;
        }
        
        .response-footer {
          font-size: 12px;
          color: #666;
          text-align: right;
          margin-top: 10px;
          border-top: 1px solid #eee;
          padding-top: 10px;
        }
        
        .example-queries {
          margin-top: 30px;
          padding: 20px;
          background: #f8f9fa;
          border-radius: 12px;
        }
        
        .examples {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .example-btn {
          background: white;
          border: 1px solid #ddd;
          padding: 10px 15px;
          border-radius: 6px;
          cursor: pointer;
          text-align: left;
          font-size: 14px;
          transition: background 0.2s;
        }
        
        .example-btn:hover:not(:disabled) {
          background: #f0f0f0;
        }
        
        .example-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default TrainedLegalBot;