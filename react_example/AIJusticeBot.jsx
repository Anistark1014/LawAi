/**
 * Example React component for AI Justice Bot integration
 * This shows how your React website should call the API
 */

import React, { useState } from 'react';

const AIJusticeBot = () => {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  // Your API endpoint - change this to your actual server URL
  const API_BASE_URL = 'http://localhost:5000';

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
          history: history,
          language: 'English'
        })
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      
      // Update history
      const newHistory = [...history, 
        { role: 'user', content: message },
        { role: 'bot', content: data.response }
      ];
      setHistory(newHistory);
      setResponse(data);
      setMessage('');

    } catch (error) {
      console.error('Error calling AI Justice Bot API:', error);
      setResponse({
        type: 'ERROR',
        response: 'Sorry, there was an error processing your request. Please try again.',
        error: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const formatResponse = (responseData) => {
    if (!responseData) return null;

    const { type, response: text, crime_type, sections, punishment } = responseData;

    return (
      <div className="response-container">
        <div className={`response-type ${type.toLowerCase()}`}>
          {type === 'GUIDE' ? '⚖️ Legal Guidance' : '❓ Need More Info'}
        </div>
        
        <div className="response-text">
          {text.split('\n').map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>

        {crime_type && (
          <div className="crime-details">
            <h4>🚨 Crime Type: {crime_type}</h4>
            {sections && (
              <p><strong>📋 Applicable Laws:</strong> {sections.join(', ')}</p>
            )}
            {punishment && (
              <p><strong>⚖️ Punishment:</strong> {punishment}</p>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="ai-justice-bot">
      <h2>🏛️ AI Justice Bot</h2>
      <p>Describe your cyber crime situation and get legal guidance</p>

      <div className="input-section">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Example: Someone hacked my email account and is sending spam messages..."
          rows={4}
          className="message-input"
        />
        <button 
          onClick={sendMessage} 
          disabled={loading || !message.trim()}
          className="send-button"
        >
          {loading ? 'Processing...' : 'Get Legal Advice'}
        </button>
      </div>

      {response && (
        <div className="response-section">
          {formatResponse(response)}
        </div>
      )}

      <style jsx>{`
        .ai-justice-bot {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .input-section {
          margin: 20px 0;
        }
        
        .message-input {
          width: 100%;
          padding: 12px;
          border: 2px solid #e1e5e9;
          border-radius: 8px;
          font-size: 16px;
          resize: vertical;
        }
        
        .send-button {
          background: #007bff;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 16px;
          margin-top: 10px;
        }
        
        .send-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }
        
        .response-container {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 20px;
          margin-top: 20px;
        }
        
        .response-type {
          font-weight: bold;
          margin-bottom: 15px;
          padding: 8px 12px;
          border-radius: 4px;
        }
        
        .response-type.guide {
          background: #d4edda;
          color: #155724;
        }
        
        .response-type.clarify {
          background: #fff3cd;
          color: #856404;
        }
        
        .crime-details {
          background: white;
          padding: 15px;
          border-radius: 6px;
          margin-top: 15px;
          border-left: 4px solid #007bff;
        }
      `}</style>
    </div>
  );
};

export default AIJusticeBot;