import React, { useEffect, useMemo, useState } from 'react';

const UI_STRINGS = {
  en: {
    title: 'AI Justice Bot',
    subtitle: 'Describe your cyber law situation to get guidance',
    placeholder: 'Example: Someone hacked my email account and is demanding money…',
    send: 'Get Legal Advice',
    processing: 'Processing…',
    clear: 'Clear history',
    empty: 'Tell me what happened and I will prepare the legal response.',
    error: 'Sorry, something went wrong. Please try again.',
    referencesTitle: '📚 References',
    englishReference: 'English reference',
    originalResponse: 'Original English Response',
    detectedLabel: 'Detected language',
    responseLabel: 'Response language',
    processedQueryLabel: 'English understanding',
    translationFallback: 'Showing the English response because translation was unavailable.',
    questionLabel: 'Your question',
    originalQuestion: 'Original input',
    documentInputLabel: 'Document text',
    originalDocumentLabel: 'Original document text',
    uploadTitle: 'Upload Legal Document',
    uploadDescription: 'Attach a PDF or image to analyse and receive guidance instantly.',
    uploadButton: 'Analyze Document',
    uploading: 'Processing document…',
    documentAnalysis: 'Document Analysis',
    noDocument: 'No document analysed yet.',
    languageNames: {
      en: 'English',
      hi: 'Hindi',
      mr: 'Marathi'
    }
  },
  hi: {
    title: 'एआई न्याय सहायक',
    subtitle: 'अपनी साइबर कानून से जुड़ी समस्या बताएं और मार्गदर्शन प्राप्त करें',
    placeholder: 'उदाहरण: किसी ने मेरा ईमेल हैक कर लिया है और पैसे मांग रहा है…',
    send: 'कानूनी सलाह प्राप्त करें',
    processing: 'प्रक्रिया जारी है…',
    clear: 'इतिहास साफ़ करें',
    empty: 'क्या हुआ यह बताएं, मैं कानूनी उत्तर तैयार करूँगा।',
    error: 'क्षमा करें, कोई त्रुटि हुई। कृपया पुनः प्रयास करें।',
    referencesTitle: '📚 संदर्भ',
    englishReference: 'अंग्रेज़ी संदर्भ',
    originalResponse: 'मूल अंग्रेज़ी उत्तर',
    detectedLabel: 'पहचानी गई भाषा',
    responseLabel: 'उत्तर की भाषा',
    processedQueryLabel: 'अंग्रेज़ी में समझ',
    translationFallback: 'अनुवाद उपलब्ध न होने के कारण अंग्रेज़ी उत्तर दिखाया जा रहा है।',
    questionLabel: 'आपका प्रश्न',
    originalQuestion: 'मूल इनपुट',
    documentInputLabel: 'दस्तावेज़ पाठ',
    originalDocumentLabel: 'मूल दस्तावेज़ पाठ',
    uploadTitle: 'कानूनी दस्तावेज़ अपलोड करें',
    uploadDescription: 'तुरंत विश्लेषण और मार्गदर्शन प्राप्त करने के लिए PDF या छवि संलग्न करें।',
    uploadButton: 'दस्तावेज़ का विश्लेषण करें',
    uploading: 'दस्तावेज़ प्रोसेस हो रहा है…',
    documentAnalysis: 'दस्तावेज़ विश्लेषण',
    noDocument: 'अभी तक कोई दस्तावेज़ का विश्लेषण नहीं किया गया।',
    languageNames: {
      en: 'अंग्रेज़ी',
      hi: 'हिंदी',
      mr: 'मराठी'
    }
  },
  mr: {
    title: 'एआय न्याय सहाय्यक',
    subtitle: 'आपली सायबर कायद्याची परिस्थिती सांगा आणि मार्गदर्शन मिळवा',
    placeholder: 'उदाहरण: कुणीतरी माझे ईमेल हॅक करून पैसे मागत आहे…',
    send: 'कायदेशीर सल्ला मिळवा',
    processing: 'प्रक्रिया सुरू आहे…',
    clear: 'इतिहास साफ करा',
    empty: 'काय झाले ते सांगा, मी कायदेशीर उत्तर तयार करेन।',
    error: 'माफ करा, काहीतरी चूक झाली. कृपया पुन्हा प्रयत्न करा।',
    referencesTitle: '📚 संदर्भ',
    englishReference: 'इंग्रजी संदर्भ',
    originalResponse: 'मूळ इंग्रजी उत्तर',
    detectedLabel: 'ओळखली गेलेली भाषा',
    responseLabel: 'उत्तराची भाषा',
    processedQueryLabel: 'इंग्रजीत समज',
    translationFallback: 'अनुवाद उपलब्ध नसल्यामुळे इंग्रजी उत्तर दाखवले जात आहे।',
    questionLabel: 'तुमचा प्रश्न',
    originalQuestion: 'मूळ इनपुट',
    documentInputLabel: 'कागदपत्र मजकूर',
    originalDocumentLabel: 'मूळ कागदपत्र मजकूर',
    uploadTitle: 'कायदेशीर कागदपत्र अपलोड करा',
    uploadDescription: 'त्वरित विश्लेषण आणि मार्गदर्शन मिळवण्यासाठी PDF किंवा प्रतिमा जोडा।',
    uploadButton: 'कागदपत्राचे विश्लेषण करा',
    uploading: 'कागदपत्र प्रक्रिया सुरू आहे…',
    documentAnalysis: 'कागदपत्र विश्लेषण',
    noDocument: 'अद्याप कोणत्याही कागदपत्राचे विश्लेषण केले नाही।',
    languageNames: {
      en: 'इंग्रजी',
      hi: 'हिंदी',
      mr: 'मराठी'
    }
  }
};

const AIJusticeBot = () => {
  const API_BASE_URL = 'http://localhost:5000';

  const [message, setMessage] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [uiLanguage, setUiLanguage] = useState('en');
  const [detectedLanguage, setDetectedLanguage] = useState('en');
  const [translationError, setTranslationError] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [documentResponse, setDocumentResponse] = useState(null);

  useEffect(() => {
    const browserLang = navigator.language.split('-')[0];
    if (UI_STRINGS[browserLang]) {
      setUiLanguage(browserLang);
      setDetectedLanguage(browserLang);
    }
  }, []);

  const strings = useMemo(() => UI_STRINGS[uiLanguage] || UI_STRINGS.en, [uiLanguage]);

  const languageLabel = (langCode) => {
    return strings.languageNames[langCode] || langCode;
  };

  const resetForNewInteraction = () => {
    setTranslationError(false);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    resetForNewInteraction();
    setDocumentResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/legal-advice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim() })
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      const responseLanguage = data.language && UI_STRINGS[data.language] ? data.language : uiLanguage;
      if (!data.translation_error && responseLanguage) {
        setUiLanguage(responseLanguage);
      }
      setDetectedLanguage(data.detected_language || responseLanguage);
      setTranslationError(Boolean(data.translation_error));
      const displayInput = data.translated_input || data.query || message;
      setHistory([...history, { query: message, response: data }]);
      setResponse({ ...data, translated_input: displayInput });
      setMessage('');
    } catch (error) {
      console.error('Error:', error);
      setResponse({
        status: 'error',
        response: strings.error,
        error: error.message
      });
      setTranslationError(false);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const uploadDocument = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('document', selectedFile);

    setUploading(true);
    resetForNewInteraction();
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/analyze-document`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      const responseLanguage = data.language && UI_STRINGS[data.language] ? data.language : uiLanguage;
      if (!data.translation_error && responseLanguage) {
        setUiLanguage(responseLanguage);
      }
      setDetectedLanguage(data.detected_language || responseLanguage);
      setTranslationError(Boolean(data.translation_error));
      const displayInput = data.translated_input || data.original_input || '';
      setDocumentResponse({ ...data, translated_input: displayInput });
    } catch (error) {
      console.error('Error analysing document:', error);
      setDocumentResponse({
        status: 'error',
        response: strings.error,
        error: error.message
      });
      setTranslationError(false);
    } finally {
      setUploading(false);
    }
  };

  const formatResponse = (responseData, { activeLanguage = uiLanguage, isDocument = false } = {}) => {
    if (!responseData) {
      return null;
    }

    const translatedOutput = responseData.response;
    if (!translatedOutput) {
      return <p>{strings.error}</p>;
    }

    const translatedInput = responseData.translated_input;
    const originalInput = isDocument ? responseData.original_input : responseData.query;
    const inputLabel = isDocument ? strings.documentInputLabel : strings.questionLabel;
    const originalLabel = isDocument ? strings.originalDocumentLabel : strings.originalQuestion;
    const supportingSnippets = responseData.supporting_snippets || [];

    return (
      <div className="response-container">
        {translatedInput && (
          <div className="user-input">
            <strong>{inputLabel}:</strong> {translatedInput}
          </div>
        )}

        {originalInput && activeLanguage !== 'en' && originalInput !== translatedInput && (
          <details className="original-input">
            <summary>{originalLabel}</summary>
            <p>{originalInput}</p>
          </details>
        )}

        <div className="response-text">
          {translatedOutput.split('\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>

        {supportingSnippets.length > 0 && (
          <div className="supporting">
            <h4>{strings.referencesTitle}</h4>
            {supportingSnippets.map((snippet, index) => (
              <div key={index} className="snippet-item">
                <p>
                  <strong>{snippet.source || 'Legal Document'}:</strong> {snippet.snippet}
                </p>
                {snippet.original_snippet && activeLanguage !== 'en' && (
                  <details className="snippet-original">
                    <summary>{strings.englishReference}</summary>
                    <p>{snippet.original_snippet}</p>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}

        {responseData.original_response && activeLanguage !== 'en' && (
          <details className="original-text">
            <summary>{strings.originalResponse}</summary>
            <p>{responseData.original_response}</p>
          </details>
        )}
      </div>
    );
  };

  const displayLanguage = response?.language || documentResponse?.language || uiLanguage;
  const displayDetected = response?.detected_language || documentResponse?.detected_language || detectedLanguage;
  const displayProcessed = response?.processed_query || documentResponse?.processed_query;

  return (
    <div className="ai-justice-bot">
      <div className="header-bar">
        <h2>🏛️ {strings.title}</h2>
      </div>

      <p>{strings.subtitle}</p>

      {(response || documentResponse) && (
        <div className="language-badge">
          <span>{strings.detectedLabel}: {languageLabel(displayDetected)}</span>
          <span>{strings.responseLabel}: {languageLabel(displayLanguage)}</span>
          {translationError && displayDetected !== 'en' && displayLanguage === 'en' && (
            <span className="translation-warning">{strings.translationFallback}</span>
          )}
          {displayProcessed && (
            <span className="processed-query">{strings.processedQueryLabel}: {displayProcessed}</span>
          )}
        </div>
      )}

      <div className="input-section">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={strings.placeholder}
          rows={4}
          className="message-input"
        />

        <div className="input-actions">
          <button
            onClick={sendMessage}
            disabled={loading || uploading || !message.trim()}
            className="send-button"
          >
            {loading ? strings.processing : strings.send}
          </button>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            className="clear-history"
            onClick={() => {
              setHistory([]);
              setResponse(null);
              setDocumentResponse(null);
              setDetectedLanguage(uiLanguage);
              setTranslationError(false);
            }}
          >
            {strings.clear}
          </button>
        )}
      </div>

      {response ? (
        <div className="response-section">
          {formatResponse(response, { activeLanguage: response.language || uiLanguage, isDocument: false })}
        </div>
      ) : (
        <div className="empty-state">
          <p>{strings.empty}</p>
        </div>
      )}

      <div className="upload-section">
        <h3>{strings.uploadTitle}</h3>
        <p className="upload-help">{strings.uploadDescription}</p>
        <input
          type="file"
          accept=".pdf,image/*"
          onChange={handleFileChange}
          disabled={uploading || loading}
        />
        <button
          type="button"
          className="upload-button"
          onClick={uploadDocument}
          disabled={!selectedFile || uploading || loading}
        >
          {uploading ? strings.uploading : strings.uploadButton}
        </button>
      </div>

      <div className="document-section">
        <h3>{strings.documentAnalysis}</h3>
        {documentResponse ? (
          <div className="response-section">
            {documentResponse.document_preview && (
              <details className="document-preview">
                <summary>{strings.originalDocumentLabel}</summary>
                <p>{documentResponse.document_preview}</p>
              </details>
            )}
            {formatResponse(documentResponse, { activeLanguage: documentResponse.language || uiLanguage, isDocument: true })}
          </div>
        ) : (
          <p className="empty-state">{strings.noDocument}</p>
        )}
      </div>

      <style jsx>{`
        .ai-justice-bot {
          max-width: 900px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .header-bar {
          display: flex;
          flex-wrap: wrap;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
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
          font-family: inherit;
        }

        .input-actions {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
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
        }

        .send-button:disabled,
        .upload-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .clear-history {
          margin-top: 10px;
          background: #6c757d;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
        }

        .language-badge {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
          padding: 12px;
          background: #e7f3ff;
          border-radius: 8px;
          margin: 15px 0;
          font-size: 14px;
        }

        .language-badge span {
          padding: 4px 8px;
          background: white;
          border-radius: 4px;
        }

        .translation-warning {
          background: #fff3cd !important;
          color: #856404;
        }

        .response-section {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          margin: 20px 0;
        }

        .user-input {
          background: #e3f2fd;
          padding: 12px;
          border-radius: 6px;
          margin-bottom: 15px;
          border-left: 4px solid #2196f3;
        }

        .response-text p {
          margin: 10px 0;
          line-height: 1.6;
        }

        .supporting {
          margin-top: 20px;
          padding: 15px;
          background: white;
          border-radius: 6px;
        }

        .snippet-item {
          margin: 10px 0;
          padding: 12px;
          background: #f8f9fa;
          border-left: 3px solid #28a745;
          border-radius: 4px;
        }

        details {
          margin: 10px 0;
          padding: 10px;
          background: #fff;
          border: 1px solid #dee2e6;
          border-radius: 4px;
        }

        summary {
          cursor: pointer;
          font-weight: 600;
          color: #495057;
        }

        .empty-state {
          text-align: center;
          padding: 40px;
          color: #6c757d;
        }

        .upload-section {
          margin: 30px 0;
          padding: 20px;
          background: #f1f3f5;
          border-radius: 8px;
        }

        .upload-help {
          color: #6c757d;
          margin: 10px 0;
        }

        .upload-button {
          background: #28a745;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 6px;
          cursor: pointer;
          margin-top: 10px;
        }

        .document-section {
          margin: 30px 0;
          padding: 20px;
          background: #fff;
          border: 2px solid #e9ecef;
          border-radius: 8px;
        }

        .document-preview {
          margin-bottom: 15px;
          background: #f8f9fa;
        }

        .processed-query {
          font-style: italic;
          color: #6c757d;
        }
      `}</style>
    </div>
  );
};

export default AIJusticeBot;
