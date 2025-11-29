# Frontend Deployment Guide for https://lawai.nexverse.in/ask

## Changes Made to AIJusticeBot Component

### ✅ Features Implemented

1. **Voice Button Removed** - No voice input functionality (as requested)

2. **Automatic Language Detection**
   - Detects browser language on page load
   - Auto-detects input language (Hindi, Marathi, English)
   - UI adapts to detected language

3. **Marathi Translation Support**
   - Complete Marathi UI strings
   - Backend integration with MarianMT for Marathi translation
   - All text elements translated (buttons, labels, placeholders)

4. **Full Multilingual Support**
   - Input queries translated and displayed
   - Responses translated to detected language
   - Legal references/snippets translated
   - Original English text available in collapsible sections

5. **Document Upload Feature**
   - Upload PDF or image files
   - OCR for images (requires pytesseract on backend)
   - PDF text extraction
   - Document content translated
   - Document preview available

### 📋 Required Backend Changes

Ensure your production backend (`simple_legal_api.py`) has:

1. **Translation Dependencies**
```python
from deep_translator import GoogleTranslator
from transformers import MarianMTModel, MarianTokenizer
```

2. **Marathi Translation Function**
```python
def translate_to_marathi(text):
    """Translate English text to Marathi using MarianMT"""
    try:
        model_name = 'Helsinki-NLP/opus-mt-en-mr'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return text
```

3. **Language Detection Function**
```python
def detect_language(text):
    """Heuristic language detection"""
    # Hindi characters
    if any('\u0900' <= char <= '\u097F' for char in text):
        return 'hi'
    # Marathi (same Unicode block but context-based)
    if any('\u0900' <= char <= '\u097F' for char in text):
        # Could be Hindi or Marathi - check for Marathi-specific words
        marathi_indicators = ['आहे', 'आहेत', 'होते', 'करतो', 'करते']
        if any(word in text for word in marathi_indicators):
            return 'mr'
        return 'hi'
    return 'en'
```

4. **Document Analysis Endpoint**
```python
@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    # Handle file upload
    # Extract text (PDF or OCR for images)
    # Detect language
    # Translate input and response
    # Return formatted response
```

### 🚀 Deployment Steps

1. **Update Frontend Code**
   - Replace your production `AIJusticeBot.jsx` with the new version
   - File location: `c:\Users\RIYA\OneDrive\Desktop\LawAi\react_example\AIJusticeBot.jsx`

2. **Update Backend API**
   - Deploy updated `simple_legal_api.py` to production
   - File location: `c:\Users\RIYA\OneDrive\Desktop\LawAi\simple_legal_api.py`

3. **Install Backend Dependencies**
```bash
pip install PyPDF2 Pillow pytesseract deep-translator transformers torch
```

4. **Install Tesseract OCR** (for image processing)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `apt-get install tesseract-ocr`
   - Update path in code if needed

5. **Update API Base URL**
   - In production frontend, change:
   ```javascript
   const API_BASE_URL = 'https://your-production-api.com';
   ```
   - Current value: `http://localhost:5000` (for local development)

### 🧪 Testing Checklist

- [ ] Test Hindi input and response
- [ ] Test Marathi input and response
- [ ] Test English input and response
- [ ] Test document upload (PDF)
- [ ] Test document upload (Image with OCR)
- [ ] Verify translations are accurate
- [ ] Check UI adapts to detected language
- [ ] Confirm no voice button appears
- [ ] Test language badge displays correctly
- [ ] Verify original text collapsibles work

### 📝 Component Features

**New UI Elements:**
- Language detection badge (shows detected & response language)
- Document upload section
- Document analysis section with preview
- Translated input display
- Original input/text in collapsible sections
- Translation fallback warnings

**Supported Languages:**
- English (en)
- Hindi (hi)
- Marathi (mr)

**API Endpoints Used:**
- `POST /api/legal-advice` - Text query processing
- `POST /api/analyze-document` - Document upload & analysis

### 🔧 Configuration

Update these constants in your production build:

```javascript
// In AIJusticeBot.jsx
const API_BASE_URL = 'YOUR_PRODUCTION_API_URL';

// Backend should return responses in this format:
{
  "response": "Translated response text",
  "original_response": "Original English text",
  "translated_input": "Translated user input",
  "query": "Original user input",
  "language": "hi" | "mr" | "en",
  "detected_language": "hi" | "mr" | "en",
  "translation_error": false,
  "supporting_snippets": [
    {
      "snippet": "Translated legal reference",
      "original_snippet": "English legal reference",
      "source": "Section/Act name"
    }
  ]
}
```

### 📦 Files to Deploy

1. **Frontend:** `react_example/AIJusticeBot.jsx` → Your React app
2. **Backend:** `simple_legal_api.py` → Your API server
3. **Dependencies:** Update `requirements.txt` in production

### ⚠️ Important Notes

- The component uses `<style jsx>` - ensure your build system supports it
- For production, consider extracting styles to CSS file
- Update CORS settings on backend to allow your production domain
- Test thoroughly before deploying to production
- Consider caching translation models to improve performance
- Add rate limiting for document upload endpoint

---

**Local Development Server Running:**
- Backend API: http://localhost:5000
- Health check: http://localhost:5000/health
- Model: 6611 legal documents loaded
- Features: Hindi, Marathi, English translation with document upload

**Next Steps:**
1. Build your React app with the updated component
2. Deploy backend with all dependencies
3. Update API_BASE_URL in frontend
4. Deploy to https://lawai.nexverse.in/ask
