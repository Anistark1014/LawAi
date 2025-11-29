# Translation Integration Guide

## How to Use Translations in Your Main Application

The Learn and News buttons (and all other UI elements) can now be translated using the provided translation files.

### Quick Start

```javascript
// Import the translation helper
import { getTranslation, getTranslationStrings, detectBrowserLanguage } from './react_example/translations';

// Or if you have the component
import AIJusticeBot, { UI_STRINGS, getTranslationStrings } from './react_example/AIJusticeBot';

// Get current language (from state, browser, or user selection)
const currentLanguage = detectBrowserLanguage(); // Returns 'en', 'hi', or 'mr'

// Get a specific translation
const learnText = getTranslation('learn', currentLanguage);
const newsText = getTranslation('news', currentLanguage);

// Get all strings for a language
const strings = getTranslationStrings(currentLanguage);
```

### Example: Translating Navigation Buttons

```jsx
import React, { useState, useEffect } from 'react';
import { getTranslation, detectBrowserLanguage } from './react_example/translations';

const Navigation = () => {
  const [language, setLanguage] = useState('en');

  useEffect(() => {
    setLanguage(detectBrowserLanguage());
  }, []);

  return (
    <nav>
      <button>{getTranslation('home', language)}</button>
      <button>{getTranslation('learn', language)}</button>
      <button>{getTranslation('news', language)}</button>
      <button>{getTranslation('about', language)}</button>
      <button>{getTranslation('contact', language)}</button>
      <button>{getTranslation('help', language)}</button>
    </nav>
  );
};
```

### Example: Using with Context/State Management

```jsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { getTranslationStrings, detectBrowserLanguage } from './react_example/translations';

// Create Translation Context
const TranslationContext = createContext();

export const TranslationProvider = ({ children }) => {
  const [language, setLanguage] = useState('en');
  const [strings, setStrings] = useState({});

  useEffect(() => {
    const detectedLang = detectBrowserLanguage();
    setLanguage(detectedLang);
    setStrings(getTranslationStrings(detectedLang));
  }, []);

  const changeLanguage = (newLang) => {
    setLanguage(newLang);
    setStrings(getTranslationStrings(newLang));
  };

  return (
    <TranslationContext.Provider value={{ language, strings, changeLanguage }}>
      {children}
    </TranslationContext.Provider>
  );
};

// Custom hook to use translations
export const useTranslation = () => useContext(TranslationContext);

// Usage in components
const MyComponent = () => {
  const { strings } = useTranslation();
  
  return (
    <div>
      <button>{strings.learn}</button>
      <button>{strings.news}</button>
    </div>
  );
};
```

### Available Translation Keys

#### Navigation
- `learn` - Learn/सीखें/शिका
- `news` - News/समाचार/बातम्या
- `home` - Home/होम/मुख्यपृष्ठ
- `about` - About/हमारे बारे में/आमच्याबद्दल
- `contact` - Contact/संपर्क करें/संपर्क
- `help` - Help/सहायता/मदत

#### Common UI
- `title` - AI Justice Bot title
- `subtitle` - Subtitle text
- `send` - Send button
- `clear` - Clear button
- `processing` - Processing message
- `error` - Error message

#### Documents
- `uploadTitle` - Upload section title
- `uploadButton` - Upload button
- `uploading` - Uploading status
- `documentAnalysis` - Document analysis title

### Supported Languages

| Code | Language | Example |
|------|----------|---------|
| `en` | English | Learn, News, Home |
| `hi` | Hindi (हिंदी) | सीखें, समाचार, होम |
| `mr` | Marathi (मराठी) | शिका, बातम्या, मुख्यपृष्ठ |

### Integration with Your Production App

If your production app at https://lawai.nexverse.in/ask has Learn and News buttons that aren't translating:

1. **Import the translation file:**
   ```javascript
   import { getTranslation } from './translations';
   ```

2. **Get the current language from the AIJusticeBot component:**
   ```javascript
   // The AIJusticeBot component auto-detects language
   // You can access it via state or props
   ```

3. **Update your button text:**
   ```jsx
   <button>{getTranslation('learn', currentLanguage)}</button>
   <button>{getTranslation('news', currentLanguage)}</button>
   ```

### Full Example: Main App Integration

```jsx
import React, { useState, useEffect } from 'react';
import AIJusticeBot from './react_example/AIJusticeBot';
import { getTranslationStrings, detectBrowserLanguage } from './react_example/translations';

const App = () => {
  const [language, setLanguage] = useState('en');
  const [strings, setStrings] = useState({});

  useEffect(() => {
    const browserLang = detectBrowserLanguage();
    setLanguage(browserLang);
    setStrings(getTranslationStrings(browserLang));
  }, []);

  // Listen for language changes from AIJusticeBot
  const handleLanguageChange = (newLang) => {
    setLanguage(newLang);
    setStrings(getTranslationStrings(newLang));
  };

  return (
    <div className="app">
      <nav className="main-nav">
        <button>{strings.home}</button>
        <button>{strings.learn}</button>
        <button>{strings.news}</button>
        <button>{strings.about}</button>
        <button>{strings.contact}</button>
      </nav>
      
      <AIJusticeBot onLanguageChange={handleLanguageChange} />
    </div>
  );
};

export default App;
```

### Notes

1. The translations automatically detect the browser language on page load
2. When a user types in Hindi or Marathi, the backend detects it and returns the language code
3. Your main app should sync with the detected language
4. All translations maintain consistent UI/UX across the application

### Testing

Test with different languages:
```javascript
// English
console.log(getTranslation('learn', 'en')); // "Learn"

// Hindi
console.log(getTranslation('learn', 'hi')); // "सीखें"

// Marathi
console.log(getTranslation('learn', 'mr')); // "शिका"
```
