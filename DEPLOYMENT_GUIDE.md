# AI Justice Bot - Deployment Guide

## 🚀 Quick Setup for React Integration

Your AI Justice Bot is now ready to be hosted as a standalone API that your React website can call.

### What You Have

✅ **Standalone API Server** (`standalone_api.py`) - No Gemini dependency
✅ **Rule-based Legal Knowledge** - 5 major cyber crime types covered
✅ **CORS Support** - Ready for React frontend calls
✅ **Structured Responses** - JSON format with crime details
✅ **React Example Component** - Ready-to-use integration code

### API Endpoints Your React App Can Use

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/legal-advice` | Main endpoint - send user queries, get legal guidance |
| GET | `/api/health` | Health check for your React app |
| GET | `/api/crimes` | List all supported crime types |
| POST | `/chat` | Legacy endpoint (backward compatibility) |

### Request/Response Format

**Request to `/api/legal-advice`:**
```json
{
  "message": "Someone hacked my email account",
  "history": [],
  "language": "English"
}
```

**Response:**
```json
{
  "type": "GUIDE",
  "response": "GUIDE: Based on your description, this appears to be a case of Unauthorized Access/Hacking...",
  "crime_type": "Unauthorized Access/Hacking", 
  "sections": ["IT Act Section 66", "IT Act Section 43"],
  "punishment": "Up to 3 years imprisonment and fine up to Rs. 5 lakh",
  "timestamp": "2025-10-28T...",
  "status": "success"
}
```

## 🖥️ How to Host/Deploy

### Option 1: Local Development
```bash
# Install dependencies
pip install flask flask-cors python-dotenv requests

# Run the server
python standalone_api.py

# Server runs on http://localhost:5000
```

### Option 2: Production Hosting

1. **Using Gunicorn (Recommended for Linux/Cloud)**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 standalone_api:app
```

2. **Using Waitress (Windows compatible)**:
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 standalone_api:app
```

3. **Cloud Platforms**:
   - **Heroku**: Add `Procfile` with `web: gunicorn standalone_api:app`
   - **Railway/Render**: Point to `standalone_api.py`
   - **AWS/Azure**: Use container deployment
   - **VPS**: Use nginx + gunicorn

### Option 3: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "standalone_api:app"]
```

## 🔧 Integration with Your React Website

### Step 1: Add the API Component
Copy `react_example/AIJusticeBot.jsx` to your React project.

### Step 2: Update API URL
In your React component, change:
```javascript
const API_BASE_URL = 'https://your-api-domain.com';  // Your hosted API URL
```

### Step 3: Use in Your React App
```jsx
import AIJusticeBot from './components/AIJusticeBot';

function App() {
  return (
    <div className="App">
      <AIJusticeBot />
    </div>
  );
}
```

### Step 4: Handle CORS (if needed)
The API already includes CORS headers, but if you have issues:
1. Update `CORS(app)` in `standalone_api.py` to specify your React domain
2. Or add environment variable for allowed origins

## 🎯 Testing Your Integration

1. **Start the API server**:
   ```bash
   python standalone_api.py
   ```

2. **Test with curl**:
   ```bash
   curl -X POST http://localhost:5000/api/legal-advice \
   -H "Content-Type: application/json" \
   -d '{"message": "my account was hacked", "language": "English"}'
   ```

3. **Test from React**: Use the provided component

## 📊 Supported Crime Types

The API currently handles these cyber crimes with structured responses:

1. **Online Financial Fraud** - Keywords: fraud, scam, fake, money, payment
2. **Identity Theft** - Keywords: identity, personal, details, stolen
3. **Hacking/Unauthorized Access** - Keywords: hack, account, unauthorized, breach
4. **Cyberbullying** - Keywords: bully, harass, threat, abuse
5. **Data Theft** - Keywords: data, information, stolen, leaked, privacy

## 🔄 Extending the Knowledge Base

To add more crime types, edit the `CYBER_CRIMES_DB` dictionary in `standalone_api.py`:

```python
CYBER_CRIMES_DB["new_crime"] = {
    "keywords": ["keyword1", "keyword2"],
    "crime_type": "New Crime Type",
    "sections": ["Relevant Law Sections"],
    "punishment": "Punishment details",
    "process": ["Step 1", "Step 2", "Step 3"]
}
```

## ⚠️ Important Notes

- **No External Dependencies**: This API works without Gemini/OpenAI keys
- **Rule-based**: Responses are based on keyword matching, not AI generation
- **Extensible**: Easy to add more crime types and legal knowledge
- **Production Ready**: Includes error handling, CORS, health checks
- **Structured Output**: Perfect for React UI components

## 🚀 Next Steps

1. Deploy the API to your preferred hosting platform
2. Update your React app to use the live API URL
3. Test the integration end-to-end
4. Customize the knowledge base for your specific needs
5. Add authentication/rate limiting if needed for production

Your AI Justice Bot is ready to help users with cyber crime legal guidance! 🏛️