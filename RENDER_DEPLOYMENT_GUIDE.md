# 🚀 Legal AI API - Render.com Deployment Guide

## 📋 Overview
This guide will help you deploy your Legal AI API to Render.com for free hosting with global access.

## 🔧 Prerequisites
- ✅ GitHub account
- ✅ Render.com account (free)
- ✅ Your code pushed to GitHub repository

## 📁 Deployment Files Created
- `production_api.py` - Production-ready Flask API
- `requirements.txt` - Python dependencies
- `Procfile` - Render deployment configuration

## 🚀 Step-by-Step Deployment

### Step 1: Push Code to GitHub
```bash
git add .
git commit -m "Add production API for Render deployment"
git push origin main
```

### Step 2: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub (recommended)
3. Authorize Render to access your repositories

### Step 3: Deploy Web Service
1. **Click "New +"** in Render dashboard
2. **Select "Web Service"**
3. **Connect Repository**: Select your `LawAi` repository
4. **Configure Service**:
   - **Name**: `legal-ai-api` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn production_api:app`

### Step 4: Environment Settings
- **Plan**: Select "Free" (0$/month)
- **Auto-Deploy**: Enable (deploys on every push)

### Step 5: Deploy
1. Click "Create Web Service"
2. Wait 3-5 minutes for deployment
3. Your API will be live at: `https://your-service-name.onrender.com`

## 🌐 Your Live API Endpoints

Once deployed, your API will be available at:
```
Base URL: https://legal-ai-api.onrender.com
```

### Available Endpoints:
- **GET** `/` - API information
- **GET** `/health` - Health check
- **POST** `/legal-advice` - Get legal advice
- **POST** `/analyze` - Analyze legal cases

## 🔌 React Integration

### Update your React API calls from:
```javascript
// Old localhost URL
const API_BASE = 'http://localhost:5000';
```

### To your live Render URL:
```javascript
// New live URL (replace with your actual URL)
const API_BASE = 'https://legal-ai-api.onrender.com';
```

### Example usage:
```javascript
// Legal advice request
const response = await fetch(`${API_BASE}/legal-advice`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: "I got hacked and need legal help"
  })
});
const data = await response.json();
```

## ⚡ Performance Notes

### Cold Starts
- **First request after 15 minutes**: ~30-60 seconds
- **Subsequent requests**: <200ms response time
- **Tip**: Keep alive with periodic health checks

### Keeping Service Warm (Optional)
Add this to your React app to ping the API every 10 minutes:
```javascript
// Keep API warm
setInterval(() => {
  fetch('https://your-api-url.onrender.com/health')
    .catch(() => {}); // Ignore errors
}, 10 * 60 * 1000); // 10 minutes
```

## 🔄 Automatic Updates

**Benefits of GitHub Integration:**
- Push code → Automatic deployment
- No manual intervention needed
- Deployment logs available in Render dashboard

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Fails**
   - Check `requirements.txt` format
   - Ensure `Procfile` is correct
   - View logs in Render dashboard

2. **Service Won't Start**
   - Check `production_api.py` syntax
   - Verify port configuration (uses `PORT` env variable)

3. **CORS Issues**
   - API configured for all origins (`origins=["*"]`)
   - Supports all common headers

### Logs Access:
- Go to Render dashboard
- Select your service
- Click "Logs" tab for real-time debugging

## 💰 Cost Breakdown

**Free Tier Includes:**
- 750 hours/month (enough for 24/7)
- Custom domain support
- Automatic SSL certificates
- GitHub integration

**Upgrade Options:**
- $7/month: No sleep, more resources
- Custom domains with your own DNS

## 🔐 Security Features

- **HTTPS by default** (SSL certificates)
- **Environment variables** for sensitive data
- **CORS properly configured**
- **No sensitive data in code**

## 📊 Monitoring

**Available in Render Dashboard:**
- Real-time logs
- Resource usage
- Deployment history
- Performance metrics

---

## 🎉 Next Steps

1. **Deploy to Render** following steps above
2. **Get your live API URL**
3. **Update React app** with new endpoint
4. **Test all functionality**
5. **Share your live legal AI assistant!**

Your Legal AI API will be accessible worldwide with professional-grade hosting! 🌍⚖️