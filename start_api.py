#!/usr/bin/env python3
"""
Simple API Server Starter
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_legal_api import app

if __name__ == '__main__':
    # ASCII-only logging to avoid encoding issues on Windows shells
    print("Starting Legal AI API Server...")
    print("URL: http://localhost:5000")
    print("React Integration: Use endpoints in REACT_INTEGRATION_GUIDE.txt")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        use_reloader=False
    )