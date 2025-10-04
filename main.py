# d:\Justicebot\main.py

import asyncio
import random
from flask import Flask, request, jsonify, render_template_string

# --- App Initialization ---
app = Flask(__name__)

# --- HTML & JavaScript Frontend Template ---
# This string contains a complete webpage that will be sent to the browser.
# It includes a chat interface and the JavaScript to communicate with our Flask backend.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JusticeBot Interface</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        #chat-container { width: 90%; max-width: 600px; height: 80vh; background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        #chat-box { flex-grow: 1; padding: 20px; overflow-y: auto; border-bottom: 1px solid #eee; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 18px; max-width: 80%; line-height: 1.5; }
        .user-message { background-color: #007bff; color: white; align-self: flex-end; margin-left: auto; }
        .bot-message { background-color: #e9ecef; color: #333; align-self: flex-start; }
        .typing-indicator { color: #888; font-style: italic; }
        #input-area { display: flex; padding: 15px; border-top: 1px solid #eee; }
        #user-input { flex-grow: 1; border: 1px solid #ddd; border-radius: 20px; padding: 10px 15px; font-size: 16px; outline: none; }
        #send-button { background-color: #007bff; color: white; border: none; padding: 10px 20px; margin-left: 10px; border-radius: 20px; cursor: pointer; font-size: 16px; }
        #send-button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="chat-box">
             <div class="message bot-message">Hello! How can I assist you with legal information today?</div>
        </div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off">
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        async function sendMessage() {
            const messageText = userInput.value.trim();
            if (messageText === '') return;

            // Display user message
            addMessage(messageText, 'user-message');
            userInput.value = '';

            // Display typing indicator
            const typingDiv = addMessage('Bot is typing...', 'bot-message typing-indicator');
            
            try {
                // Send message to backend
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: messageText })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // Remove typing indicator and display bot response
                chatBox.removeChild(typingDiv);
                addMessage(data.reply, 'bot-message');

            } catch (error) {
                chatBox.removeChild(typingDiv);
                addMessage('Sorry, an error occurred. Please try again.', 'bot-message');
                console.error('Fetch error:', error);
            }
        }

        function addMessage(text, className) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', className);
            messageDiv.textContent = text;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to bottom
            return messageDiv;
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# --- Backend Logic ---

@app.route('/')
def home():
    """Serves the main chat page."""
    return render_template_string(HTML_TEMPLATE)

async def get_bot_response(message: str) -> str:
    """
    A placeholder function to simulate getting a response from an AI.
    It waits for a short, random duration to feel more realistic.
    """
    print(f"Received message: '{message}'. Processing asynchronously...")
    delay = random.uniform(0.5, 2.0)  # Simulate network/model latency
    await asyncio.sleep(delay)
    
    # Example canned responses
    responses = [
        f"Regarding '{message}', the standard procedure involves...",
        f"Thank you for asking about '{message}'. Here is some general information...",
        f"I have processed your query about '{message}'. The legal precedent suggests...",
        "That is an interesting question. Could you provide more specific details?",
    ]
    response = random.choice(responses)
    
    print("Processing complete. Sending reply.")
    return response

@app.route('/chat', methods=['POST'])
async def chat():
    """
    This is the asynchronous API endpoint for the chatbot.
    It receives a message and returns a simulated AI response.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Missing 'message' key in JSON body"}), 400

    # Get the response from our async function
    bot_reply = await get_bot_response(user_message)

    return jsonify({"reply": bot_reply})

# --- Main Execution ---

if __name__ == '__main__':
    # The debug=True flag enables the interactive debugger and auto-reloading.
    # The host='0.0.0.0' makes the server accessible on your local network.
    app.run(host='0.0.0.0', port=5000, debug=True)
