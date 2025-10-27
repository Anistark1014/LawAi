from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/respond', methods=['POST'])
def respond():
    data = request.get_json() or {}
    message = data.get('message', '')

    # Very small heuristic: if the message contains the word 'stolen' -> return a GUIDE, else ask for clarification
    if 'stolen' in message.lower() or 'cheat' in message.lower() or 'hacked' in message.lower():
        reply = (
            "GUIDE: It appears you may be a victim of an online theft or cheating. "
            "Applicable laws may include sections of the IT Act and IPC. "
            "Next steps: collect evidence, report to police, preserve logs."
        )
    else:
        reply = "CLARIFY: Could you provide the platform where this happened and approximate date/time?"

    return jsonify({"response": reply})


if __name__ == '__main__':
    app.run(port=9000, debug=True)
