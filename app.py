from flask import Flask, render_template, request, jsonify
import json
import os
import random

app = Flask(__name__, template_folder='Static')

# Load the intents JSON
def load_intents():
    try:
        with open('intents.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"intents": []}
    except json.JSONDecodeError:
        return {"intents": []}

# Find the matching intent based on user input
def find_intent(user_input, intents):
    user_input_words = set(user_input.lower().split())
    best_match = None
    max_matches = 0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = set(pattern.lower().split())
            common_words = user_input_words.intersection(pattern_words)
            if len(common_words) > max_matches:
                max_matches = len(common_words)
                best_match = intent['tag']

    return best_match

# Get a random response based on the matched intent
def get_response(intent_tag, intents):
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Ngiyaxolisa angisiqondi isicelo sakho"

# Route for the chatbot interface
@app.route("/")
def index():
    return render_template('index.html')

# Handle chatbot POST requests for user interaction
@app.route("/get", methods=["POST"])
def chatbot():
    user_input = request.json.get('input', '').lower()

    # Load intents
    intents = load_intents()

    # Find intent based on user input
    intent = find_intent(user_input, intents)

    # Generate a response based on the identified intent
    if intent:
        response = get_response(intent, intents)
    else:
        response = "Ngiyaxolisa angisiqondi isicelo sakho"

    return jsonify({'response': response})

# Webhook for external data or input if needed
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    # Load intents
    intents = load_intents()

    # Find intent and get response
    intent = find_intent(user_message, intents)
    response = get_response(intent, intents) if intent else "Ngiyaxolisa angiqondi."

    return jsonify({'reply': response})

# Handle updating intents data via API
@app.route('/update', methods=['POST'])
def update_data():
    new_data = request.get_json()
    with open('intents.json', 'r+') as file:
        data = json.load(file)
        data['intents'].extend(new_data.get('intents', []))  # Append new intents
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
