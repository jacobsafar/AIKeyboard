from flask import Flask, render_template, request, jsonify, session
import os
import time
from keyboard_predictor import KeyboardPredictor

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize the keyboard predictor
predictor = KeyboardPredictor()

@app.route('/')
def index():
    """Main page with the keyboard interface"""
    # Initialize session variables if not present
    if 'button_sequence' not in session:
        session['button_sequence'] = []
    if 'current_word' not in session:
        session['current_word'] = ""
    if 'predicted_words' not in session:
        session['predicted_words'] = []
    if 'next_word_predictions' not in session:
        session['next_word_predictions'] = []
    if 'typed_text' not in session:
        session['typed_text'] = ""
    if 'start_time' not in session:
        session['start_time'] = time.time()
    if 'word_count' not in session:
        session['word_count'] = 0
    
    return render_template('index.html')

@app.route('/press_button', methods=['POST'])
def press_button():
    """Handle button press and return AI prediction"""
    try:
        data = request.get_json()
        button_num = data.get('button')
        
        if button_num not in [1, 2, 3, 4]:
            return jsonify({'error': 'Invalid button number'}), 400
        
        # Add button to sequence
        session['button_sequence'].append(button_num)
        
        # Get prediction from AI
        result = predictor.predict_word(session['button_sequence'])
        session['current_word'] = result.get('current_word', '')
        session['predicted_words'] = result.get('alternative_words', [])
        
        # Get next word predictions if we have a complete word
        next_words = []
        if session['current_word']:
            next_words = predictor.predict_next_words(
                session['typed_text'], 
                session['current_word']
            )
        session['next_word_predictions'] = next_words
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'current_word': session['current_word'],
            'alternative_words': session['predicted_words'],
            'next_word_predictions': session['next_word_predictions'],
            'button_sequence': session['button_sequence'],
            'typed_text': session['typed_text'],
            'word_count': session['word_count'],
            'elapsed_time': round(elapsed_time, 1),
            'wpm': round(wpm, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/accept_word', methods=['POST'])
def accept_word():
    """Accept the current predicted word"""
    try:
        data = request.get_json()
        word = data.get('word', session.get('current_word', ''))
        
        if word:
            # Add word to typed text
            if session['typed_text']:
                session['typed_text'] += " " + word
            else:
                session['typed_text'] = word
            
            # Reset for next word
            session['button_sequence'] = []
            session['current_word'] = ""
            session['predicted_words'] = []
            session['next_word_predictions'] = []
            session['word_count'] += 1
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'typed_text': session['typed_text'],
            'word_count': session['word_count'],
            'elapsed_time': round(elapsed_time, 1),
            'wpm': round(wpm, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/backspace', methods=['POST'])
def backspace():
    """Remove last button press"""
    try:
        if session['button_sequence']:
            session['button_sequence'].pop()
            
            if session['button_sequence']:
                # Re-predict with remaining sequence
                result = predictor.predict_word(session['button_sequence'])
                session['current_word'] = result.get('current_word', '')
                session['predicted_words'] = result.get('alternative_words', [])
            else:
                session['current_word'] = ""
                session['predicted_words'] = []
                session['next_word_predictions'] = []
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'current_word': session['current_word'],
            'alternative_words': session['predicted_words'],
            'next_word_predictions': session['next_word_predictions'],
            'button_sequence': session['button_sequence'],
            'typed_text': session['typed_text'],
            'word_count': session['word_count'],
            'elapsed_time': round(elapsed_time, 1),
            'wpm': round(wpm, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/new_word', methods=['POST'])
def new_word():
    """Start a new word (clear current sequence)"""
    try:
        session['button_sequence'] = []
        session['current_word'] = ""
        session['predicted_words'] = []
        session['next_word_predictions'] = []
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'current_word': session['current_word'],
            'alternative_words': session['predicted_words'],
            'next_word_predictions': session['next_word_predictions'],
            'button_sequence': session['button_sequence'],
            'typed_text': session['typed_text'],
            'word_count': session['word_count'],
            'elapsed_time': round(elapsed_time, 1),
            'wpm': round(wpm, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all', methods=['POST'])
def clear_all():
    """Clear everything and start over"""
    try:
        session['button_sequence'] = []
        session['current_word'] = ""
        session['predicted_words'] = []
        session['next_word_predictions'] = []
        session['typed_text'] = ""
        session['start_time'] = time.time()
        session['word_count'] = 0
        
        return jsonify({
            'current_word': session['current_word'],
            'alternative_words': session['predicted_words'],
            'next_word_predictions': session['next_word_predictions'],
            'button_sequence': session['button_sequence'],
            'typed_text': session['typed_text'],
            'word_count': session['word_count'],
            'elapsed_time': 0.0,
            'wpm': 0.0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_state', methods=['GET'])
def get_state():
    """Get current application state"""
    try:
        # Calculate performance metrics
        elapsed_time = time.time() - session.get('start_time', time.time())
        wpm = (session.get('word_count', 0) / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'current_word': session.get('current_word', ''),
            'alternative_words': session.get('predicted_words', []),
            'next_word_predictions': session.get('next_word_predictions', []),
            'button_sequence': session.get('button_sequence', []),
            'typed_text': session.get('typed_text', ''),
            'word_count': session.get('word_count', 0),
            'elapsed_time': round(elapsed_time, 1),
            'wpm': round(wpm, 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)