from flask import Flask, render_template, request, jsonify, session
import os
import time
import threading
from collections import defaultdict
from keyboard_predictor import KeyboardPredictor

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize the keyboard predictor
predictor = KeyboardPredictor()

# Rate limiting for rapid requests
request_times = defaultdict(list)
rate_limit_lock = threading.Lock()

def is_rate_limited(session_id, max_requests=10, time_window=1.0):
    """Check if the session is rate limited"""
    with rate_limit_lock:
        now = time.time()
        # Clean old requests outside time window
        request_times[session_id] = [t for t in request_times[session_id] if now - t < time_window]
        
        # Check if exceeded limit
        if len(request_times[session_id]) >= max_requests:
            return True
        
        # Add current request
        request_times[session_id].append(now)
        return False

def init_session():
    """Initialize session variables if not present"""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    if 'button_sequence' not in session:
        session['button_sequence'] = []
    if 'top_predictions' not in session:
        session['top_predictions'] = []
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

@app.route('/')
def index():
    """Main page with the keyboard interface"""
    init_session()
    return render_template('index.html')

@app.route('/press_button', methods=['POST'])
def press_button():
    """Handle button press and return AI prediction"""
    try:
        init_session()
        
        # Rate limiting check
        session_id = session.get('session_id', id(session))
        if is_rate_limited(session_id):
            return jsonify({'error': 'Too many requests, please slow down'}), 429
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        button_num = data.get('button')
        
        if button_num not in [1, 2, 3, 4, 5, 6]:
            return jsonify({'error': 'Invalid button number'}), 400
        
        # Add button to sequence
        session['button_sequence'].append(button_num)
        
        # Get prediction from AI with context
        result = predictor.predict_word(session['button_sequence'], session['typed_text'])
        session['top_predictions'] = result.get('top_predictions', [])
        session['predicted_words'] = result.get('alternative_words', [])
        
        # Get next word predictions based on current typed text
        next_words = []
        if session['typed_text'].strip():  # Only predict next words if there's existing text
            next_words = predictor.predict_next_words(session['typed_text'], "")
        session['next_word_predictions'] = next_words
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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
        init_session()
        
        data = request.get_json()
        word = data.get('word')
        
        # If no word provided, use the first top prediction
        if not word and session['top_predictions']:
            word = session['top_predictions'][0]
        
        if word:
            # Add word to typed text
            if session['typed_text']:
                session['typed_text'] += " " + word
            else:
                session['typed_text'] = word
            
            # Reset for next word
            session['button_sequence'] = []
            session['top_predictions'] = []
            session['predicted_words'] = []
            session['word_count'] += 1
            
            # Generate next word predictions based on new text
            next_words = []
            if session['typed_text'].strip():
                next_words = predictor.predict_next_words(session['typed_text'], "")
            session['next_word_predictions'] = next_words
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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

@app.route('/backspace', methods=['POST'])
def backspace():
    """Remove last button press"""
    try:
        init_session()
        
        # Rate limiting check
        session_id = session.get('session_id', id(session))
        if is_rate_limited(session_id):
            return jsonify({'error': 'Too many requests, please slow down'}), 429
        
        if session['button_sequence']:
            session['button_sequence'].pop()
            
            if session['button_sequence']:
                # Re-predict with remaining sequence and context
                result = predictor.predict_word(session['button_sequence'], session['typed_text'])
                session['top_predictions'] = result.get('top_predictions', [])
                session['predicted_words'] = result.get('alternative_words', [])
            else:
                session['top_predictions'] = []
                session['predicted_words'] = []
                session['next_word_predictions'] = []
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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
        init_session()
        
        # Rate limiting check
        session_id = session.get('session_id', id(session))
        if is_rate_limited(session_id):
            return jsonify({'error': 'Too many requests, please slow down'}), 429
        
        session['button_sequence'] = []
        session['top_predictions'] = []
        session['predicted_words'] = []
        session['next_word_predictions'] = []
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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

@app.route('/add_space', methods=['POST'])
def add_space():
    """Add space - same functionality as accept word"""
    try:
        init_session()
        
        # If there's a current prediction, accept it
        if session['top_predictions']:
            word = session['top_predictions'][0]
            
            # Add word to typed text
            if session['typed_text']:
                session['typed_text'] += " " + word
            else:
                session['typed_text'] = word
            
            # Reset for next word
            session['button_sequence'] = []
            session['top_predictions'] = []
            session['predicted_words'] = []
            session['word_count'] += 1
        else:
            # If no prediction, just add space
            if session['typed_text']:
                session['typed_text'] += ' '
        
        # Generate next word predictions based on updated text
        next_words = []
        if session['typed_text'].strip():
            next_words = predictor.predict_next_words(session['typed_text'], "")
        session['next_word_predictions'] = next_words
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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

@app.route('/add_next_word', methods=['POST'])
def add_next_word():
    """Add a suggested next word to the typed text"""
    try:
        init_session()
        
        data = request.get_json()
        word = data.get('word')
        
        if word:
            # Add word to typed text
            if session['typed_text']:
                session['typed_text'] += " " + word
            else:
                session['typed_text'] = word
            
            # Clear current word predictions but keep next word suggestions
            session['button_sequence'] = []
            session['top_predictions'] = []
            session['predicted_words'] = []
            session['word_count'] += 1
            
            # Update next word predictions based on new text
            next_words = []
            if session['typed_text'].strip():
                next_words = predictor.predict_next_words(session['typed_text'], "")
            session['next_word_predictions'] = next_words
        
        # Calculate performance metrics
        elapsed_time = time.time() - session['start_time']
        wpm = (session['word_count'] / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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
        init_session()
        
        # Rate limiting check
        session_id = session.get('session_id', id(session))
        if is_rate_limited(session_id):
            return jsonify({'error': 'Too many requests, please slow down'}), 429
        
        # Initialize and clear all session variables
        session['button_sequence'] = []
        session['top_predictions'] = []
        session['predicted_words'] = []
        session['next_word_predictions'] = []
        session['typed_text'] = ""
        session['start_time'] = time.time()
        session['word_count'] = 0
        
        return jsonify({
            'top_predictions': session['top_predictions'],
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
        init_session()
        
        # Calculate performance metrics
        elapsed_time = time.time() - session.get('start_time', time.time())
        wpm = (session.get('word_count', 0) / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        return jsonify({
            'top_predictions': session.get('top_predictions', []),
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