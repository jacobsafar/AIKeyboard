# AI 4-Button Keyboard

## Overview

This is a Flask-based web application that implements an AI-powered 4-button keyboard system. The application allows users to type words using only 4 buttons, where each button represents a group of letters from the alphabet. An AI model (GPT-4o) predicts the intended words based on the button sequence pressed by the user.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a simple two-tier architecture:

1. **Frontend**: Flask web application with HTML/CSS/JavaScript interface that provides the user interaction layer
2. **Backend Logic**: Python classes that handle keyboard prediction using OpenAI's API

The architecture is designed for simplicity and real-time interaction, with state management handled through Flask's session mechanism.

## Key Components

### Frontend (app.py)
- **Flask Web Application**: Provides RESTful API endpoints and serves the HTML interface
- **Session Management**: Maintains user typing state, button sequences, predictions, and typing statistics using Flask sessions
- **Real-time AJAX Communication**: JavaScript handles button presses and updates the UI dynamically
- **Responsive HTML Interface**: Modern CSS styling with keyboard shortcuts support

### Backend (keyboard_predictor.py)
- **KeyboardPredictor Class**: Core logic for converting button sequences to word predictions
- **OpenAI Integration**: Uses GPT-4o model to predict words based on button patterns
- **Alphabet Grouping**: Maps 4 buttons to letter groups:
  - Button 1: A-G (ABCDEFG)
  - Button 2: H-M (HIJKLM)
  - Button 3: N-S (NOPQRS)
  - Button 4: T-Z (TUVWXYZ)

## Data Flow

1. **User Input**: User presses one of 4 buttons representing letter groups
2. **Sequence Building**: Button presses are accumulated into a sequence
3. **AI Prediction**: Button sequence is sent to OpenAI API with context about letter groupings
4. **Word Suggestions**: AI returns the most likely word plus alternatives
5. **Display Update**: UI updates to show predictions and allows word selection
6. **Text Building**: Selected words are added to the final typed text

## External Dependencies

### Core Libraries
- **Flask**: Web framework for the user interface and API endpoints
- **OpenAI**: API client for GPT-4o integration
- **Python Standard Library**: time, os, json modules

### API Services
- **OpenAI API**: Requires OPENAI_API_KEY environment variable
- **Model**: Uses GPT-4o (latest model as of May 2024)

### Environment Requirements
- Python environment with package installation capability
- Internet connection for OpenAI API calls
- Environment variable support for API key storage

## Deployment Strategy

The application is designed for simple deployment:

1. **Environment Setup**: Requires OPENAI_API_KEY to be set as environment variable
2. **Package Installation**: Standard pip install for Flask and OpenAI packages
3. **Single Command Launch**: Can be started with `python app.py`
4. **State Persistence**: Uses Flask's built-in session management (no external database required)

### Key Deployment Considerations
- API key security through environment variables
- No persistent data storage required
- RESTful API design allows for easy scaling and integration
- Web-based interface accessible through browser
- Keyboard shortcuts for improved user experience

## Recent Changes

### July 19, 2025
- **Framework Migration**: Converted from Streamlit to Flask based on user preference
- **Enhanced UI**: Implemented modern HTML/CSS interface with responsive design
- **AJAX Integration**: Added real-time communication between frontend and backend
- **Keyboard Shortcuts**: Added support for number keys (1-4), Enter, Backspace, and Escape
- **API Architecture**: Restructured as RESTful API with clear endpoint separation
- **Session Management**: Migrated from Streamlit session state to Flask sessions

The system prioritizes simplicity and real-time interaction over complex data persistence or multi-user management, making it suitable for single-user or demonstration environments.