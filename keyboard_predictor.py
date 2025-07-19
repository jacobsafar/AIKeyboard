import os
import json
from openai import OpenAI

class KeyboardPredictor:
    def __init__(self):
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Alphabet groups mapping
        self.groups = {
            1: "ABCDEFG",
            2: "HIJKLM", 
            3: "NOPQRS",
            4: "TUVWXYZ"
        }
    
    def predict_word(self, button_sequence, context_text=""):
        """
        Predict word based on button sequence and context using OpenAI API
        """
        if not button_sequence:
            return {"top_predictions": [], "alternative_words": []}
        
        # Convert button sequence to possible letter combinations
        sequence_str = " → ".join([str(x) for x in button_sequence])
        
        # Create context-aware prompt for OpenAI
        context_part = ""
        if context_text.strip():
            context_part = f"""
        CONTEXT: The user has already typed: "{context_text.strip()}"
        Consider this context when predicting the next word to ensure grammatical correctness and natural flow.
        """
        
        prompt = f"""
        I have a 4-button keyboard where each button represents a group of letters:
        - Button 1: A, B, C, D, E, F, G
        - Button 2: H, I, J, K, L, M
        - Button 3: N, O, P, Q, R, S
        - Button 4: T, U, V, W, X, Y, Z
        
        The user pressed buttons in this sequence: {sequence_str}
        {context_part}
        
        Based on this sequence and context, predict the 3 most likely English words the user is trying to type.
        List them in order of probability (most likely first).
        Also provide up to 5 additional alternative words that could match this sequence.
        
        Consider:
        - Common English words
        - Letter frequency and common patterns
        - Context and grammatical flow (if context provided)
        - Word usage frequency in English
        - Sentence structure and natural language patterns
        
        Respond with JSON in this format:
        {{
            "top_predictions": ["word1", "word2", "word3"],
            "alternative_words": ["word4", "word5", "word6", "word7", "word8"],
            "confidence": 0.85
        }}
        
        Only include real English words. Make sure all words actually match the button sequence.
        Prioritize words that fit naturally with the given context.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at predicting words from keyboard input sequences. You understand letter frequency, common word patterns, and English language structure."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate the result
            top_predictions = [word.upper() for word in result.get("top_predictions", [])]
            alternative_words = [word.upper() for word in result.get("alternative_words", [])]
            
            # Verify words match the sequence
            validated_top = [
                word for word in top_predictions 
                if self._validate_word_sequence(word, button_sequence)
            ]
            validated_alternatives = [
                word for word in alternative_words 
                if self._validate_word_sequence(word, button_sequence)
            ]
            
            # Ensure we have at least one prediction
            if validated_top:
                return {
                    "top_predictions": validated_top[:3],  # Limit to 3
                    "alternative_words": validated_alternatives[:5],  # Limit to 5
                    "confidence": result.get("confidence", 0.0)
                }
            else:
                # If validation fails, try a simpler approach
                return self._fallback_prediction(button_sequence)
                
        except Exception as e:
            print(f"API Error: {str(e)}")
            return self._fallback_prediction(button_sequence)
    
    def predict_next_words(self, current_text, current_word=""):
        """
        Predict the next word(s) based on context
        """
        if not current_text.strip() and not current_word:
            return []
        
        # Use current_text as context, add current_word if provided
        if current_word:
            context = current_text + " " + current_word if current_text else current_word
        else:
            context = current_text.strip()
        
        prompt = f"""
        Given this text: "{context}"
        
        Predict the 3 most likely next words that would follow naturally in English.
        Consider:
        - Grammar and sentence structure
        - Common word combinations and phrases
        - Context and meaning
        - Natural language flow and sentence completion
        - Proper punctuation needs (if sentence is complete)
        
        Respond with JSON in this format:
        {{
            "next_words": ["word1", "word2", "word3"]
        }}
        
        Only suggest common English words that would make grammatical sense and help complete thoughts naturally.
        If the sentence seems complete, suggest words that would start a new sentence or thought.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at predicting the next word in English text based on context and grammar."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("next_words", [])
            
        except Exception as e:
            print(f"Next word prediction error: {str(e)}")
            return []
    
    def _validate_word_sequence(self, word, button_sequence):
        """
        Validate that a word matches the button sequence
        """
        if len(word) != len(button_sequence):
            return False
        
        for i, letter in enumerate(word):
            button = button_sequence[i]
            if letter not in self.groups.get(button, ""):
                return False
        
        return True
    
    def _fallback_prediction(self, button_sequence):
        """
        Simple fallback prediction when API fails
        """
        # Common word patterns for each sequence length
        common_words = {
            1: {"A": 1, "I": 2, "H": 2, "N": 3, "T": 4},
            2: {"AM": [1, 2], "AN": [1, 3], "AS": [1, 3], "AT": [1, 4], "BE": [1, 1], "DO": [1, 3], "GO": [1, 3], "HE": [2, 1], "IF": [2, 1], "IN": [2, 3], "IS": [2, 3], "IT": [2, 4], "ME": [2, 1], "MY": [2, 4], "NO": [3, 3], "OF": [3, 1], "ON": [3, 3], "OR": [3, 3], "SO": [3, 3], "TO": [4, 3], "UP": [4, 3], "WE": [4, 1]},
            3: {"AND": [1, 3, 1], "ARE": [1, 3, 1], "BUT": [1, 4, 4], "CAN": [1, 1, 3], "FOR": [1, 3, 3], "HAD": [2, 1, 1], "HAS": [2, 1, 3], "HER": [2, 1, 3], "HIM": [2, 2, 2], "HIS": [2, 2, 3], "HOW": [2, 3, 4], "NOT": [3, 3, 4], "NOW": [3, 3, 4], "OLD": [3, 2, 1], "ONE": [3, 3, 1], "OUR": [3, 4, 3], "OUT": [3, 4, 4], "SHE": [3, 2, 1], "THE": [4, 2, 1], "TWO": [4, 4, 3], "USE": [4, 3, 1], "WAS": [4, 1, 3], "WAY": [4, 1, 4], "WHO": [4, 2, 3], "YOU": [4, 3, 4]},
            4: {"EACH": [1, 1, 1, 2], "FROM": [1, 3, 3, 2], "HAVE": [2, 1, 4, 1], "HEAR": [2, 1, 1, 3], "HELP": [2, 1, 2, 3], "HERE": [2, 1, 3, 1], "JUST": [2, 4, 3, 4], "KNOW": [2, 3, 3, 4], "LIKE": [2, 2, 2, 1], "LOOK": [2, 3, 3, 2], "MAKE": [2, 1, 2, 1], "MANY": [2, 1, 3, 4], "MORE": [2, 3, 3, 1], "MOST": [2, 3, 3, 4], "MUCH": [2, 4, 1, 2], "NEED": [3, 1, 1, 1], "ONLY": [3, 3, 2, 4], "OVER": [3, 4, 1, 3], "SAID": [3, 1, 2, 1], "SUCH": [3, 4, 1, 2], "TAKE": [4, 1, 2, 1], "THAN": [4, 2, 1, 3], "THAT": [4, 2, 1, 4], "THEM": [4, 2, 1, 2], "THEY": [4, 2, 1, 4], "THIS": [4, 2, 2, 3], "TIME": [4, 2, 2, 1], "VERY": [4, 1, 3, 4], "WANT": [4, 1, 3, 4], "WELL": [4, 1, 2, 2], "WERE": [4, 1, 3, 1], "WHAT": [4, 2, 1, 4], "WHEN": [4, 2, 1, 3], "WILL": [4, 2, 2, 2], "WITH": [4, 2, 4, 2], "WORD": [4, 3, 3, 1], "WORK": [4, 3, 3, 2], "YEAR": [4, 1, 1, 3]},
            5: {"ABOUT": [1, 1, 3, 4, 4], "AFTER": [1, 1, 4, 1, 3], "AGAIN": [1, 1, 1, 2, 3], "COULD": [1, 3, 4, 2, 1], "EVERY": [1, 4, 1, 3, 4], "FIRST": [1, 2, 3, 3, 4], "FOUND": [1, 3, 4, 3, 1], "GREAT": [1, 3, 1, 1, 4], "GROUP": [1, 3, 3, 4, 3], "HELLO": [2, 1, 2, 2, 3], "HOUSE": [2, 3, 4, 3, 1], "LARGE": [2, 1, 3, 1, 1], "LEARN": [2, 1, 1, 3, 3], "MIGHT": [2, 2, 1, 2, 4], "NEVER": [3, 1, 4, 1, 3], "OTHER": [3, 4, 2, 1, 3], "PLACE": [3, 2, 1, 1, 1], "RIGHT": [3, 2, 1, 2, 4], "SHALL": [3, 2, 1, 2, 2], "SMALL": [3, 2, 1, 2, 2], "STATE": [3, 4, 1, 4, 1], "THEIR": [4, 2, 1, 2, 3], "THERE": [4, 2, 1, 3, 1], "THESE": [4, 2, 1, 3, 1], "THINK": [4, 2, 2, 3, 2], "THREE": [4, 2, 3, 1, 1], "UNDER": [4, 3, 1, 1, 3], "WATER": [4, 1, 4, 1, 3], "WHERE": [4, 2, 1, 3, 1], "WHICH": [4, 2, 2, 1, 2], "WHILE": [4, 2, 2, 2, 1], "WORLD": [4, 3, 3, 2, 1], "WOULD": [4, 3, 4, 2, 1], "WRITE": [4, 3, 2, 4, 1], "YOUNG": [4, 3, 4, 3, 1]}
        }
        
        # Find matching words for this sequence
        sequence_length = len(button_sequence)
        if sequence_length in common_words:
            matches = []
            for word, word_sequence in common_words[sequence_length].items():
                if isinstance(word_sequence, list) and word_sequence == button_sequence:
                    matches.append(word)
                elif isinstance(word_sequence, int) and len(button_sequence) == 1 and word_sequence == button_sequence[0]:
                    matches.append(word)
            
            if matches:
                return {
                    "top_predictions": matches[:3],  # First 3 as top predictions
                    "alternative_words": matches[3:8],  # Next 5 as alternatives
                    "confidence": 0.6
                }
        
        # If no matches found, return empty
        return {"top_predictions": [], "alternative_words": []}
