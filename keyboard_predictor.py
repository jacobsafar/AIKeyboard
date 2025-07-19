import os
import json
from openai import OpenAI
from names_database import get_names_for_sequence, is_name

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
        
        # Frequency-based alphabet groups mapping for 6-button layout
        # Button 1: E L
        # Button 2: T R C Q
        # Button 3: A D F V
        # Button 4: O H W Z
        # Button 5: I S K G
        # Button 6: N U M P Y B J X
        self.groups = {
            1: "EL",
            2: "TRCQ", 
            3: "ADFV",
            4: "OHWZ",
            5: "ISKG",
            6: "NUMPYBJX"
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
        I have a 6-button keyboard where each button represents a group of letters:
        - Button 1: E, L
        - Button 2: T, R, C, Q
        - Button 3: A, D, F, V
        - Button 4: O, H, W, Z
        - Button 5: I, S, K, G
        - Button 6: N, U, M, P, Y, B, J, X
        
        The user pressed buttons in this sequence: {sequence_str}
        {context_part}
        
        Find ALL possible words that match this exact button sequence. Each letter in the word MUST come from the corresponding button group.
        
        For sequence {sequence_str}, valid words must have:
        {self._explain_sequence_requirements(button_sequence)}
        
        List the 3 most likely words first, then up to 5 additional alternatives.
        Include:
        - Common English words
        - Proper names (Jacob, Maria, David, Smith, Johnson, etc.)
        - Place names when relevant
        - All grammatically correct words that fit the pattern
        
        IMPORTANT: Pay special attention to context clues for names:
        - If context mentions "my name is", "I am", "called", strongly prioritize proper names
        - If context suggests introducing someone, favor first names
        - If context mentions family/surname, favor last names
        
        Respond with JSON in this format:
        {{
            "top_predictions": ["word1", "word2", "word3"],
            "alternative_words": ["word4", "word5", "word6", "word7", "word8"],
            "confidence": 0.85
        }}
        
        CRITICAL: Only suggest words where every letter matches the button sequence exactly.
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
            print(f"OpenAI response for sequence {button_sequence}: {result}")
            
            # Validate the result
            top_predictions = [word.upper() for word in result.get("top_predictions", [])]
            alternative_words = [word.upper() for word in result.get("alternative_words", [])]
            print(f"Top predictions: {top_predictions}")
            print(f"Alternative words: {alternative_words}")
            
            # Verify words match the sequence
            validated_top = []
            validated_alternatives = []
            
            for word in top_predictions:
                if self._validate_word_sequence(word, button_sequence):
                    validated_top.append(word)
                else:
                    print(f"Validation failed for top prediction '{word}' with sequence {button_sequence}")
            
            for word in alternative_words:
                if self._validate_word_sequence(word, button_sequence):
                    validated_alternatives.append(word)
                else:
                    print(f"Validation failed for alternative '{word}' with sequence {button_sequence}")
            
            # Get name candidates that match the sequence
            name_candidates = get_names_for_sequence(button_sequence, self.groups)
            print(f"Name candidates for sequence {button_sequence}: {name_candidates[:5]}")
            
            # Combine validated predictions with name candidates
            all_candidates = validated_top + validated_alternatives + name_candidates
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for word in all_candidates:
                if word not in seen:
                    unique_candidates.append(word)
                    seen.add(word)
            
            # Always return OpenAI results (context-aware) combined with name matches
            # This ensures context like "my name is..." can prioritize names appropriately
            final_top = unique_candidates[:3] if unique_candidates else top_predictions[:3]
            final_alternatives = unique_candidates[3:8] if len(unique_candidates) > 3 else alternative_words[:5]
            
            return {
                "top_predictions": final_top,
                "alternative_words": final_alternatives,
                "confidence": result.get("confidence", 0.0)
            }
                
        except Exception as e:
            print(f"API Error for sequence {button_sequence}: {str(e)}")
            print(f"Context text: '{context_text}'")
            import traceback
            traceback.print_exc()
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
    
    def _explain_sequence_requirements(self, button_sequence):
        """
        Explain what letters are allowed for each position in the sequence
        """
        explanation = ""
        for i, button in enumerate(button_sequence):
            letters = ", ".join(self.groups[button])
            explanation += f"- Position {i+1}: Must be one of {letters}\n"
        return explanation.strip()
    
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
    
    def _get_fallback_table(self):
        """
        Generate fallback table with correct frequency-based letter mapping
        """
        if hasattr(self, '_fallback_table'):
            return self._fallback_table
            
        # Build a comprehensive word list with correct 6-button sequences
        # Button 1: E L, Button 2: T R C Q, Button 3: A D F V, Button 4: O H W Z, Button 5: I S K G, Button 6: N U M P Y B J X
        words_with_sequences = [
            # Length 1
            ("E", [1]), ("L", [1]),
            ("T", [2]), ("R", [2]), ("C", [2]), ("Q", [2]),
            ("A", [3]), ("D", [3]), ("F", [3]), ("V", [3]),
            ("O", [4]), ("H", [4]), ("W", [4]), ("Z", [4]),
            ("I", [5]), ("S", [5]), ("K", [5]), ("G", [5]),
            ("N", [6]), ("U", [6]), ("M", [6]), ("P", [6]), ("Y", [6]), ("B", [6]), ("J", [6]), ("X", [6]),
            
            # Length 2
            ("EL", [1, 1]), ("LE", [1, 1]),
            ("TO", [2, 4]), ("IT", [5, 2]), ("IN", [5, 6]), ("AT", [3, 2]), ("HE", [4, 1]), ("AN", [3, 6]), ("ON", [4, 6]),
            ("IS", [5, 5]), ("AS", [3, 5]), ("OR", [4, 2]), ("HI", [4, 5]), ("NO", [6, 4]), ("OH", [4, 4]),
            ("GO", [5, 4]), ("SO", [5, 4]), ("DO", [3, 4]), ("UP", [6, 6]), ("MY", [6, 6]), ("BY", [6, 6]),
            ("WE", [4, 1]), ("ME", [6, 1]), ("BE", [6, 1]),
            
            # Length 3  
            ("THE", [2, 4, 1]), ("AND", [3, 6, 3]), ("YOU", [6, 4, 6]), ("NOT", [6, 4, 2]), ("CAN", [2, 3, 6]),
            ("HAD", [4, 3, 3]), ("HER", [4, 1, 2]), ("HAS", [4, 3, 5]), ("HIS", [4, 5, 5]), ("ONE", [4, 6, 1]),
            ("OUT", [4, 6, 2]), ("SHE", [5, 4, 1]), ("HOW", [4, 4, 4]), ("ARE", [3, 2, 1]), ("GET", [5, 1, 2]),
            ("ALL", [3, 1, 1]), ("NEW", [6, 1, 4]), ("SEE", [5, 1, 1]), ("TWO", [2, 4, 4]), ("WHO", [4, 4, 4]),
            ("OIL", [4, 5, 1]), ("USE", [6, 5, 1]), ("MAN", [6, 3, 6]), ("DAY", [3, 3, 6]), ("WAY", [4, 3, 6]),
            ("MAY", [6, 3, 6]), ("OLD", [4, 1, 3]), ("GOT", [5, 4, 2]), ("BAD", [6, 3, 3]), ("BIG", [6, 5, 5]),
            ("BOY", [6, 4, 6]), ("PUT", [6, 6, 2]), ("END", [1, 6, 3]), ("TRY", [2, 2, 6]), ("LET", [1, 1, 2]),
            
            # Length 4
            ("THAT", [2, 4, 3, 2]), ("THIS", [2, 4, 5, 5]), ("HAVE", [4, 3, 3, 1]), ("THEY", [2, 4, 1, 6]),
            ("THEN", [2, 4, 1, 6]), ("THEM", [2, 4, 1, 6]), ("THAN", [2, 4, 3, 6]), ("HEAR", [4, 1, 3, 2]),
            ("HERE", [4, 1, 2, 1]), ("HELP", [4, 1, 1, 6]), ("HATE", [4, 3, 2, 1]), ("HINT", [4, 5, 6, 2]),
            ("TONE", [2, 4, 6, 1]), ("NONE", [6, 4, 6, 1]), ("NOTE", [6, 4, 2, 1]), ("NEED", [6, 1, 1, 3]),
            ("NEAT", [6, 1, 3, 2]), ("NEAR", [6, 1, 3, 2]), ("NAME", [6, 3, 6, 1]), ("GAME", [5, 3, 6, 1]),
            ("CAME", [2, 3, 6, 1]), ("SAME", [5, 3, 6, 1]), ("TIME", [2, 5, 6, 1]), ("MADE", [6, 3, 3, 1]),
            ("TAKE", [2, 3, 5, 1]), ("MAKE", [6, 3, 5, 1]), ("LIFE", [1, 5, 3, 1]), ("HOME", [4, 4, 6, 1]),
            ("COME", [2, 4, 6, 1]), ("SOME", [5, 4, 6, 1]), ("GOOD", [5, 4, 4, 3]), ("COOL", [2, 4, 4, 1]),
            ("POOL", [6, 4, 4, 1]),
            
            # Length 5
            ("HELLO", [4, 1, 1, 1, 4]), ("THERE", [2, 4, 1, 2, 1]), ("THESE", [2, 4, 1, 5, 1]),
            ("THREE", [2, 4, 2, 1, 1]), ("THOSE", [2, 4, 4, 5, 1]), ("THANK", [2, 4, 3, 6, 5]),
            ("NIGHT", [6, 5, 5, 4, 2]), ("OTHER", [4, 2, 4, 1, 2]), ("HANDS", [4, 3, 6, 3, 5]),
            ("HOUSE", [4, 4, 6, 5, 1]), ("EARTH", [1, 3, 2, 2, 4]), ("HEART", [4, 1, 3, 2, 2]),
            ("ENTER", [1, 6, 2, 1, 2]), ("EATEN", [1, 3, 2, 1, 6]), ("TEETH", [2, 1, 1, 2, 4]),
            ("WHERE", [4, 4, 1, 2, 1]), ("WATER", [4, 3, 2, 1, 2]), ("WATCH", [4, 3, 2, 2, 4]),
            ("SWEET", [5, 4, 1, 1, 2]), ("GREAT", [5, 2, 1, 3, 2]), ("GREEN", [5, 2, 1, 1, 6]),
            ("WOULD", [4, 4, 6, 1, 3]), ("WORLD", [4, 4, 2, 1, 3]), ("WRITE", [4, 2, 5, 2, 1]),
            ("WHITE", [4, 4, 5, 2, 1]), ("WHILE", [4, 4, 5, 1, 1]), ("WOMAN", [4, 4, 6, 3, 6]),
            ("MONEY", [6, 4, 6, 1, 6]), ("MONTH", [6, 4, 6, 2, 4]), ("MIGHT", [6, 5, 5, 4, 2]),
            ("MUSIC", [6, 6, 5, 5, 2]), ("PLACE", [6, 1, 3, 2, 1]), ("POWER", [6, 4, 4, 1, 2]),
            ("POINT", [6, 4, 5, 6, 2]), ("PHONE", [6, 4, 4, 6, 1]), ("YOUNG", [6, 4, 6, 6, 5]),
            ("STORY", [5, 2, 4, 2, 6]), ("START", [5, 2, 3, 2, 2]), ("STUDY", [5, 2, 6, 3, 6]),
            ("SMALL", [5, 6, 3, 1, 1]), ("SPEAK", [5, 6, 1, 3, 5]), ("SOUND", [5, 4, 6, 6, 3]),
            ("COULD", [2, 4, 6, 1, 3]), ("CLEAN", [2, 1, 1, 3, 6]), ("CLEAR", [2, 1, 1, 3, 2]),
            ("CLASS", [2, 1, 3, 5, 5]), ("CLOSE", [2, 1, 4, 5, 1]), ("COLOR", [2, 4, 1, 4, 2]),
            
            # Length 6
            ("PEOPLE", [6, 1, 4, 6, 1, 1]), ("PERSON", [6, 1, 2, 5, 4, 6]), ("PRETTY", [6, 2, 1, 2, 2, 6]),
            ("PLEASE", [6, 1, 1, 3, 5, 1]), ("FRIEND", [3, 2, 5, 1, 6, 3]), ("FATHER", [3, 3, 2, 4, 1, 2]),
            ("MOTHER", [6, 4, 2, 4, 1, 2]), ("SISTER", [5, 5, 5, 2, 1, 2]),
            ("SECOND", [5, 1, 2, 4, 6, 3]), ("SCHOOL", [5, 2, 4, 4, 4, 1]), ("SUMMER", [5, 6, 6, 6, 1, 2]),
            ("SIMPLE", [5, 5, 6, 6, 1, 1]), ("STREET", [5, 2, 2, 1, 1, 2]), ("STRONG", [5, 2, 2, 4, 6, 5]),
            ("CHANGE", [2, 4, 3, 6, 5, 1]), ("CHANCE", [2, 4, 3, 6, 2, 1]), ("CHOOSE", [2, 4, 4, 4, 5, 1]),
            ("CORNER", [2, 4, 2, 6, 1, 2]), ("COUPLE", [2, 4, 6, 6, 1, 1]), ("COURSE", [2, 4, 6, 2, 5, 1]),
            ("LOVELY", [1, 4, 3, 1, 1, 6]), ("LETTER", [1, 1, 2, 2, 1, 2]), ("LISTEN", [1, 5, 5, 2, 1, 6]),
            ("LITTLE", [1, 5, 2, 2, 1, 1]), ("MOMENT", [6, 4, 6, 1, 6, 2]), ("MEMORY", [6, 1, 6, 4, 2, 6]),
            ("MASTER", [6, 3, 5, 2, 1, 2]), ("MATTER", [6, 3, 2, 2, 1, 2]), ("MARKET", [6, 3, 2, 5, 1, 2]),
            ("ANIMAL", [3, 6, 5, 6, 3, 1]), ("AROUND", [3, 2, 4, 6, 6, 3]), ("ALWAYS", [3, 1, 4, 3, 6, 5]),
            ("HAPPEN", [4, 3, 6, 6, 1, 6]), ("NATURE", [6, 3, 2, 6, 2, 1]), ("TRAVEL", [2, 2, 3, 3, 1, 1])
        ]
        
        # Organize by length
        self._fallback_table = {}
        for word, sequence in words_with_sequences:
            length = len(sequence)
            if length not in self._fallback_table:
                self._fallback_table[length] = []
            self._fallback_table[length].append((word, sequence))
        
        return self._fallback_table
    
    def _fallback_prediction(self, button_sequence):
        """
        Fallback prediction using the rebuilt frequency-based word table
        """
        fallback_table = self._get_fallback_table()
        sequence_length = len(button_sequence)
        
        if sequence_length not in fallback_table:
            return {"top_predictions": [], "alternative_words": []}
        
        # Find words that match the button sequence
        matches = []
        for word, word_sequence in fallback_table[sequence_length]:
            if word_sequence == button_sequence:
                matches.append(word)
        
        if matches:
            return {
                "top_predictions": matches[:3],  # First 3 as top predictions
                "alternative_words": matches[3:8],  # Next 5 as alternatives  
                "confidence": 0.7
            }
        
        return {"top_predictions": [], "alternative_words": []}
