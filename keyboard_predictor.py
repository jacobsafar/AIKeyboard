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
        
        # Frequency-based alphabet groups mapping
        # Button 1: Most frequent letters (62% corpus coverage)
        # Button 2: Second most frequent (24% corpus coverage) 
        # Button 3: Third most frequent (11% corpus coverage)
        # Button 4: Least frequent letters (3% corpus coverage)
        self.groups = {
            1: "ETAOINH",
            2: "SRDLCUG", 
            3: "MPFYWB",
            4: "VKXQJZ"
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
        - Button 1: E, T, A, O, I, N, H (most frequent letters - 62% coverage)
        - Button 2: S, R, D, L, C, U, G (second most frequent - 24% coverage)  
        - Button 3: M, P, F, Y, W, B (third most frequent - 11% coverage)
        - Button 4: V, K, X, Q, J, Z (least frequent - 3% coverage)
        
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
            
        # Build a comprehensive word list with correct button sequences
        words_with_sequences = [
            # Length 1
            ("A", [1]), ("I", [1]), ("E", [1]), ("T", [1]), ("O", [1]), ("N", [1]), ("H", [1]),
            ("S", [2]), ("R", [2]), ("D", [2]), ("L", [2]), ("C", [2]), ("U", [2]), ("G", [2]),
            ("M", [3]), ("P", [3]), ("F", [3]), ("Y", [3]), ("W", [3]), ("B", [3]),
            ("V", [4]), ("K", [4]), ("X", [4]), ("Q", [4]), ("J", [4]), ("Z", [4]),
            
            # Length 2
            ("TO", [1, 1]), ("IT", [1, 1]), ("IN", [1, 1]), ("AT", [1, 1]), ("HE", [1, 1]), ("AN", [1, 1]), ("ON", [1, 1]),
            ("IS", [1, 2]), ("AS", [1, 2]), ("OR", [1, 2]), ("HI", [1, 1]), ("NO", [1, 1]), ("OH", [1, 1]),
            ("GO", [2, 1]), ("SO", [2, 1]), ("DO", [2, 1]), ("UP", [2, 3]), ("MY", [3, 3]), ("BY", [3, 3]),
            ("WE", [3, 1]), ("ME", [3, 1]), ("BE", [3, 1]),
            
            # Length 3  
            ("THE", [1, 1, 1]), ("AND", [1, 1, 2]), ("YOU", [3, 1, 2]), ("NOT", [1, 1, 1]), ("CAN", [2, 1, 1]),
            ("HAD", [1, 1, 2]), ("HER", [1, 1, 2]), ("HAS", [1, 1, 2]), ("HIS", [1, 1, 2]), ("ONE", [1, 1, 1]),
            ("OUT", [1, 2, 1]), ("SHE", [2, 1, 1]), ("HOW", [1, 1, 3]), ("ARE", [1, 2, 1]), ("GET", [2, 1, 1]),
            ("ALL", [1, 2, 2]), ("NEW", [1, 1, 3]), ("SEE", [2, 1, 1]), ("TWO", [1, 3, 1]), ("WHO", [3, 1, 1]),
            ("OIL", [1, 1, 2]), ("USE", [2, 2, 1]), ("MAN", [3, 1, 1]), ("DAY", [2, 1, 3]), ("WAY", [3, 1, 3]),
            ("MAY", [3, 1, 3]), ("OLD", [1, 2, 2]), ("GOT", [2, 1, 1]), ("BAD", [3, 1, 2]), ("BIG", [3, 1, 2]),
            ("BOY", [3, 1, 3]), ("PUT", [3, 2, 1]), ("END", [1, 1, 2]), ("TRY", [1, 2, 3]), ("LET", [2, 1, 1]),
            
            # Length 4
            ("THAT", [1, 1, 1, 1]), ("THIS", [1, 1, 1, 2]), ("HAVE", [1, 1, 4, 1]), ("THEY", [1, 1, 1, 3]),
            ("THEN", [1, 1, 1, 1]), ("THEM", [1, 1, 1, 3]), ("THAN", [1, 1, 1, 1]), ("HEAR", [1, 1, 1, 2]),
            ("HERE", [1, 1, 2, 1]), ("HELP", [1, 1, 2, 3]), ("HATE", [1, 1, 1, 1]), ("HINT", [1, 1, 1, 1]),
            ("TONE", [1, 1, 1, 1]), ("NONE", [1, 1, 1, 1]), ("NOTE", [1, 1, 1, 1]), ("NEED", [1, 1, 1, 2]),
            ("NEAT", [1, 1, 1, 1]), ("NEAR", [1, 1, 1, 2]), ("NAME", [1, 1, 3, 1]), ("GAME", [2, 1, 3, 1]),
            ("CAME", [2, 1, 3, 1]), ("SAME", [2, 1, 3, 1]), ("TIME", [1, 1, 3, 1]), ("MADE", [3, 1, 2, 1]),
            ("TAKE", [1, 1, 4, 1]), ("MAKE", [3, 1, 4, 1]), ("LIFE", [2, 1, 3, 1]), ("HOME", [1, 1, 3, 1]),
            ("COME", [2, 1, 3, 1]), ("SOME", [2, 1, 3, 1]), ("GOOD", [2, 1, 1, 2]), ("COOL", [2, 1, 1, 2]),
            ("POOL", [3, 1, 1, 2]),
            
            # Length 5
            ("HELLO", [1, 1, 2, 2, 1]), ("THERE", [1, 1, 1, 2, 1]), ("THESE", [1, 1, 1, 2, 1]),
            ("THREE", [1, 1, 2, 1, 1]), ("THOSE", [1, 1, 1, 2, 1]), ("THANK", [1, 1, 1, 1, 4]),
            ("NIGHT", [1, 1, 2, 1, 1]), ("OTHER", [1, 1, 1, 1, 2]), ("HANDS", [1, 1, 1, 2, 2]),
            ("HOUSE", [1, 1, 2, 2, 1]), ("EARTH", [1, 1, 2, 1, 1]), ("HEART", [1, 1, 1, 2, 1]),
            ("ENTER", [1, 1, 1, 1, 2]), ("EATEN", [1, 1, 1, 1, 1]), ("TEETH", [1, 1, 1, 1, 1]),
            ("WHERE", [3, 1, 1, 2, 1]), ("WATER", [3, 1, 1, 1, 2]), ("WATCH", [3, 1, 1, 2, 1]),
            ("SWEET", [2, 3, 1, 1, 1]), ("GREAT", [2, 2, 1, 1, 1]), ("GREEN", [2, 2, 1, 1, 1]),
            ("WOULD", [3, 1, 2, 2, 2]), ("WORLD", [3, 1, 2, 2, 2]), ("WRITE", [3, 2, 1, 1, 1]),
            ("WHITE", [3, 1, 1, 1, 1]), ("WHILE", [3, 1, 1, 2, 1]), ("WOMAN", [3, 1, 3, 1, 1]),
            ("MONEY", [3, 1, 1, 1, 3]), ("MONTH", [3, 1, 1, 1, 1]), ("MIGHT", [3, 1, 2, 1, 1]),
            ("MUSIC", [3, 2, 2, 1, 2]), ("PLACE", [3, 2, 1, 2, 1]), ("POWER", [3, 1, 3, 1, 2]),
            ("POINT", [3, 1, 1, 1, 1]), ("PHONE", [3, 1, 1, 1, 1]), ("YOUNG", [3, 1, 2, 1, 2]),
            ("STORY", [2, 1, 1, 2, 3]), ("START", [2, 1, 1, 2, 1]), ("STUDY", [2, 1, 2, 2, 3]),
            ("SMALL", [2, 3, 1, 2, 2]), ("SPEAK", [2, 3, 1, 1, 4]), ("SOUND", [2, 1, 2, 1, 2]),
            ("COULD", [2, 1, 2, 2, 2]), ("CLEAN", [2, 2, 1, 1, 1]), ("CLEAR", [2, 2, 1, 1, 2]),
            ("CLASS", [2, 2, 1, 2, 2]), ("CLOSE", [2, 2, 1, 2, 1]), ("COLOR", [2, 1, 2, 1, 2]),
            ("COOLER", [2, 1, 1, 2, 1, 2]), ("COOLEST", [2, 1, 1, 2, 1, 2, 1]), ("COOLLY", [2, 1, 1, 2, 2, 3]),
            
            # Length 6
            ("PEOPLE", [3, 1, 1, 3, 2, 1]), ("PERSON", [3, 1, 2, 2, 1, 1]), ("PRETTY", [3, 2, 1, 1, 1, 3]),
            ("PLEASE", [3, 2, 1, 1, 2, 1]), ("FRIEND", [3, 2, 1, 1, 1, 2]), ("FATHER", [3, 1, 1, 1, 1, 2]),
            ("MOTHER", [3, 1, 1, 1, 1, 2]), ("BROTHER", [3, 2, 1, 1, 1, 1, 2]), ("SISTER", [2, 1, 2, 1, 1, 2]),
            ("SECOND", [2, 1, 2, 1, 1, 2]), ("SCHOOL", [2, 2, 1, 1, 1, 2]), ("SUMMER", [2, 2, 3, 3, 1, 2]),
            ("SIMPLE", [2, 1, 3, 3, 2, 1]), ("STREET", [2, 1, 2, 1, 1, 1]), ("STRONG", [2, 1, 2, 1, 1, 2]),
            ("CHANGE", [2, 1, 1, 1, 2, 1]), ("CHANCE", [2, 1, 1, 1, 2, 1]), ("CHOOSE", [2, 1, 1, 1, 2, 1]),
            ("CORNER", [2, 1, 2, 1, 1, 2]), ("COUPLE", [2, 1, 2, 3, 2, 1]), ("COURSE", [2, 1, 2, 2, 2, 1]),
            ("LOVELY", [2, 1, 4, 1, 2, 3]), ("LETTER", [2, 1, 1, 1, 1, 2]), ("LISTEN", [2, 1, 2, 1, 1, 1]),
            ("LITTLE", [2, 1, 1, 1, 2, 1]), ("MOMENT", [3, 1, 3, 1, 1, 1]), ("MEMORY", [3, 1, 3, 1, 2, 3]),
            ("MASTER", [3, 1, 2, 1, 1, 2]), ("MATTER", [3, 1, 1, 1, 1, 2]), ("MARKET", [3, 1, 2, 4, 1, 1]),
            ("ANIMAL", [1, 1, 1, 3, 1, 2]), ("AROUND", [1, 2, 1, 2, 1, 2]), ("ALWAYS", [1, 2, 3, 1, 3, 2]),
            ("HAPPEN", [1, 1, 3, 3, 1, 1]), ("NATURE", [1, 1, 1, 2, 2, 1]), ("TRAVEL", [1, 2, 1, 4, 1, 2])
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
