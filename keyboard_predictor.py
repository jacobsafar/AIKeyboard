import os
import json
import re
from openai import OpenAI
from names_database import get_names_for_sequence

# Few-shot examples for the LLM
EXAMPLES = """
Example 1
  Previous text: ""
  Sequence: 2 4 1 2 1
  -> ["THERE", "THEIR", "THREE"]

Example 2
  Previous text: "Today"
  Sequence: 4 3 5
  -> ["WAS", "HAS", "WAG"]

Example 3
  Previous text: "My name is"
  Sequence: 4 1 2 1 3
  -> ["JACOB"]

Example 4
  Previous text: ""
  Sequence: 5 2
  -> ["IT", "IN", "IS"]
"""

class KeyboardPredictor:
    def __init__(self):
        # Use the newest OpenAI model unless changed by the user
        self.model = "gpt-4o"

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)

        # Frequency-based alphabet groups mapping for 6-button layout
        self.groups = {
            1: "EL",
            2: "TRCQ",
            3: "ADFV",
            4: "OHWZ",
            5: "ISKG",
            6: "NUMPYBJX",
        }

    def _context_suggests_name(self, context_text: str) -> bool:
        """Return True if the context likely indicates a name will follow."""
        text = context_text.lower().strip()
        text = re.sub(r"[.!?,]*$", "", text)
        endings = ["my name is", "name is", "i am", "i'm"]
        return any(text.endswith(e) for e in endings)

    def _context_is_sentence_start(self, context_text: str) -> bool:
        """Return True if the next word begins a new sentence."""
        if not context_text.strip():
            return True
        trimmed = context_text.rstrip()
        return trimmed.endswith(('.', '!', '?'))

    def predict_word(self, button_sequence, context_text=""):
        """
        Predict a word based on button sequence and context using OpenAI API.
        """
        if not button_sequence:
            return {"top_predictions": [], "alternative_words": []}

        prompt = self._build_prompt(button_sequence, context_text)
        temperature = 0.1

        # Two-pass LLM call: retry once with slightly higher temperature if invalid
        for attempt in range(2):
            response = self._call_llm(prompt, temperature)
            data = json.loads(response.choices[0].message.content)

            # Combine and uppercase
            raw_top = [w.upper() for w in data.get("top_predictions", [])]
            raw_alt = [w.upper() for w in data.get("alternative_words", [])]
            all_raw = raw_top + raw_alt

            # Validate each candidate
            valid = [w for w in all_raw if self._validate_word_sequence(w, button_sequence)]

            if self._context_suggests_name(context_text):
                name_candidates = [n.upper() for n in get_names_for_sequence(button_sequence, self.groups)]
                name_candidates = [n for n in name_candidates if self._validate_word_sequence(n, button_sequence)]
                if name_candidates:
                    valid = list(dict.fromkeys(name_candidates + valid))
            elif self._context_is_sentence_start(context_text):
                start_words = [w.upper() for w in self.predict_next_words(context_text)]
                start_words = [w for w in start_words if self._validate_word_sequence(w, button_sequence)]
                if start_words:
                    valid = list(dict.fromkeys(start_words + valid))

            if valid:
                return {
                    "top_predictions": valid[:3],
                    "alternative_words": valid[3:8],
                    "confidence": data.get("confidence", 0.0),
                }

            # If invalid, instruct model and relax temperature
            prompt = (
                "Your previous answer included invalid words. "
                "ONLY output words whose letters match the button groups exactly.\n\n"
                + prompt
            )
            temperature = 0.2

        # If still no valid output, return the raw predictions with low confidence
        # This allows user to see what the AI predicted even if validation failed
        if self._context_suggests_name(context_text):
            name_candidates = [n.upper() for n in get_names_for_sequence(button_sequence, self.groups)]
            name_candidates = [n for n in name_candidates if self._validate_word_sequence(n, button_sequence)]
            if name_candidates:
                return {
                    "top_predictions": name_candidates[:3],
                    "alternative_words": name_candidates[3:8],
                    "confidence": 0.1,
                    "validation_failed": True,
                }
        elif self._context_is_sentence_start(context_text):
            start_words = [w.upper() for w in self.predict_next_words(context_text)]
            start_words = [w for w in start_words if self._validate_word_sequence(w, button_sequence)]
            if start_words:
                return {
                    "top_predictions": start_words[:3],
                    "alternative_words": start_words[3:8],
                    "confidence": 0.1,
                    "validation_failed": True,
                }

        return {
            "top_predictions": raw_top[:3] if raw_top else [],
            "alternative_words": raw_alt[:5] if raw_alt else [],
            "confidence": 0.1,  # Low confidence to indicate these are unvalidated
            "validation_failed": True
        }

    def _build_prompt(self, button_sequence, context_text):
        """
        Construct the few-shot prompt including examples, legend, context, and sequence.
        """
        legend = "\n".join(f"- Button {k}: {', '.join(v)}" for k, v in self.groups.items())
        seq_str = " ".join(str(x) for x in button_sequence)
        context_line = f"Previous text: \"{context_text}\"\n" if context_text.strip() else ""
        start_line = "Sentence start: true\n" if self._context_is_sentence_start(context_text) else ""

        return f"""
{EXAMPLES}
Keyboard legend:
{legend}

{context_line}{start_line}Sequence: {seq_str}

Respond with JSON:
{{
  "top_predictions": ["word1", "word2", "word3"],
  "alternative_words": ["word4", "word5", "word6", "word7", "word8"],
  "confidence": 0.85
}}
"""

    def _call_llm(self, prompt, temperature):
        """
        Invoke the OpenAI chat completion endpoint with given temperature.
        """
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a word-prediction engine. NEVER output a word "
                        "whose letters do not match the specified button groups."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

    def _validate_word_sequence(self, word, button_sequence):
        """
        Ensure each letter of the word matches the corresponding button group.
        """
        if len(word) != len(button_sequence):
            print(f"Length mismatch: word '{word}' has {len(word)} letters, sequence has {len(button_sequence)} buttons")
            return False
        for i, (ch, btn) in enumerate(zip(word, button_sequence)):
            if ch not in self.groups.get(btn, ""):
                expected_letters = self.groups.get(btn, "")
                print(f"Validation failed for word '{word}' at position {i}: letter '{ch}' not in button {btn} group '{expected_letters}'")
                return False
        return True

    def predict_next_words(self, current_text, current_word=""):
        """
        Predict the next words based on context using OpenAI.
        """
        context = ((current_text + " " + current_word).strip()
                   if current_word else current_text.strip())
        if not context:
            return []

        prompt = f"""
Given this text: \"{context}\"

Predict the 3 most likely next words that would follow naturally.
Respond with JSON:
{{ "next_words": ["word1", "word2", "word3"] }}
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert at predicting the next word in English text."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            return result.get("next_words", [])
        except Exception:
            return []  # silent fallback
