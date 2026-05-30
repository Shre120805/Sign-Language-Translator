import time
from collections import deque, Counter
from difflib import get_close_matches

# ─── DICTIONARY ───────────────────────────────────────────────────────────────
try:
    import nltk
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words
    DICTIONARY = {w.upper() for w in nltk_words.words()}
    print(f"[NLP] Dictionary loaded: {len(DICTIONARY):,} words (NLTK)")
except Exception:
    DICTIONARY = {
        "HELLO", "WORLD", "HELP", "PLEASE", "THANK", "THANKS", "YES", "NO",
        "GOOD", "BAD", "LOVE", "WANT", "NEED", "HAVE", "LIKE", "COME", "STOP",
        "WAIT", "GO", "FINE", "OKAY", "HI", "BYE", "NAME", "NICE", "MEET",
        "FOOD", "WATER", "TIME", "HOME", "WORK", "DAY", "NIGHT", "LEARN",
        "SIGN", "SORRY", "PAIN", "SICK", "HAPPY", "SAD", "TIRED", "HOT", "COLD",
    }
    print(f"[NLP] Dictionary loaded: {len(DICTIONARY)} words (fallback)")


class NLPPipeline:
    

    def __init__(
        self,
        stability_window=10,
        stability_threshold=0.6,
        min_confidence=0.65,
        letter_cooldown=1.2,
        word_pause_threshold=2.0,
        sentence_pause=5.0,
        max_word_length=15,
        spell_correct=True,
        correction_cutoff=0.65,
    ):
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.min_confidence = min_confidence
        self.letter_cooldown = letter_cooldown
        self.word_pause_threshold = word_pause_threshold
        self.sentence_pause = sentence_pause
        self.max_word_length = max_word_length
        self.spell_correct = spell_correct
        self.correction_cutoff = correction_cutoff

        self.letter_buffer = deque(maxlen=stability_window)
        self.word_buffer = []
        self.sentence = []
        self.last_accepted_letter = None
        self.last_letter_time = 0.0
        self.last_input_time = time.time()

        self.total_letters_received = 0
        self.total_letters_accepted = 0
        self.total_words_formed = 0

    def receive(self, letter, confidence):
        """Process a CNN prediction frame."""
        now = time.time()
        self.total_letters_received += 1

        # Skip 'nothing' / 'space' classes
        if letter in ('nothing', None):
            return self._state(event="NOTHING")
        if letter == 'space':
            self.manual_word_break()
            return self._state(event="SPACE")

        if confidence < self.min_confidence:
            # Motion signs (J, Z) use 60% threshold, others use 65%
            threshold = 0.60 if letter in ('j', 'J', 'z', 'Z') else self.min_confidence
            if confidence < threshold:
                return self._state(event="LOW_CONF")

        self.letter_buffer.append(letter.upper())
        stable = self._get_stable_letter()
        if stable is None:
            return self._state(event="UNSTABLE")

        same_as_last = (stable == self.last_accepted_letter)
        cooldown_ok = (now - self.last_letter_time) >= self.letter_cooldown
        if same_as_last and not cooldown_ok:
            return self._state(event="COOLDOWN")

        # Accept
        self.last_accepted_letter = stable
        self.last_letter_time = now
        self.last_input_time = now
        self.total_letters_accepted += 1

        if len(self.word_buffer) < self.max_word_length:
            self.word_buffer.append(stable)

        return self._state(event="ACCEPTED", letter=stable)

    def _get_stable_letter(self):
        if len(self.letter_buffer) < self.stability_window // 2:
            return None
        counts = Counter(self.letter_buffer)
        top, top_count = counts.most_common(1)[0]
        if top_count / len(self.letter_buffer) >= self.stability_threshold:
            return top
        return None

    def tick(self):
        now = time.time()
        elapsed = now - self.last_input_time
        result = {}

        if elapsed >= self.word_pause_threshold and self.word_buffer:
            word = self._form_word()
            if word:
                result['word'] = word
                self.sentence.append(word)
                self.total_words_formed += 1
            self.word_buffer = []
            self.last_input_time = now

        if elapsed >= self.sentence_pause and self.sentence:
            result['sentence'] = " ".join(self.sentence)
            self.sentence = []

        return result if result else None

    def _form_word(self):
        raw = "".join(self.word_buffer)
        if not raw:
            return None
        if raw in DICTIONARY:
            return raw
        if self.spell_correct and len(raw) >= 2:
            candidates = get_close_matches(raw, DICTIONARY, n=1, cutoff=self.correction_cutoff)
            if candidates:
                return candidates[0]
        return raw

    def manual_word_break(self):
        if self.word_buffer:
            word = self._form_word()
            if word:
                self.sentence.append(word)
                self.total_words_formed += 1
            self.word_buffer = []
            return word
        return None

    def delete_last_letter(self):
        if self.word_buffer:
            return self.word_buffer.pop()
        return None

    def get_current_word_raw(self):
        return "".join(self.word_buffer)

    def get_sentence(self):
        parts = self.sentence.copy()
        if self.word_buffer:
            parts.append(f"[{self.get_current_word_raw()}]")
        return " ".join(parts)

    def reset(self):
        self.letter_buffer.clear()
        self.word_buffer = []
        self.sentence = []
        self.last_accepted_letter = None
        self.last_letter_time = 0.0
        self.last_input_time = time.time()

    def get_stats(self):
        return {
            "received": self.total_letters_received,
            "accepted": self.total_letters_accepted,
            "words": self.total_words_formed,
        }

    def _state(self, event="", letter=""):
        return {
            "event": event,
            "letter": letter,
            "word_buffer": self.get_current_word_raw(),
            "sentence": self.get_sentence(),
        }
