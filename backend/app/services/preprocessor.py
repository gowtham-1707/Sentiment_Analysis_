import re
import string
from functools import lru_cache
from typing import List, Optional

import nltk
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources() -> None:
    resources = [
        ("corpora/stopwords",   "stopwords"),
        ("corpora/wordnet",     "wordnet"),
        ("corpora/omw-1.4",     "omw-1.4"),
        ("tokenizers/punkt",    "punkt"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

download_nltk_resources()

class TextPreprocessor:
    CONTRACTIONS = {
        "ain't":    "is not",
        "aren't":   "are not",
        "can't":    "cannot",
        "couldn't": "could not",
        "didn't":   "did not",
        "doesn't":  "does not",
        "don't":    "do not",
        "hadn't":   "had not",
        "hasn't":   "has not",
        "haven't":  "have not",
        "he'd":     "he would",
        "he'll":    "he will",
        "he's":     "he is",
        "i'd":      "i would",
        "i'll":     "i will",
        "i'm":      "i am",
        "i've":     "i have",
        "isn't":    "is not",
        "it's":     "it is",
        "let's":    "let us",
        "mustn't":  "must not",
        "shan't":   "shall not",
        "she'd":    "she would",
        "she'll":   "she will",
        "she's":    "she is",
        "shouldn't":"should not",
        "that's":   "that is",
        "there's":  "there is",
        "they'd":   "they would",
        "they'll":  "they will",
        "they're":  "they are",
        "they've":  "they have",
        "wasn't":   "was not",
        "we'd":     "we would",
        "we're":    "we are",
        "we've":    "we have",
        "weren't":  "were not",
        "what'll":  "what will",
        "what're":  "what are",
        "what's":   "what is",
        "what've":  "what have",
        "where's":  "where is",
        "who'll":   "who will",
        "who's":    "who is",
        "won't":    "will not",
        "wouldn't": "would not",
        "you'd":    "you would",
        "you'll":   "you will",
        "you're":   "you are",
        "you've":   "you have",
    }

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
        language: str = "english",
    ) -> None:
        self.remove_stopwords  = remove_stopwords
        self.lemmatize         = lemmatize
        self.min_token_length  = min_token_length
        self.language          = language
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = (
            set(stopwords.words(language)) if remove_stopwords else set()
        )

        negation_words = {
            "no", "not", "nor", "neither", "never", "nobody",
            "nothing", "nowhere", "neither", "hardly", "barely", "scarcely"
        }
        self.stop_words -= negation_words

        logger.info(
            f"TextPreprocessor initialized | "
            f"stopwords={remove_stopwords} | "
            f"lemmatize={lemmatize} | "
            f"min_token_length={min_token_length}"
        )

    def _to_lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _remove_urls(self, text: str) -> str:
        return re.sub(
            r"http[s]?://\S+|www\.\S+", " ", text
        )

    def _remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def _expand_contractions(self, text: str) -> str:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in self.CONTRACTIONS.keys()) + r")\b",
            re.IGNORECASE,
        )
        def replace(match):
            return self.CONTRACTIONS[match.group(0).lower()]
        return pattern.sub(replace, text)

    def _remove_special_characters(self, text: str) -> str:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text

    def _remove_numbers(self, text: str) -> str:
        return re.sub(r"\b\d+\b", " ", text)

    def _remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def _filter_short_tokens(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if len(t) >= self.min_token_length]

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        lemmatized = []
        for token in tokens:
            verb_form = self.lemmatizer.lemmatize(token, pos="v")
            if verb_form != token:
                lemmatized.append(verb_form)
            else:
                lemmatized.append(self.lemmatizer.lemmatize(token, pos="n"))
        return lemmatized

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")

        if not text.strip():
            return ""

        try:
            text = self._to_lowercase(text)
            text = self._remove_html_tags(text)
            text = self._remove_urls(text)
            text = self._remove_emails(text)
            text = self._expand_contractions(text)
            text = self._remove_special_characters(text)
            text = self._remove_numbers(text)
            text = self._remove_extra_whitespace(text)
            tokens = self._tokenize(text)
            if self.remove_stopwords:
                tokens = self._remove_stopwords(tokens)
            tokens = self._filter_short_tokens(tokens)
            if self.lemmatize and self.lemmatizer:
                tokens = self._lemmatize_tokens(tokens)

            return " ".join(tokens)

        except Exception as e:
            logger.error(f"Preprocessing failed for text: '{text[:50]}...' | Error: {e}")
            raise

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        logger.info(f"Preprocessing batch of {len(texts)} reviews...")
        results = []
        for i, text in enumerate(texts):
            try:
                results.append(self.preprocess(text))
            except Exception as e:
                logger.warning(f"Skipping review {i} due to error: {e}")
                results.append("")  # Return empty string for failed items
        logger.info(f"Batch preprocessing complete.")
        return results


@lru_cache(maxsize=1)
def get_preprocessor(
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_token_length: int = 2,
) -> TextPreprocessor:
    
    return TextPreprocessor(
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
        min_token_length=min_token_length,
    )



if __name__ == "__main__":
    sample_reviews = [
        "This product is AMAZING!!! Best purchase I've ever made.",
        "Terrible quality. Don't buy this. Complete waste of money!!!",
        "It's okay, not great not bad. Does what it's supposed to do.",
        "<br/>Check out www.example.com for more info. Email us at help@example.com",
        "The battery life isn't good at all. Very disappointed.",
    ]

    preprocessor = get_preprocessor()

    print("\n" + "="*60)
    print("TextPreprocessor — Sample Output")
    print("="*60)
    for review in sample_reviews:
        clean = preprocessor.preprocess(review)
        print(f"\n  Original : {review}")
        print(f"  Cleaned  : {clean}")
    print("="*60)