from typing import List, Dict
from collections import defaultdict, Counter
import math

def get_char_ngrams(text: str, n: int) -> List[str]:
    return [text[i:i+n] for i in range(len(text)-n+1)]

class NBSentimentModel:
    def __init__(self, ngram_size: int = 2):
        """
        NBSentimentModel constructor

        Args:
            ngram_size (int, optional): Size of character n-grams. Defaults to 2.
        """
        self._priors = {}
        self._likelihoods = {}
        self.ngram_size = ngram_size
        self._vocab = set()

    def fit(self, train_sentences: List[str], train_labels: List[int]):
        """
        Train the Naive Bayes model for sentiment analysis

        Args:
            train_sentences (List[str]): Sentences from the training data
            train_labels (List[int]): Labels from the training data (1 for good, 0 for bad)
        """
        # Calculate priors
        label_count = Counter(train_labels)
        total_docs = len(train_labels)
        for label in label_count:
            self._priors[label] = math.log(label_count[label] / total_docs)
        
        # Calculate likelihoods
        ngram_counts = defaultdict(Counter)
        for sentence, label in zip(train_sentences, train_labels):
            ngrams = get_char_ngrams(sentence, self.ngram_size)
            for ngram in ngrams:
                ngram_counts[label][ngram] += 1
                self._vocab.add(ngram)

        # Apply Laplace smoothing
        for label in ngram_counts:
            total_ngrams = sum(ngram_counts[label].values())
            vocab_size = len(self._vocab)
            for ngram in self._vocab:
                count = ngram_counts[label].get(ngram, 0)
                self._likelihoods.setdefault(label, {})
                self._likelihoods[label][ngram] = math.log((count + 1) / (total_ngrams + vocab_size))

    def predict(self, test_sentences: List[str]) -> List[int]:
        """
        Predict labels for a list of sentences

        Args:
            test_sentences (List[str]): Sentences to predict the sentiment of

        Returns:
            List[int]: The predicted labels (1 for good, 0 for bad)
        """
        predictions = []
        for sentence in test_sentences:
            log_probs = self.predict_one_log_proba(sentence)
            predicted = max(log_probs, key=log_probs.get)  # get the label with the highest log probability
            predictions.append(predicted)
        return predictions

    def predict_one_log_proba(self, test_sentence: str) -> Dict[int, float]:
        """
        Computes the log probability of a single sentence being good or bad

        Args:
            test_sentence (str): the sentence to predict the sentiment of

        Returns:
            Dict[int, float]: mapping of label --> probability
        """
        log_prob = {}
        ngrams = get_char_ngrams(test_sentence, self.ngram_size)
        for label in self._likelihoods:
            log_prob[label] = self._priors[label]  # Initialize log probability for the label
            for ngram in ngrams:
                if ngram in self._likelihoods[label]:
                    log_prob[label] += self._likelihoods[label][ngram]  # Accumulate log probability for each ngram
        return log_prob