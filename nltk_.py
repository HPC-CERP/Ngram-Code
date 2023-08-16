import nltk
from nltk import bigrams, trigrams
from nltk.probability import FreqDist, ConditionalFreqDist

# Sample text corpus
corpus = "the cat is on the roof the cat is on the table"

# Tokenization - Split the text into words
tokens = nltk.word_tokenize(corpus)

# Create unigrams, bigrams, and trigrams
unigrams = tokens
bigrams = list(bigrams(tokens))
trigrams = list(trigrams(tokens))

# Compute frequency distributions
unigram_freq = FreqDist(unigrams)
bigram_freq = ConditionalFreqDist(bigrams)
trigram_freq = ConditionalFreqDist([(w1_w2, w3) for w1_w2, _, w3 in trigrams])  # Fixed this line

# Test the model
test_word = "on"
unigram_prob = unigram_freq.freq(test_word)

# Display predictions for unigram
print(f"Predictions for '{test_word}' using unigram model:")
for word in unigram_freq:
    prob = unigram_freq.freq(word)
    print(f"{word}: {prob:.4f}")
