from collections import defaultdict

# Sample text corpus
corpus = "the cat is on the roof the cat is on the table"

# Tokenization - Split the text into words
tokens = corpus.split()

# Create a dictionary to store counts of unigrams
unigram_counts = defaultdict(int)

# Count the occurrences of each word (unigram)
for token in tokens:
    unigram_counts[token] += 1

# Compute probabilities for unigrams
unigram_probabilities = {}
total_tokens = len(tokens)
for word, count in unigram_counts.items():
    unigram_probabilities[word] = count / total_tokens

# Test the model
test_word = "on"
predicted_words = []
for word in unigram_probabilities:
    predicted_words.append((word, unigram_probabilities[word]))

# Sort predictions by probability in descending order
predicted_words.sort(key=lambda x: x[1], reverse=True)

# Display predictions
print(f"Predictions for '{test_word}':")
for word, prob in predicted_words:
    print(f"{word}: {prob:.4f}")
