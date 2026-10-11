corpus = [
    "machine learning is fun",
    "deep learning is a part of machine learning",
    "word embeddings capture meaning of words",
    "neural networks learn useful representations"
]

processed_corpus = [
    sentence.lower().split()
    for sentence in corpus
]

print(processed_corpus)

from gensim.models import Word2Vec

model = Word2Vec(
    sentences=processed_corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,
    epochs=200
)

print("Training complete.")

embedding = model.wv["networks"]

print("Word embedding for 'networks':")
print(embedding)
