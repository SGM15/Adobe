from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./sbert_model')
print("Model saved as'sbert_model'")