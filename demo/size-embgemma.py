from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m", device='cpu')

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"

query_embeddings = model.encode_query(query)

print(query_embeddings.shape)
print(query_embeddings.dtype)
