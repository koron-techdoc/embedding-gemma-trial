from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m", device='cuda')

print(model.prompts)
