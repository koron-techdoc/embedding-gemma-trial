import time

t = []
def t_split(label):
    t.append((label, time.time()))
def t_list():
    for i in range(1, len(t)):
        print(f'#{i}\t{t[i][1] - t[i-1][1]}\t{t[i][0]}')
t_split('start')

from sentence_transformers import SentenceTransformer
t_split('import sentence_transformers')

# Download from the 🤗 Hub
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cuda')
t_split('load model')

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
t_split('define query and documents')

query_embeddings = model.encode_query(query)
t_split('embedding query')

document_embeddings = model.encode_document(documents)
t_split('embedding documents')

print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)
t_split('print shape of embeddings')

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
t_split('calculate similarities')

print(similarities)
t_split('print similarities')
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])

t_list()
