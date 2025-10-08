import csv

def load_prefectures(file='prefectures.tsv'):
    out = []
    with open(file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) > 1:
                out.append(row[1])
    return out

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m", device='cuda')

documents = load_prefectures()

embeddings = model.encode(documents, prompt="task: clustering | query: ")

with open('embeddings.txt', 'w', encoding='utf-8') as wf:
    index = 0
    for pref in documents:
        wf.write(f'{index+1}\t{pref}')
        for v in embeddings[index]:
            wf.write(f'\t{v}')
        wf.write('\n')
        index += 1
