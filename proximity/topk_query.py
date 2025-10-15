#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

class Anchor:
    def __init__(self, name, embeddings):
        self.name = name
        self.embeddings = np.array(embeddings)

    def distance(self, embedding):
        return np.linalg.norm(np.array(embedding) - self.embeddings)

def load_anchors(file):
    import csv
    anchors = []
    with open(file, 'r', newline='', encoding='utf-8') as file:
        for row in csv.reader(file, delimiter='\t'):
            anchors.append(Anchor(row[1], list(map(float, row[2:]))))
    return anchors

def load_model(model_id='google/embeddinggemma-300m'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device='cuda')

def calc_topk(sentence, model, anchors, k=10, task_name='Clustering'):
    embeddings = model.encode([sentence], prompt=model.prompts[task_name])
    embedding = embeddings[0]
    scores = []
    for a in anchors:
        scores.append({
            'name': a.name,
            'score': a.distance(embedding),
        })
    scores = sorted(scores, key=lambda x: x['score'])
    return scores[:k]

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model ID or directory', default='google/embeddinggemma-300m')
    parser.add_argument('-a', '--anchor', help='anchor vectors', default='embeddings.txt')
    args = parser.parse_args()

    anchors = load_anchors(args.anchor)

    model = load_model(args.model)

    for line in sys.stdin:
        sentence = line.strip()
        top_k = calc_topk(sentence, model, anchors, k=11)
        print(f'{sentence}:')
        rank = 0
        for entry in top_k:
            print(f'  #{rank}\t{entry["name"]}\t{entry["score"]}')
            rank += 1
