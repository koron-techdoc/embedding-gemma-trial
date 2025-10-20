#!/usr/bin/env python

import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import csv
import numpy as np

class Prefecture:
    def __init__(self, name, embeddings):
        self.name = name
        self.embeddings = np.array(embeddings)

    def distance(self, embedding):
        return np.linalg.norm(np.array(embedding) - self.embeddings)

    @staticmethod
    def load(file):
        prefs = []
        with open(file, 'r', newline='', encoding='utf-8') as file:
            for row in csv.reader(file, delimiter='\t'):
                prefs.append(Prefecture(row[1], list(map(float, row[2:]))))
        return prefs

class LocalGov:
    def __init__(self, name, prefs):
        self.name = name
        self.prefs = prefs

    @staticmethod
    def load(file):
        lgovs = []
        with open(file, 'r', newline='', encoding='utf-8') as file:
            for row in csv.reader(file, delimiter='\t'):
                lgovs.append(LocalGov(row[0], row[1:]))
        return lgovs

def load_model(model_id='google/embeddinggemma-300m'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device='cuda')

def calc_top_k(embedding, prefs, k=10):
    scores = []
    for p in prefs:
        scores.append({'name': p.name, 'score': p.distance(embedding)})
    scores = sorted(scores, key=lambda x: x['score'])
    return scores[:k]

def count_matching(a, b):
    return len(set(a) & set(b))

task_name = 'Clustering'

def calc_accuracy(model, prefs, govs, batch_size=100):
    total = 0.0
    match = 0.0

    for i in range(0, len(govs), batch_size):
        chunk = govs[i:i + batch_size]
        names = [ g.name for g in chunk ]
        embeddings = model.encode(names, prompt=model.prompts['Clustering'])
        for j in range(len(chunk)):
            g = chunk[j]
            k = len(g.prefs)
            tops = calc_top_k(embeddings[j], prefs, k=k)
            top_names = [ e['name'] for e in tops ]
            count = count_matching(g.prefs, top_names)
            print(f'{g.name}\t{count}\t{"\t".join(top_names)}')

            total += k
            match += count

    return match / total

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model ID or directory', default='google/embeddinggemma-300m')
    parser.add_argument('-p', '--prefecture', help='prefecture embeddings', default='../embeddings.txt')
    parser.add_argument('-l', '--localgovs', help='local goverments', default='./truth_full.txt')
    parser.add_argument('-b', '--batchsize', help='batch size', default=100)
    args = parser.parse_args()

    govs = LocalGov.load(args.localgovs)
    prefs = Prefecture.load(args.prefecture)
    model = load_model(args.model)

    acc = calc_accuracy(model, prefs, govs, batch_size=int(args.batchsize))
    print()
    print(f'accuracy: {acc}')
