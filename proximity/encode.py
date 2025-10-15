#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################################################

def load_prefectures(file='prefectures.tsv'):
    import csv
    out = []
    with open(file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) > 1:
                out.append(row[1])
    return out

def load_model(model_id='google/embeddinggemma-300m'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device='cuda')

if __name__ == '__main__':
    import argparse
    import sys, io

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model ID or directory', default='google/embeddinggemma-300m')
    args = parser.parse_args()

    model_id = args.model

    task_name = 'Clustering'
    documents = load_prefectures()
    model = load_model(model_id)
    embeddings = model.encode(documents, prompt=model.prompts[task_name])

    wf = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    index = 0
    for pref in documents:
        wf.write(f'{index+1}\t{pref}')
        for v in embeddings[index]:
            wf.write(f'\t{v}')
        wf.write('\n')
        index += 1
