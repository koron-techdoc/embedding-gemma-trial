#!/usr/bin/env python
#
# Show prompts which embedded in the model.
#
# Exapmle:
#
#   Show prompts of google/embeddinggemma-300m
#
#       $ ./show_prompts.py
#
#   Show prompts of any model (Qen/Qwen3-Embedding-0.6B
#
#       $ ./show_prompts.py --model Qwen/Qwen3-Embedding-0.6B

from sentence_transformers import SentenceTransformer

def load_model(model_id='google/embeddinggemma-300m'):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device='cuda')

def print_prompts(prompts):
    for key in prompts:
        print(f"{key}\t{repr(prompts[key])}")

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model ID or directory', default='google/embeddinggemma-300m')
    args = parser.parse_args()

    model = load_model(args.model)
    print_prompts(model.prompts)
