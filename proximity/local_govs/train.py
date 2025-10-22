#!/usr/bin/env python

import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traindata', help='training data', default='train_full.tsv')
parser.add_argument('-o', '--outdir', help='output model directory', default='trained-lgov-full')
args = parser.parse_args()

TRAIN_DATA = args.traindata
OUTPUT_MODEL_DIR = args.outdir
EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 2e-5

# MAGIC NUMBER
SAVE_STEPS = 20
LOGGING_STEPS = 20

##############################################################################

import csv

dataset = []
with open(TRAIN_DATA, 'r', encoding='utf-8') as file:
    last = ''
    for row in csv.reader(file, delimiter='\t'):
        dataset.append(row)
logger.info('loaded training data')

##############################################################################

from datasets import Dataset

# Convert the list-based dataset into a list of dictionaries.
data_as_dicts = [ {'anchor': row[0], 'positive': row[1], 'negative': row[2]} for row in dataset ]

# Create a Hugging Face `Dataset` object from the list of dictionaries.
train_dataset = Dataset.from_list(data_as_dicts)
logger.info(f'Transformed the dataset for training: {train_dataset}')

##############################################################################

import torch
from sentence_transformers import SentenceTransformer

model_id = 'google/embeddinggemma-300M'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(model_id).to(device=device)
logger.info(f'Loaded the model: {model}')

##############################################################################

task_name = 'Clustering'

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import TrainerCallback

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=OUTPUT_MODEL_DIR,
    # Optional training parameters:
    prompts=model.prompts[task_name],    # use model's prompt to train
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    # Optional tracking/debugging parameters:
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to='none',
)

class MyCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_log(self, args, state, control, **kwargs):
        print(f'Step {state.global_step} finished.')

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    callbacks=[MyCallback()]
)
trainer.train()
