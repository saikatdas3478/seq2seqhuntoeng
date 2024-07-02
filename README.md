# Hungarian-English Translator using Seq2Seq Model

This repository contains an implementation of a sequence-to-sequence (Seq2Seq) model for translating Hungarian sentences to English. The model uses LSTM layers for both the encoder and decoder, and it is trained on paired Hungarian-English sentences.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)
- [License](#license)

## Introduction

This project implements a neural machine translation system for translating Hungarian text to English using a Seq2Seq model. Seq2Seq models are a type of neural network architecture commonly used for tasks where the input and output are sequences, such as translation, summarization, and conversational agents.

The model is trained on a dataset of Hungarian-English sentence pairs. The encoder processes the input Hungarian sentence and produces a context vector, which is then used by the decoder to generate the corresponding English sentence. The training process involves minimizing the difference between the predicted English sentence and the actual English sentence in the dataset.

## Dataset

The dataset used in this project consists of Hungarian-English sentence pairs. The training data can be obtained from the following URL:
- [Training Data](https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/hun_eng_pairs/hun_eng_pairs_train.txt)
- [Validation Data](https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/hun_eng_pairs/hun_eng_pairs_val.txt)

Each line in the dataset files contains a Hungarian sentence and its corresponding English translation, separated by the string `<sep>`.

## Methodology

The methodology for building and training the Hungarian-English translator involves several key steps:

1. **Data Preprocessing**: 
   - Load and split the dataset into Hungarian and English sentences.
   - Normalize Unicode characters and preprocess text by adding spaces around punctuation and converting to lowercase.
   - Add start-of-sequence (`<sos>`) and end-of-sequence (`<eos>`) tokens to the target sentences.

2. **Tokenization**:
   - Tokenize the Hungarian and English sentences using Keras' `Tokenizer` class.
   - Convert sentences to sequences of integers.

3. **Padding**:
   - Pad the sequences to ensure uniform input length for the model.

4. **Model Architecture**:
   - Define the Seq2Seq model with LSTM layers for both the encoder and decoder.
   - Use an embedding layer to represent input and output tokens.
   - Compile the model with an appropriate loss function and optimizer.

5. **Training**:
   - Train the model using the prepared data.
   - Use callbacks for model checkpointing and early stopping.

## Setup

To set up the environment and run the code, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saikatdas3478/seq2seqhuntoeng.git
   cd hun-eng-translator
   ```
2. **Install the required packages**:
```bash
pip install -r requirements.txt
```
3. **Download the dataset**:
- Download the training and validation data from the provided URLs and place them in the project directory.

4. **Run the training script**:
```bash
python train.py
```
## Training
The training process involves the following steps:

1. Load and preprocess the dataset.
2. Tokenize and pad the sequences.
3. Define the Seq2Seq model architecture.
4. Compile and train the model with appropriate callbacks for checkpointing and early stopping.

## Evaluation
After training, the model can be evaluated on the validation dataset to check its performance. The evaluation script computes the accuracy of the translations and displays sample translations for inspection.

##Results
The model's performance can be summarized by evaluating it on a set of test sentences and comparing the machine translations to the human translations. Example results are as follows:

| Original Sentence (Hungarian) | Human Translation (English) | Machine Translation (English) |
| ----------------------------- | --------------------------- | ----------------------------- |
| Csinálom.                     | I got it.                   | i'm doing it.                 |
| Mondd el nekem.               | Let me know.                | tell me.                      |
| Ritkán járok oda.             | I rarely go there.          | i rarely go there.            |


## References
- Sequence to Sequence Learning with Neural Networks
- Neural Machine Translation by Jointly Learning to Align and Translate
- The dataset I've used comes from Tatoeba, a collection of sentence translations in a variety of languages sourced from volunteers: [dataset](https://tatoeba.org/en)
