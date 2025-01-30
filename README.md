# Neural Networks From Scratch

This repository contains implementations of three fundamental neural network architectures from scratch using PyTorch: Word Embeddings, LSTM (Long Short-Term Memory), and Transformers. Each implementation is designed to be minimal yet functional, serving as educational examples of these architectures.

## Implementations

### 1. Word Embeddings
A simple implementation of word embeddings that learns to represent words in a 2D space while capturing sequential relationships between words. Features:
- Custom word embedding layer
- Visualization of word vectors in 2D space
- Sequential word prediction task

### 2. LSTM (Long Short-Term Memory)
A basic LSTM implementation that demonstrates the core mechanisms of long-term memory in neural networks. Features:
- Custom LSTM cell implementation
- Memory gates (forget, input, output)
- Sequence prediction capabilities

### 3. Transformer (Decoder-Only)
A simplified implementation of a decoder-only transformer, similar to the architecture used in models like GPT. Features:
- Self-attention mechanism
- Positional encoding
- Multi-token prediction

## Requirements
```
torch
matplotlib
seaborn
pandas
```

## Installation
```bash
git clone https://github.com/Oussiddi/neural-networks-from-scratch.git
pip install -r requirements.txt
```

## Project Structure
```
neural-networks-from-scratch/
│── word_embedding.py
│── lstm.py
│── transformer.py
├── requirements.txt
└── README.md
```

## Usage Examples

### Word Embeddings
```python
from word_embedding import WordEmbedding, test_word_embedding

# Train and visualize word embeddings
model = test_word_embedding()
```

### LSTM
```python
from lstm import Lstm, test_lstm

# Train and test LSTM model
model = test_lstm()
```

### Transformer
```python
from transformer import DecoderOnlyTransformer, test_transformer

# Train and test transformer model
model = test_transformer()
```

## Results

Each model demonstrates basic capabilities in its respective domain:
- Word Embeddings: Learns meaningful 2D representations of words
- LSTM: Captures long-term dependencies in sequential data
- Transformer: Performs basic sequence-to-sequence tasks with attention

