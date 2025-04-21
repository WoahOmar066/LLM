# LLM
A custom GPT-style transformer script for training and interactive chat, built from scratch with PyTorch and Hugging Face libraries.

# Custom GPT Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Model Architecture](#model-architecture)
4. [Features](#features)
5. [Training Implementation](#training-implementation)
6. [Text Generation](#text-generation)
7. [Usage Instructions](#usage-instructions)
8. [Configuration Parameters](#configuration-parameters)
9. [Advanced Details](#advanced-details)

## Overview

This project implements a custom GPT-like transformer model for text generation using PyTorch. The implementation includes a complete training pipeline, text generation capabilities, and an interactive chat mode. The model architecture is based on the transformer decoder with self-attention mechanisms.

## Project Structure

The script is structured into several key components:

```
├── System Configuration
│   ├── Warning Configuration
│   ├── Device Information
│   └── Logging Setup
│
├── Model Architecture
│   ├── PositionalEncoding
│   ├── CustomGPT
│   ├── MultiHeadSelfAttention
│   ├── FeedForward
│   └── TransformerBlock
│
├── Text Generation
│   ├── generate_text()
│   ├── generate_text_streaming()
│   └── interactive_chat()
│
├── Training Pipeline
│   ├── Data Preprocessing
│   ├── Checkpoint Management
│   └── Training Loop
│
└── Main Execution Logic
```

## Model Architecture

### CustomGPT

The core model architecture is implemented in the `CustomGPT` class, which uses a transformer decoder architecture with the following components:

```python
class CustomGPT(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 768, 
                 num_heads: int = 12, num_layers: int = 12, dropout: float = 0.1):
        super(CustomGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, 
                                                   nhead=num_heads, 
                                                   dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 
                                                         num_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self._cached_tgt_mask = None
```

### Key Components

1. **Positional Encoding**: Adds position information to token embeddings
2. **Transformer Decoder**: Processes sequential data with self-attention
3. **Multi-Head Self-Attention**: Enables the model to focus on different parts of the input
4. **Feed-Forward Networks**: Transforms representations between attention layers

## Features

### Hardware Optimization

- CUDA support for GPU acceleration
- Automatic mixed precision (AMP) training
- PyTorch compilation optimization where available
- Dynamic GPU/CPU worker allocation

### Training Enhancements

- Cyclical learning rate scheduling
- Adaptive learning rate adjustment
- Gradient clipping for stability
- Checkpoint saving and resumption
- Interrupt handling for safe training termination

### Text Generation

- Regular text generation
- Streaming text generation with token-by-token output
- Interactive chat interface with context management

## Training Implementation

The training loop includes several optimizations:

```python
def train(model, dataset, tokenizer, epochs, batch_size, checkpoint_path):
    # Device setup and optimization
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW([{
        'params': model.parameters(), 
        'lr': config["learning_rate"], 
        'initial_lr': config["learning_rate"]
    }])
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-6, max_lr=1e-4, 
        step_size_up=2000, mode="triangular2"
    )
    
    # Training loop with mixed precision
    for epoch in range(start_epoch, epochs):
        for batch_idx, batch in progress_bar:
            with autocast():
                outputs = model(input_ids)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

### Key Training Features:

- **Mixed Precision**: Uses `torch.cuda.amp` for faster training
- **Dynamic Learning Rate**: Automatically adjusts based on loss trends
- **Progress Tracking**: Uses tqdm for progress visualization
- **Logging**: Records training statistics to files

## Text Generation

The model supports two text generation modes:

### Standard Generation

Generates complete text at once:

```python
def generate_text(model, tokenizer, user_input, instructions, device="cpu"):
    input_ids = tokenizer.encode(instructions + "\n" + user_input, 
                                return_tensors="pt").to(device)
    # Text generation logic...
    return output
```

### Streaming Generation

Generates and displays text token-by-token in real-time:

```python
def generate_text_streaming(model, tokenizer, user_input, instructions, device="cpu"):
    # Similar to standard generation but prints tokens as they're generated
    for _ in range(50):
        # Generation logic...
        print(token_text, end='', flush=True)
    return full_output
```

## Usage Instructions

The script provides two main modes of operation:

### Training Mode

1. Loads or creates a tokenized dataset
2. Initializes the model and training parameters
3. Executes the training loop with checkpointing

### Conversation Mode

1. Loads a pre-trained model from checkpoint
2. Provides an interactive chat interface
3. Generates responses in streaming mode

To use:
```
Choose mode: 
[T] 'Train'  
[C] 'Converse'
```

## Configuration Parameters

The script uses the following default configuration:

```python
config = {
    "learning_rate": 1e-5,
    "max_lr": 1e-4,
    "batch_size": 8,
    "epochs": 3,
    "checkpoint_dir": "checkpoint",
    "log_dir": "logs",
    "instructions_file": "data/instructions.txt",
    "large_dataset_file": "data/large_dataset.txt"
}
```

## Advanced Details

### Attention Mechanism

The model uses multi-head self-attention to process sequential data:

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int):
        # Initialization...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Query, Key, Value projections
        Q = self.query(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention = torch.softmax(
            torch.einsum("nqhd,nkhd->nhqk", Q, K) / math.sqrt(self.head_dim), 
            dim=-1
        )
        
        # Output projection
        out = torch.einsum("nhql,nlhd->nqhd", attention, V).reshape(N, seq_len, emb_size)
        return self.fc_out(out)
```

### Data Preprocessing

The script handles data preprocessing for different datasets:

1. **OpenWebText**: General text corpus for training
   ```python
   def process_openwebtext(example, tokenizer):
       inputs = tokenizer(example["text"], truncation=True, 
                         padding="max_length", max_length=128, 
                         return_tensors="pt")
       return {
           "input_ids": inputs["input_ids"].squeeze(0).tolist(), 
           "labels": inputs["input_ids"].squeeze(0).tolist()
       }
   ```

2. **Conversation Format**: Structures data as user-model exchanges
   ```python
   def preprocess_conversations(example, tokenizer):
       conversation = f"User: {example['question']} Model: {example['answer']}"
       inputs = tokenizer(conversation, return_tensors="pt", 
                         max_length=512, truncation=True, 
                         padding="max_length")
       return {
           "input_ids": inputs["input_ids"].squeeze(), 
           "labels": inputs["input_ids"].squeeze()
       }
   ```
