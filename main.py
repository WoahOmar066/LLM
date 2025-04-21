import signal
import os 
import torch
import torch.nn as nn
import math
import logging
import warnings
import torch.cuda.amp
import psutil
import platform
from py3nvml.py3nvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import logging as transformers_logging
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from torch.nn import functional as F
from torch.cuda.amp import autocast
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings_configured = False  

# Enable some cudnn magic
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# --- Configure Warnings --- 
def configure_warnings(enable: bool):
    # Check if this function has already configured warnings
    if not hasattr(configure_warnings, "configured"):
        if enable:
            # Resetting warnings like a playful toddler, Mashallah!
            warnings.resetwarnings()
            transformers_logging.set_verbosity_warning()
            os.environ["PYTORCH_JIT_LOG_LEVEL"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        else:
            # Silencing warnings completely, wallahi no distractions!
            warnings.filterwarnings("ignore")
            transformers_logging.set_verbosity_error()
            os.environ["PYTORCH_JIT_LOG_LEVEL"] = "0"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # Mark as configured to avoid reconfiguration
        configure_warnings.configured = True
        
# --- Check Device Info (GPU, CPU, RAM) --- 
def check_device():
    print("\n==============================")
    print("GPU's:")
    nvmlInit()
    for i in range(torch.cuda.device_count()):
        print(f"{i + 1}. {torch.cuda.get_device_name(i)}")
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        print(f"   GPU RAM: {mem_info.total / (1024 ** 3):.2f} GB")
    print("\nCPU's:")
    print(f"1. {platform.processor()}")
    print("\nComputer RAM:")
    ram_info = psutil.virtual_memory()
    print(f"{ram_info.total // (1024 ** 3)} GB")
    print("==============================\n")

configure_warnings(enable=False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Positional Encoding Module (Yallah, let’s add positions!) ---
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Random toddler comment: Adding extra dims for fun!
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

# --- Custom GPT-like Model (Wallahi, it's lit!) ---
class CustomGPT(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 768, num_heads: int = 12, num_layers: int = 12, dropout: float = 0.1):
        super(CustomGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.position_encoding = PositionalEncoding(emb_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self._cached_tgt_mask = None  # Cache for target mask, so cool!

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        # Transpose if needed, mashallah
        if input_ids.dim() == 2 and input_ids.size(0) != input_ids.size(1):
            input_ids = input_ids.transpose(0, 1)
        embedded = self.embedding(input_ids)
        embedded = self.position_encoding(embedded)
        seq_len = embedded.size(0)
        if self._cached_tgt_mask is None or self._cached_tgt_mask.size(0) != seq_len:
            self._cached_tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        transformer_out = self.transformer_decoder(embedded, embedded, tgt_mask=self._cached_tgt_mask)
        return self.fc_out(transformer_out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configurations (Bismillah, let's set some parameters) ---
config = {
    "learning_rate": 1e-5,      # Base learning rate
    "max_lr": 1e-4,             # Not used now but still here for reference
    "batch_size": 8,
    "epochs": 3,
    "checkpoint_dir": "checkpoint",
    "log_dir": "logs",
    "instructions_file": "data/instructions.txt",
    "large_dataset_file": "data/large_dataset.txt"
}

os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["log_dir"], exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size

model = CustomGPT(vocab_size=vocab_size).to(device)

# --- Use torch.compile on non-Windows (Mashallah, optimization time!) ---
import os
if hasattr(torch, "compile") and os.name != "nt":
    model = torch.compile(model)
else:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

scaler = torch.cuda.amp.GradScaler()

# --- Logging Functions (Random logging, like a toddler scribbling!) ---
def log_message(log_type: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(config["log_dir"], f"{log_type}_log.txt")
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")

def log_training_data(epoch: int, batch_idx: int, loss: float):
    log_file = os.path.join(config["log_dir"], "training_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}\n")

# --- Text Generation Function (Inshallah, magic words!) ---
def generate_text(model: nn.Module, tokenizer: AutoTokenizer, user_input: str, instructions: str, device: str = "cpu") -> str:
    input_ids = tokenizer.encode(instructions + "\n" + user_input, return_tensors="pt").to(device)
    model.eval()
    generated = input_ids  
    for _ in range(50):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs[-1, 0, :]
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.zeros_like(logits)
            temperature = 1.0
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            next_token = torch.multinomial(probs, 1)
            next_token = next_token.unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output

def generate_text_streaming(model: nn.Module, tokenizer: AutoTokenizer, user_input: str, instructions: str, device: str = "cpu") -> str:
    input_ids = tokenizer.encode(instructions + "\n" + user_input, return_tensors="pt").to(device)
    model.eval()
    generated = input_ids  
    full_output = ""
    for _ in range(50):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs[-1, 0, :]
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.zeros_like(logits)
            temperature = 1.0
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            next_token = torch.multinomial(probs, 1)
            next_token = next_token.unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            full_output += token_text
            print(token_text, end='', flush=True)
    print()  # Newline after generation is complete
    return full_output


def interactive_chat(model: nn.Module, tokenizer: AutoTokenizer, device: str = "cpu") -> None:
    context = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q", "nahbroyoudoneforrealrealnoslackonthechimichangabagofcheesballsinsouthdakota"]:
            break
        context.append(f"You: {user_input}")
        print("Model: ", end='', flush=True)
        response = generate_text_streaming(model, tokenizer, user_input, "\n".join(context[-5:]), device=device)
        context.append(f"Model: {response}")
        log_message("chat", f"You: {user_input}\nModel: {response}")

# --- Multi-Head Self-Attention Module (Gibberish time!) ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads."
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, seq_len, emb_size = x.shape
        Q = self.query(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax(torch.einsum("nqhd,nkhd->nhqk", Q, K) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", attention, V).reshape(N, seq_len, emb_size)
        return self.fc_out(out)

# --- Feed Forward Network (Tiny brain power boost!) ---
class FeedForward(nn.Module):
    def __init__(self, emb_size: int, hidden_dim: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

# --- Transformer Block (Extra spicy transformer!) ---
class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm1(x + self.dropout(self.feed_forward(x)))

# --- Data Preprocessing Functions (For the data ninjas!) ---
def load_instructions(file_path: str) -> str:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            instructions = file.read()
        logging.info("Custom instructions loaded.")
        return instructions
    logging.warning("Instructions file not found. Using default behavior.")
    return ""

def preprocess_conversations(example: dict, tokenizer: AutoTokenizer) -> dict:
    conversation = f"User: {example['question']} Model: {example['answer']}"
    inputs = tokenizer(conversation, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    return {"input_ids": inputs["input_ids"].squeeze(), "labels": inputs["input_ids"].squeeze()}

def process_openwebtext(example: dict, tokenizer: AutoTokenizer) -> dict:
    inputs = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {"input_ids": inputs["input_ids"].squeeze(0).tolist(), "labels": inputs["input_ids"].squeeze(0).tolist()}

def get_gpu_worker_count():
    return torch.cuda.device_count() * 2 if torch.cuda.is_available() else os.cpu_count()

def get_data_loader(dataset: torch.utils.data.Dataset, tokenizer: AutoTokenizer, batch_size: int) -> DataLoader:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=get_gpu_worker_count(),
        prefetch_factor=4,
        persistent_workers=True
    )

# --- Checkpoint Saving/Loading (Safety first, inshallah!) ---
def save_checkpoint_on_interrupt(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, epoch: int) -> None:
    def handler(signum, frame):
        logging.info("Training interrupted. Saving checkpoint...")
        state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        logging.info("Checkpoint saved.")
        exit(1)
    signal.signal(signal.SIGINT, handler)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str) -> int:
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
        except Exception as e:
            logging.warning(f"Invalid checkpoint file at {checkpoint_path}. Removing file and starting from scratch. Error: {e}")
            os.remove(checkpoint_path)
            return 0
        state_dict = checkpoint['model_state_dict']
        if 'position_encoding.pe' in state_dict:
            current_shape = model.position_encoding.pe.shape
            ckpt_shape = state_dict['position_encoding.pe'].shape
            if ckpt_shape != current_shape:
                state_dict['position_encoding.pe'] = state_dict['position_encoding.pe'].transpose(0, 1)
        if hasattr(model, "_orig_mod"):
            if not any(key.startswith("_orig_mod") for key in state_dict.keys()):
                model._orig_mod.load_state_dict(state_dict)
            else:               model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0

def train(model: nn.Module, dataset: torch.utils.data.Dataset, tokenizer: AutoTokenizer, epochs: int, batch_size: int, checkpoint_path: str) -> None:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': config["learning_rate"], 'initial_lr': config["learning_rate"]}]
    )
    
    # Load checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path) if os.path.exists(checkpoint_path) else 0
    
    # Cyclical learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=2000, mode="triangular2"
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    data_loader = get_data_loader(dataset, tokenizer, batch_size=batch_size)
    
    save_checkpoint_on_interrupt(model, optimizer, checkpoint_path, start_epoch)
    global_step = 0
    prev_loss = float("inf")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(input_ids)
                outputs = outputs.permute(1, 0, 2).reshape(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Adaptive Learning Rate Adjustment
            if loss.item() > prev_loss * 1.01:  # Loss increasing significantly, decrease LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
            elif loss.item() < prev_loss * 0.99:  # Loss decreasing well, increase LR slightly
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1.05
            prev_loss = loss.item()
            
            total_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({
                "Loss": f"{total_loss / (batch_idx + 1):.4f}",
                "Batch/sec": f"{progress_bar.format_dict['rate']:.2f}" if progress_bar.format_dict['rate'] else "N/A"
            })
            
            if (batch_idx + 1) % 251 == 0:
                log_training_data(epoch, batch_idx, loss.item())
        
        avg_loss = total_loss / len(data_loader)
        logging.info(f"Epoch [{epoch + 1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }, checkpoint_path)
    
    logging.info("Training completed.")

if __name__ == "__main__":
    configure_warnings(enable=False)
    check_device()
    mode = input("Choose mode: \n[T] 'Train'  \n[C] 'Converse'\n").strip().lower()
    if mode == "t":
        if os.path.exists("tokenized_dataset"):
            tokenized_dataset = load_from_disk("tokenized_dataset")
        else:
            dataset = load_dataset("openwebtext", trust_remote_code=True)
            tokenized_dataset = dataset.map(
                lambda x: process_openwebtext(x, tokenizer),
                remove_columns=["text"]
            )
            tokenized_dataset.save_to_disk("tokenized_dataset")
            logging.info("Saved tokenized dataset to disk.")
        train(
            model=model,
            dataset=tokenized_dataset["train"],
            tokenizer=tokenizer,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            checkpoint_path=os.path.join(config["checkpoint_dir"], "model.pth")
        )
    elif mode == "c":
        checkpoint_path = os.path.join(config["checkpoint_dir"], "model.pth")
        load_checkpoint(model, None, checkpoint_path)
        model.eval()
        logging.info("Chat with the model! Type 'exit' to end the conversation.")
        interactive_chat(model, tokenizer, device=device)
    else:
        logging.error("Invalid mode selected. Please choose 'Train', 'Converse', or 'GUI'.")
