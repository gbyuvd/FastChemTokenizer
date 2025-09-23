#
# Molecule Tokenizer Benchmark & VAE Training Pipeline
# 
#

#
# Step 1.1 — Imports & Reproducibility
#

import os
import time
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# Tokenizers
from transformers import AutoTokenizer
from FastChemTokenizer import FastChemTokenizer  # assuming it's in PYTHONPATH
# Optional: for progress bars
from tqdm import tqdm
from rdkit import Chem
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from ranger21 import Ranger21
from torch.utils.data import DataLoader, Dataset
from scipy.stats import entropy
import json

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#
# Step 1.2 — Load & Preprocess SMILES Corpus
#

data_path = "./sample_all_8k_smi.csv"
df = pd.read_csv(data_path)

if 'SMILES' not in df.columns:
    raise ValueError("Expected column 'SMILES' in CSV")

smiles_list = df['SMILES'].dropna().tolist()
print(f"Loaded {len(smiles_list)} SMILES (assumed pre-canonicalized)")

# Validate with RDKit

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

print("Validating SMILES with RDKit...")
valid_mask = [is_valid_smiles(s) for s in tqdm(smiles_list)]
smiles_list = [s for s, valid in zip(smiles_list, valid_mask) if valid]
print(f"After RDKit filtering: {len(smiles_list)} valid SMILES")

#
# Step 1.3 — Train/Val/Test Split (80/10/10)
#

train_smiles, temp_smiles = train_test_split(smiles_list, test_size=0.2, random_state=42, shuffle=True)
val_smiles, test_smiles = train_test_split(temp_smiles, test_size=0.5, random_state=42, shuffle=True)

print(f"Train: {len(train_smiles)}")
print(f"Val:   {len(val_smiles)}")
print(f"Test:  {len(test_smiles)}")

# Cache splits
splits = {'train': train_smiles, 'val': val_smiles, 'test': test_smiles}
for split_name, smiles in splits.items():
    with open(f"../data/{split_name}_smiles.txt", "w") as f:
        f.write("\n".join(smiles))

#
# Step 1.4 — Tokenizer Wrapper (Fixed Bug #2, #3, #6)
#

class TokenizerWrapper:
    def __init__(self, tokenizer, name, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>"):
        self.tokenizer = tokenizer
        self.name = name
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        if hasattr(tokenizer, 'add_special_tokens'):
            tokenizer.add_special_tokens({
                'bos_token': bos_token,
                'eos_token': eos_token,
                'pad_token': pad_token,
                'unk_token': unk_token
            })

    def encode(self, smiles: str, add_special_tokens: bool = True):
        if isinstance(self.tokenizer, FastChemTokenizer):
            # 1. get ids directly
            ids = self.tokenizer.encode(smiles)          # ← no .tokenize() here
            # 2. add specials ourselves
            if add_special_tokens:
                ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
            return {'input_ids': ids}
        else:
            # Hugging-Face style tokenizer
            return self.tokenizer(
                smiles,
                add_special_tokens=add_special_tokens,
                return_attention_mask=False,
                return_tensors=None
            )

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(self.tokenizer, FastChemTokenizer):
            # 1. map single ids → tokens
            tokens = [self.tokenizer.id_to_token.get(tid, self.tokenizer.unk_token)
                    for tid in token_ids]
            # 2. drop specials if requested
            if skip_special_tokens:
                specials = {self.tokenizer.bos_token,
                            self.tokenizer.eos_token,
                            self.tokenizer.pad_token,
                            self.tokenizer.unk_token}   # add any others you use
                tokens = [t for t in tokens if t not in specials]
            # 3. detokenise
            if hasattr(self.tokenizer, 'detokenize'):
                return self.tokenizer.detokenize(tokens)
            else:
                return "".join(tokens)          # chemistry tokens are atomic
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self):
        if isinstance(self.tokenizer, FastChemTokenizer):
            # FastChemTokenizer uses ._vocab or .vocab depending on version
            return len(getattr(self.tokenizer, 'vocab',
                            getattr(self.tokenizer, '_vocab', self.tokenizer)))
        else:
            return len(self.tokenizer)

    def get_vocab(self):
        if isinstance(self.tokenizer, FastChemTokenizer):
            return self.tokenizer.vocab
        else:
            return self.tokenizer.get_vocab()

#
# Step 1.5 — Initialize Tokenizers
#

tok1_hf = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tok2_fast = FastChemTokenizer.from_pretrained("../smitok")

tokenizer1 = TokenizerWrapper(tok1_hf, name="ChemBERTa", bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
tokenizer2 = TokenizerWrapper(tok2_fast, name="FastChemTokenizer", bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]")

TOKENIZERS = [tokenizer1, tokenizer2]

#
# Step 1.6 — Benchmarking Functions (Fixed Bug #4 implicitly via epsilon)
#

def benchmark_tokenizer(tokenizer, smiles_sample, encode_only=False):
    V = len(tokenizer)
    sample = smiles_sample[:10000] if len(smiles_sample) > 10000 else smiles_sample

    encode_times = []
    token_counts = []
    char_counts = []
    unk_counts = 0
    total_tokens = 0

    for smiles in tqdm(sample, desc=f"Encoding with {tokenizer.name}", leave=False):
        char_counts.append(len(smiles))

        start = time.perf_counter()
        enc = tokenizer.encode(smiles, add_special_tokens=True)
        end = time.perf_counter()
        encode_times.append(end - start)

        input_ids = enc['input_ids']
        token_counts.append(len(input_ids))
        total_tokens += len(input_ids)

        if isinstance(tokenizer.tokenizer, FastChemTokenizer):
            unk_id = tokenizer.tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        else:
            unk_id = tokenizer.tokenizer.unk_token_id

        unk_counts += input_ids.count(unk_id)

    L̅ = np.mean(token_counts)
    C = np.mean(char_counts) / L̅
    U = unk_counts / total_tokens if total_tokens > 0 else 0.0
    Tenc = len(sample) / sum(encode_times)

    metrics = {
        'vocab_size': V,
        'avg_tokens_per_mol': L̅,
        'compression_ratio': C,
        'percent_unknown': U * 100,
        'encode_throughput_smiles_per_sec': Tenc,
    }

    if encode_only:
        return metrics

    decode_times = []
    reconstruction_ok = 0

    for smiles in tqdm(sample, desc=f"Decoding with {tokenizer.name}", leave=False):
        enc = tokenizer.encode(smiles, add_special_tokens=True)
        input_ids = enc['input_ids']

        start = time.perf_counter()
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        end = time.perf_counter()
        decode_times.append(end - start)

        if decoded == smiles:
            reconstruction_ok += 1

    Tdec = len(sample) / sum(decode_times)
    recon_acc = reconstruction_ok / len(sample)

    metrics.update({
        'decode_throughput_smiles_per_sec': Tdec,
        'decode_reconstruction_accuracy': recon_acc * 100,
    })

    return metrics

#
# Step 1.7 — Run Benchmark
#

benchmark_sample = train_smiles
results = []

for tokenizer in TOKENIZERS:
    print(f"\n=== Benchmarking {tokenizer.name} ===")
    metrics = benchmark_tokenizer(tokenizer, benchmark_sample)
    metrics['tokenizer'] = tokenizer.name
    results.append(metrics)
    for k, v in metrics.items():
        if k != 'tokenizer':
            print(f"{k:35s}: {v:.4f}" if isinstance(v, float) else f"{k:35s}: {v}")

df_results = pd.DataFrame(results)
df_results.to_csv("tokenizer_benchmark_results.csv", index=False)
print("\nTokenizer benchmark results saved to 'tokenizer_benchmark_results.csv'")

#
# Step 2.1 — VAE Model Class
#

class MoleculeVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=128, num_layers=2, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.latent2hidden = nn.Linear(latent_dim, num_layers * hidden_dim)
        self.latent2cell = nn.Linear(latent_dim, num_layers * hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def encode(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, _) = self.encoder_lstm(packed)
        h_forward = hidden[-2]
        h_backward = hidden[-1]
        h = torch.cat([h_forward, h_backward], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, target_seq=None, max_length=128, teacher_forcing_ratio=0.5, temperature=1.0):
        B = z.size(0)
        h0 = self.latent2hidden(z).view(self.num_layers, B, self.hidden_dim)
        c0 = self.latent2cell(z).view(self.num_layers, B, self.hidden_dim)
        hidden = (h0, c0)

        if target_seq is not None and random.random() < teacher_forcing_ratio:
            embedded = self.embedding(target_seq)
            output, _ = self.decoder_lstm(embedded, hidden)
            logits = self.fc_out(output)
            return logits
        else:
            input_token = torch.full((B, 1), 1, dtype=torch.long, device=z.device)
            outputs = []
            for t in range(max_length):
                embedded = self.embedding(input_token)
                output, hidden = self.decoder_lstm(embedded, hidden)
                logits = self.fc_out(output)
                outputs.append(logits)

                #   FIXED BUG #7: Sampling with temperature
                logits = logits.squeeze(1) / temperature  # [B, vocab]
                probs = F.softmax(logits, dim=-1)
                input_token = torch.multinomial(probs, 1)  # [B, 1]

            return torch.cat(outputs, dim=1)

    def forward(self, x, lengths, target_seq=None, teacher_forcing_ratio=0.5, temperature=1.0):
        mu, logvar = self.encode(x, lengths)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, target_seq, teacher_forcing_ratio=teacher_forcing_ratio, temperature=temperature)
        return logits, mu, logvar

#
# Step 2.2 — Loss Function (Fixed Bug #4: div by zero)
#

def vae_loss(logits, targets, mu, logvar, pad_token_id, beta=1.0):
    # 1. align lengths
    max_len = max(logits.size(1), targets.size(1))
    if logits.size(1) < max_len:
        logits = F.pad(logits, (0, 0, 0, max_len - logits.size(1)))
    if targets.size(1) < max_len:
        targets = F.pad(targets, (0, max_len - targets.size(1)), value=pad_token_id)

    logits_flat = logits.view(-1, logits.size(-1))          # [B*L, V]
    targets_flat = targets.reshape(-1)                      # [B*L]

    mask = (targets_flat != pad_token_id).float()
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    mask_sum = mask.sum()
    ce_loss = (ce_loss * mask).sum() / (mask_sum + 1e-8)

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return ce_loss + beta * kl_loss, ce_loss, kl_loss

#
# Step 2.3 — KL Annealer (Fixed Bug #5: double increment)
#

class KLAnnealer:
    def __init__(self, total_steps, n_cycle=1, ratio=0.3):
        self.total_steps = total_steps
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.current_step = 0

    def get_beta(self):
        self.current_step += 1  #  Only increment here
        if self.current_step > self.total_steps:
            return 1.0
        fraction = self.current_step / self.total_steps
        if fraction < self.ratio:
            return fraction / self.ratio
        else:
            return 1.0

#
# Step 2.4 — Collate Function (Fixed Bug #2: dynamic pad id)
#

def collate_fn(batch, tokenizer, max_length=128):
    encodings = [tokenizer.encode(s, add_special_tokens=True) for s in batch]
    input_ids = [e['input_ids'] for e in encodings]

    max_len = min(max(len(ids) for ids in input_ids), max_length)
    padded = []
    lengths = []

    pad_token_id = tokenizer.tokenizer.pad_token_id  #   FIXED: dynamic

    for ids in input_ids:
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [pad_token_id] * (max_len - len(ids))
        padded.append(ids)
        lengths.append(min(len(ids), max_length))

    return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

#
# Step 2.5 — Dataset & DataLoader
#

class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]

#
# Step 3.x — Training Loop (Fixed Bug #1: loop over tokenizers)
#

LEARNING_RATE = 5e-6
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
MAX_SEQ_LEN = 128
KL_ANNEAL_RATIO = 0.3

def train_vae(
    model,
    train_loader,
    val_loader,
    optimizer,
    kl_annealer,
    pad_token_id,
    device,
    num_epochs,
    accumulation_steps=4,
    save_dir="./checkpoints",
    tokenizer_name="default"
):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"training_log_{tokenizer_name}.csv")

    with open(log_file, "w") as f:
        f.write("epoch,step,train_loss,train_ce,train_kl,val_loss,val_ce,val_kl,kl_beta\n")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        model.train()
        total_train_loss = total_train_ce = total_train_kl = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for step, (input_ids, lengths) in enumerate(tqdm(train_loader, desc="Training")):
            input_ids, lengths = input_ids.to(device), lengths.to(device)

            #   Pitfall A: Decay teacher forcing
            tfr = max(0.5, 1.0 - (epoch / num_epochs))  # decay from 1.0 → 0.5

            logits, mu, logvar = model(input_ids, lengths, target_seq=input_ids, teacher_forcing_ratio=tfr)
            beta = kl_annealer.get_beta()
            loss, ce_loss, kl_loss = vae_loss(logits, input_ids, mu, logvar, pad_token_id, beta=beta)

            loss = loss / accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * accumulation_steps
            total_train_ce += ce_loss.item()
            total_train_kl += kl_loss.item()
            num_batches += 1

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        total_val_loss = total_val_ce = total_val_kl = 0.0
        val_batches = 0

        with torch.no_grad():
            for input_ids, lengths in tqdm(val_loader, desc="Validating"):
                input_ids, lengths = input_ids.to(device), lengths.to(device)
                #   Pitfall B: Use same beta as training for fair comparison
                beta = kl_annealer.get_beta()  # ← NOT hardcoded 1.0
                logits, mu, logvar = model(input_ids, lengths, target_seq=input_ids, teacher_forcing_ratio=0.0)
                loss, ce_loss, kl_loss = vae_loss(logits, input_ids, mu, logvar, pad_token_id, beta=beta)

                total_val_loss += loss.item()
                total_val_ce += ce_loss.item()
                total_val_kl += kl_loss.item()
                val_batches += 1

        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / val_batches

        current_step = (epoch + 1) * len(train_loader)
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{current_step},{avg_train_loss:.6f},{total_train_ce/num_batches:.6f},{total_train_kl/num_batches:.6f},"
                    f"{avg_val_loss:.6f},{total_val_ce/val_batches:.6f},{total_val_kl/val_batches:.6f},{beta:.6f}\n")

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"KL Beta:    {beta:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(save_dir, f"best_model_{tokenizer_name}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"→ Saved best model to {checkpoint_path}")

    return best_val_loss

#
#   TRAINING LOOP OVER TOKENIZERS (Fixed Bug #1)
#
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

for tokenizer in TOKENIZERS:
    print(f"\n  STARTING TRAINING FOR: {tokenizer.name}\n")

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.tokenizer.pad_token_id

    #   Pitfall C: Validate token IDs
    sample_ids = tokenizer.encode(train_smiles[0], add_special_tokens=True)['input_ids']
    max_id_in_sample = max(sample_ids)
    assert max_id_in_sample < vocab_size, f"Token ID {max_id_in_sample} >= vocab size {vocab_size} in {tokenizer.name}"

    model = MoleculeVAE(vocab_size, pad_token_id=pad_token_id).to(device)
    total_steps = (len(train_smiles) // (BATCH_SIZE * ACCUMULATION_STEPS)) * NUM_EPOCHS
    kl_annealer = KLAnnealer(total_steps, ratio=KL_ANNEAL_RATIO)

    optimizer = Ranger21(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        use_adabelief=True,
        use_warmup=True,
        use_madgrad=True,
        num_epochs=NUM_EPOCHS,
        num_batches_per_epoch=len(train_smiles) // (BATCH_SIZE * ACCUMULATION_STEPS),
        warmdown_active=False,
    )

    # Define optimizer
#    optimizer = AdamW(
#        model.parameters(),
#        lr=LEARNING_RATE,
#        weight_decay=0.01,  # same as before
#    )
#
#    # Define scheduler: cosine annealing over total steps
#    total_steps = NUM_EPOCHS * (len(train_smiles) // (BATCH_SIZE * ACCUMULATION_STEPS))
#    scheduler = CosineAnnealingLR(
#        optimizer,
#        T_max=total_steps,
#        eta_min=1e-6,  # minimum learning rate (optional, prevents going to 0)
#    )
    
    train_dataset = SmilesDataset(train_smiles)
    val_dataset = SmilesDataset(val_smiles)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=MAX_SEQ_LEN),
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, max_length=MAX_SEQ_LEN),
        num_workers=0,
        pin_memory=True
    )

    train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        kl_annealer=kl_annealer,
        pad_token_id=pad_token_id,
        device=device,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        save_dir=f"./checkpoints/{tokenizer.name}",
        tokenizer_name=tokenizer.name
    )

#
# Step 4.x — Evaluation Pipeline (Fixed Bug #6, #7, #8)
#

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def evaluate_reconstruction(model, dataloader, tokenizer, device, max_length=128):
    model.eval()
    total_token_correct = total_tokens = exact_matches = valid_count = total_samples = 0
    all_generated, all_targets = [], []

    pad_id = tokenizer.tokenizer.pad_token_id
    eos_id = tokenizer.tokenizer.eos_token_id
    special_ids = {pad_id, eos_id}

    def trim_to_special(ids, specials):
        for i, id_ in enumerate(ids):
            if id_ in specials:
                return ids[:i]
        return ids

    with torch.no_grad():
        for input_ids, lengths in tqdm(dataloader, desc="Evaluating Reconstruction"):
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            B = input_ids.size(0)

            mu, logvar = model.encode(input_ids, lengths)
            z = model.reparameterize(mu, logvar)
            logits = model.decode(z, max_length=max_length, temperature=0.8)  #   FIXED #7
            preds = logits.argmax(dim=-1)

            #   FIXED: Align logits and targets to same sequence length
            min_len = min(logits.size(1), input_ids.size(1))
            preds = preds[:, :min_len]          # trim predictions
            input_ids = input_ids[:, :min_len]  # trim targets

            mask = (input_ids != pad_id)
            token_correct = ((preds == input_ids) & mask).sum().item()
            total_token_correct += token_correct
            total_tokens += mask.sum().item()

            for i in range(B):
                target_ids = input_ids[i].cpu().tolist()
                pred_ids = preds[i].cpu().tolist()

                #   FIXED BUG #6: Trim before decode
                target_ids_trim = trim_to_special(target_ids, special_ids)
                pred_ids_trim = trim_to_special(pred_ids, special_ids)

                target_smiles = tokenizer.decode(target_ids_trim, skip_special_tokens=False)
                pred_smiles = tokenizer.decode(pred_ids_trim, skip_special_tokens=False)

                all_targets.append(target_smiles)
                all_generated.append(pred_smiles)

                if pred_smiles == target_smiles:
                    exact_matches += 1
                if Chem.MolFromSmiles(pred_smiles) is not None:
                    valid_count += 1
                total_samples += 1

    token_acc = total_token_correct / total_tokens if total_tokens > 0 else 0.0
    exact_match_rate = exact_matches / total_samples
    validity_rate = valid_count / total_samples

    print(f"Token-level Accuracy: {token_acc:.4f}")
    print(f"Exact Match Rate:     {exact_match_rate:.4f}")
    print(f"Validity Rate:        {validity_rate:.4f}")

    return {
        'token_accuracy': token_acc,
        'exact_match_rate': exact_match_rate,
        'validity_rate': validity_rate,
        'generated_smiles': all_generated,
        'target_smiles': all_targets
    }

def compute_uniqueness_and_novelty(generated_smiles, train_smiles_set):
    total = len(generated_smiles)
    unique = len(set(generated_smiles))
    novel = len([s for s in generated_smiles if s not in train_smiles_set])
    uniqueness = unique / total if total > 0 else 0.0
    novelty = novel / total if total > 0 else 0.0
    print(f"Uniqueness: {uniqueness:.4f} ({unique}/{total})")
    print(f"Novelty:    {novelty:.4f} ({novel}/not in train)")
    return uniqueness, novelty

def kl_divergence_from_samples(samples, bins=512):
    dim_kls = []
    for d in range(samples.shape[1]):
        data = samples[:, d]
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        norm_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * bin_centers**2)
        hist = np.clip(hist, 1e-10, None)
        norm_pdf = np.clip(norm_pdf, 1e-10, None)
        kl = entropy(hist, norm_pdf)
        dim_kls.append(kl)
    return np.mean(dim_kls)

def evaluate_latent_kl(model, dataloader, device, latent_dim=128, bins=512):
    model.eval()
    all_z = []
    with torch.no_grad():
        for input_ids, lengths in tqdm(dataloader, desc="Sampling Latents"):
            input_ids, lengths = input_ids.to(device), lengths.to(device)
            mu, logvar = model.encode(input_ids, lengths)
            z = model.reparameterize(mu, logvar)
            all_z.append(z.cpu().numpy())
    all_z = np.concatenate(all_z, axis=0)
    kl_div = kl_divergence_from_samples(all_z, bins=bins)
    print(f"KL Divergence (empirical vs N(0,1)): {kl_div:.4f}")
    return kl_div

def evaluate_interpolation_validity(model, tokenizer, test_smiles, device, num_pairs=100, steps=10, max_length=128):
    model.eval()
    pairs = random.sample(list(zip(test_smiles[::2], test_smiles[1::2])), min(num_pairs, len(test_smiles)//2))
    valid_interps = total_interps = 0

    with torch.no_grad():
        for smiles_a, smiles_b in tqdm(pairs, desc="Interpolation Validity"):
            if not smiles_a or not smiles_b: continue

            enc_a = tokenizer.encode(smiles_a, add_special_tokens=True)
            enc_b = tokenizer.encode(smiles_b, add_special_tokens=True)

            ids_a = torch.tensor([enc_a['input_ids']], device=device)
            ids_b = torch.tensor([enc_b['input_ids']], device=device)
            len_a = torch.tensor([len(enc_a['input_ids'])], device=device)
            len_b = torch.tensor([len(enc_b['input_ids'])], device=device)

            mu_a, _ = model.encode(ids_a, len_a)
            mu_b, _ = model.encode(ids_b, len_b)

            alphas = torch.linspace(0, 1, steps, device=device)
            for alpha in alphas:
                z_interp = alpha * mu_b + (1 - alpha) * mu_a
                logits = model.decode(z_interp, max_length=max_length, temperature=0.8)
                preds = logits.argmax(dim=-1).squeeze(0)
                pred_smiles = tokenizer.decode(preds.cpu().tolist(), skip_special_tokens=True)
                if Chem.MolFromSmiles(pred_smiles) is not None:
                    valid_interps += 1
                total_interps += 1

    interp_validity = valid_interps / total_interps if total_interps > 0 else 0.0
    print(f"Interpolation Validity: {interp_validity:.4f}")
    return interp_validity

def sample_from_latent(model, tokenizer, num_samples=30000, latent_dim=128, max_length=128, device=device, temperature=0.8):
    model.eval()
    generated_smiles = []
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, BATCH_SIZE), desc="Sampling from Latent"):
            current_batch_size = min(BATCH_SIZE, num_samples - len(generated_smiles))
            if current_batch_size <= 0: break
            z = torch.randn(current_batch_size, latent_dim, device=device)
            logits = model.decode(z, max_length=max_length, temperature=temperature)
            probs = F.softmax(logits / temperature, dim=-1)
            preds = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            for i in range(current_batch_size):
                pred_ids = preds[i].cpu().tolist()
                smiles = tokenizer.decode(pred_ids, skip_special_tokens=True)
                generated_smiles.append(smiles)
                if len(generated_smiles) >= num_samples: break
    return generated_smiles

def measure_inference_throughput(model, tokenizer, test_smiles, device,
                                 max_length=128,
                                 batch_sizes=[1, 4, 8, 16]):
    """
    Benchmark inference speed & peak GPU memory across several batch sizes.
    Returns a JSON-serialisable dict:
        {batch_size: {'tokens_per_sec': <float>, 'peak_mem_mb': <float>}, ...}
    """
    model.eval()
    results = {}

    for bs in batch_sizes:
        # Build a small fixed subset so every BS processes the same #samples
        subset = SmilesDataset(test_smiles[:bs * 10])
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length=max_length),
        )

        total_tokens = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            for input_ids, lengths in loader:
                input_ids, lengths = input_ids.to(device), lengths.to(device)
                mu, logvar = model.encode(input_ids, lengths)
                z = model.reparameterize(mu, logvar)
                logits = model.decode(z, max_length=max_length)
                total_tokens += logits.numel()  # number of float elements
        duration = time.perf_counter() - start_time

        tokens_per_sec = total_tokens / duration
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )

        # Store as plain Python floats
        results[bs] = {
            "tokens_per_sec": float(tokens_per_sec),
            "peak_mem_mb": float(peak_mem_mb),
        }
        print(f"BS {bs:3d} → {tokens_per_sec:8.2f} tok/s | Peak Mem: {peak_mem_mb:.2f} MB")

    return results

#
# FCD — Uncomment if fcd is installed
#
# from fcd import get_fcd
# def compute_fcd(true_smiles, gen_smiles, n_jobs=1):
#     try:
#         fcd_score = get_fcd(true_smiles, gen_smiles, n_jobs=n_jobs)
#         print(f"FCD: {fcd_score:.4f}")
#         return fcd_score
#     except Exception as e:
#         print(f"FCD failed: {e}")
#         return None

#
#   FINAL EVALUATION PIPELINE
#

def full_evaluation_pipeline(model, tokenizer, train_smiles, test_smiles, device, save_dir):
    print(f"\n  FULL EVALUATION FOR: {tokenizer.name}")

    test_dataset = SmilesDataset(test_smiles)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=MAX_SEQ_LEN),
        num_workers=0)

    # 1. Reconstruction
    recon_metrics = evaluate_reconstruction(model, test_loader, tokenizer, device)

    # 2. Uniqueness & Novelty
    train_set = set(train_smiles)
    uniqueness, novelty = compute_uniqueness_and_novelty(recon_metrics['generated_smiles'], train_set)

    # 3. KL Divergence
    kl_div = evaluate_latent_kl(model, test_loader, device)

    # 4. Interpolation Validity
    interp_validity = evaluate_interpolation_validity(model, tokenizer, test_smiles, device)

    # 5. Latent Sampling (for FCD — optional)
    # gen_smiles_30k = sample_from_latent(model, tokenizer, num_samples=10000, temperature=0.8)  # reduce for speed
    # fcd_score = compute_fcd(test_smiles, gen_smiles_30k) if 'get_fcd' in globals() else None

    # 6. Throughput & Memory
    # throughput = measure_inference_throughput(model, tokenizer, test_loader, device)

    eval_results = {
        **recon_metrics,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'kl_divergence': kl_div,
        'interpolation_validity': interp_validity,
        # 'fcd': fcd_score,
        # 'inference_throughput': throughput,
    }

    import json
    eval_path = os.path.join(save_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    print(f" Evaluation saved to {eval_path}")
    return eval_results

#
#   RUN EVALUATION FOR EACH TOKENIZER
#

for tokenizer in TOKENIZERS:
    print(f"\n🔍 LOADING BEST MODEL FOR: {tokenizer.name}")
    checkpoint_path = f"./checkpoints/{tokenizer.name}/best_model_{tokenizer.name}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        continue

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.tokenizer.pad_token_id
    model = MoleculeVAE(vocab_size, pad_token_id=pad_token_id).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    full_evaluation_pipeline(
        model=model,
        tokenizer=tokenizer,
        train_smiles=train_smiles,
        test_smiles=test_smiles,
        device=device,
        save_dir=f"./checkpoints/{tokenizer.name}"
    )


print("\n🎉 PIPELINE COMPLETE — ALL TOKENIZERS BENCHMARKED, TRAINED, AND EVALUATED!")

