#!/usr/bin/env python3
"""
Latent Space Visualization for Molecule VAE Models
Integrated with existing benchmark pipeline structure
"""

import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import from existing benchmark code
from transformers import AutoTokenizer
try:
    from FastChemTokenizer import FastChemTokenizer
except ImportError:
    print("FastChemTokenizer not found. Please ensure it's in your PYTHONPATH.")
    FastChemTokenizer = None

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
    
    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id

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


class SmilesDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles_list = smiles_list
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        return self.smiles_list[idx]



class MoleculeVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=128, num_layers=2,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

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

    def decode(self, z, max_length=128, mode="greedy", temperature=1.0):
        """
        Decode latent vector z into a sequence.
        Returns full logits at each step.
        PATCHED: stops generation when EOS is predicted.
        """
        batch_size = z.size(0)
        device = z.device

        # Initialize hidden states from latent
        h0 = self.latent2hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.latent2cell(z).view(self.num_layers, batch_size, self.hidden_dim)
        hidden = (h0, c0)

        # Start with BOS token — shape: (batch_size, 1)
        input_token = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        logits = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # ← TRACK FINISHED SEQS

        for _ in range(max_length):
            embedded = self.embedding(input_token)  # (batch, 1, embed_dim)
            output, hidden = self.decoder_lstm(embedded, hidden)
            logit = self.fc_out(output)  # (batch, 1, vocab)
            logits.append(logit)

            if mode == "greedy":
                input_token = logit.argmax(dim=-1)  # (batch, 1)
            elif mode == "sample":
                probs = torch.softmax(logit.squeeze(1) / temperature, dim=-1)  # (batch, vocab)
                input_token = torch.multinomial(probs, 1)  # (batch, 1)
            else:
                raise ValueError(f"Unknown decode mode: {mode}")

            # ← EARLY STOPPING AT EOS
            just_finished = (input_token.squeeze(1) == self.eos_token_id)
            finished |= just_finished
            input_token[finished] = self.pad_token_id  # pad finished sequences
            if finished.all():
                break

        return torch.cat(logits, dim=1)  # (batch, seq_len, vocab)

    def forward(self, input_ids, lengths, target_seq=None, teacher_forcing_ratio=0.0, temperature=1.0):
        mu, logvar = self.encode(input_ids, lengths)
        z = self.reparameterize(mu, logvar)

        if self.training and target_seq is not None and teacher_forcing_ratio > 0:
            # Training with teacher forcing
            batch_size, seq_len = target_seq.size()
            device = target_seq.device
            
            # Initialize hidden states
            h0 = self.latent2hidden(z).view(self.num_layers, batch_size, self.hidden_dim)
            c0 = self.latent2cell(z).view(self.num_layers, batch_size, self.hidden_dim)
            hidden = (h0, c0)
            
            logits = []
            input_token = target_seq[:, 0].unsqueeze(1)  # BOS

            for t in range(1, seq_len):
                embedded = self.embedding(input_token)
                output, hidden = self.decoder_lstm(embedded, hidden)
                logit = self.fc_out(output)
                logits.append(logit)

                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher:
                    input_token = target_seq[:, t].unsqueeze(1)
                else:
                    input_token = logit.argmax(dim=-1)

            logits = torch.cat(logits, dim=1)
        else:
            # Inference mode
            max_len = target_seq.size(1) if target_seq is not None else 128
            logits = self.decode(z, max_length=max_len, mode="greedy", temperature=temperature)

        return logits, mu, logvar

class LatentSpaceVisualizer:
    def __init__(self, model_path, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained VAE model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Get vocab size from tokenizer
        vocab_size = len(self.tokenizer)
        pad_token_id = self.tokenizer.tokenizer.pad_token_id
        
        # Initialize model with correct parameters
        model = MoleculeVAE(vocab_size=vocab_size, pad_token_id=pad_token_id)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        return model
    
    def encode_molecules(self, smiles_list, batch_size=32):
        """Encode molecules to latent space"""
        dataset = SmilesDataset(smiles_list)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer, max_length=128)
        )
        
        all_mus = []
        with torch.no_grad():
            for input_ids, lengths in tqdm(dataloader, desc="Encoding molecules"):
                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                
                mu, logvar = self.model.encode(input_ids, lengths)
                all_mus.append(mu.cpu().numpy())
        
        return np.concatenate(all_mus, axis=0)
    
    def create_grid_latent_points(self, grid_size=100, z_range=4):
        """Create a grid of points in 2D latent space"""
        x = np.linspace(-z_range, z_range, grid_size)
        y = np.linspace(-z_range, z_range, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Create circular mask
        center = grid_size // 2
        radius = grid_size // 2
        y_coords, x_coords = np.ogrid[:grid_size, :grid_size]
        mask = (x_coords - center) ** 2 + (y_coords - center) ** 2 <= radius ** 2
        
        return xx, yy, mask
    
    def classify_latent_points(self, latent_points, dim1=0, dim2=1, additional_dim=None):
        """
        Classify latent points by chemical validity (RDKit parseable)
        Returns: 0 for invalid/unparseable molecules, 1 for valid molecules
        """
        classifications = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(latent_points), batch_size):
                batch_points = latent_points[i:i+batch_size]
                
                # Create full latent vectors (sample from normal for other dimensions)
                full_z = torch.randn(len(batch_points), self.model.latent_dim, device=self.device) * 0.1
                full_z[:, dim1] = torch.FloatTensor(batch_points[:, 0]).to(self.device)
                full_z[:, dim2] = torch.FloatTensor(batch_points[:, 1]).to(self.device)
                
                # If additional dimension specified (for z2 plots)
                if additional_dim is not None:
                    if isinstance(additional_dim, dict):
                        for dim_idx, dim_val in additional_dim.items():
                            full_z[:, dim_idx] = dim_val
                
                try:
                    # Decode to SMILES
                    logits = self.model.decode(full_z, max_length=64, temperature=0.8)
                    predictions = torch.argmax(logits, dim=-1)
                    
                    # Check chemical validity for each decoded molecule
                    batch_classes = []
                    for pred in predictions:
                        pred_ids = pred.cpu().tolist()
                        
                        # Remove padding and special tokens
                        pad_id = self.tokenizer.tokenizer.pad_token_id
                        eos_id = self.tokenizer.tokenizer.eos_token_id
                        
                        # Trim at EOS or pad
                        for j, token_id in enumerate(pred_ids):
                            if token_id in [pad_id, eos_id]:
                                pred_ids = pred_ids[:j]
                                break
                        
                        try:
                            decoded_smiles = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                            # Test chemical validity with RDKit
                            mol = Chem.MolFromSmiles(decoded_smiles)
                            
                            if mol is None:
                                # Invalid/unparseable molecule
                                batch_classes.append(0)
                            else:
                                # Valid, RDKit-parseable molecule
                                batch_classes.append(1)
                                
                        except Exception:
                            # Decoding or parsing failed - invalid
                            batch_classes.append(0)
                    
                    classifications.extend(batch_classes)
                    
                except Exception as e:
                    # If decoding fails, all points in batch are invalid
                    classifications.extend([0] * len(batch_points))
        
        return np.array(classifications)
    
    def plot_latent_space_interpolation(self, grid_size=100, z_range=4, save_path=None):
        """
        Create latent space interpolation plots similar to the reference images
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Create color map (RED for invalid molecules, GREEN for valid molecules)
        colors = ['#FF4444', '#44AA44']  # Red (invalid) and Green (valid)
        cmap = ListedColormap(colors)
        
        plot_idx = 0
        
        # First row: different dimension pairs
        dimension_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        for dim_pair in dimension_pairs:
            dim1, dim2 = dim_pair
            
            # Create grid
            xx, yy, mask = self.create_grid_latent_points(grid_size, z_range)
            
            # Get points within circular boundary
            valid_points = []
            valid_coords = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if mask[i, j]:
                        valid_points.append([xx[i, j], yy[i, j]])
                        valid_coords.append([i, j])
            
            valid_points = np.array(valid_points)
            
            # Classify points based on chemical validity
            print(f"Classifying latent space chemical validity for dimensions {dim1}, {dim2}...")
            classifications = self.classify_latent_points(valid_points, dim1, dim2)
            
            # Create classification grid
            class_grid = np.zeros((grid_size, grid_size))
            class_grid.fill(np.nan)  # Fill with NaN for areas outside circle
            
            for point_idx, (i, j) in enumerate(valid_coords):
                class_grid[i, j] = classifications[point_idx]
            
            # Plot
            ax = axes[plot_idx]
            im = ax.imshow(class_grid, extent=[-z_range, z_range, -z_range, z_range], 
                          origin='lower', cmap=cmap, alpha=0.8, vmin=0, vmax=1)
            
            # Add concentric circles
            circles = [1, 2, 3, 4]
            for radius in circles:
                if radius <= z_range:
                    circle = plt.Circle((0, 0), radius, fill=False, color='black', 
                                      alpha=0.3, linewidth=0.5)
                    ax.add_patch(circle)
            
            # Set labels and title
            ax.set_xlabel(f'Latent dimension z{dim1}')
            ax.set_ylabel(f'Latent dimension z{dim2}')
            ax.set_title('SMILES')
            ax.set_xlim(-z_range, z_range)
            ax.set_ylim(-z_range, z_range)
            ax.set_aspect('equal')
            
            plot_idx += 1
        
        # Second row: fix z0, z1 and vary z2
        for z2_val in [-2, -1, 1, 2]:
            dim1, dim2 = 0, 1  # Use z0 and z1 for x,y
            
            # Create grid
            xx, yy, mask = self.create_grid_latent_points(grid_size, z_range)
            
            # Get points within circular boundary
            valid_points = []
            valid_coords = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if mask[i, j]:
                        valid_points.append([xx[i, j], yy[i, j]])
                        valid_coords.append([i, j])
            
            valid_points = np.array(valid_points)
            
            # Classify points with z2 fixed - check chemical validity
            print(f"Classifying latent space chemical validity for z0, z1 with z2 = {z2_val}...")
            classifications = self.classify_latent_points(
                valid_points, dim1, dim2, 
                additional_dim={2: z2_val}
            )
            
            # Create classification grid
            class_grid = np.zeros((grid_size, grid_size))
            class_grid.fill(np.nan)
            
            for point_idx, (i, j) in enumerate(valid_coords):
                class_grid[i, j] = classifications[point_idx]
            
            # Plot
            ax = axes[plot_idx]
            im = ax.imshow(class_grid, extent=[-z_range, z_range, -z_range, z_range], 
                          origin='lower', cmap=cmap, alpha=0.8, vmin=0, vmax=1)
            
            # Add concentric circles
            for radius in circles:
                if radius <= z_range:
                    circle = plt.Circle((0, 0), radius, fill=False, color='black', 
                                      alpha=0.3, linewidth=0.5)
                    ax.add_patch(circle)
            
            ax.set_xlabel('Latent dimension z0')
            ax.set_ylabel('Latent dimension z1') 
            ax.set_title(f'SMILES; z2 = {z2_val}')
            ax.set_xlim(-z_range, z_range)
            ax.set_ylim(-z_range, z_range)
            ax.set_aspect('equal')
            
            plot_idx += 1
        
        plt.suptitle(f'Latent Space Chemical Validity - {self.tokenizer.name}\n(Red: Invalid molecules, Green: Valid molecules)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_molecule_embeddings(self, smiles_list, method='tsne', save_path=None):
        """Plot actual molecule embeddings in 2D using dimensionality reduction"""
        print(f"Encoding {len(smiles_list)} molecules...")
        embeddings = self.encode_molecules(smiles_list)
        
        # Create simple labels based on molecular properties
        labels = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                labels.append(0)
            else:
                # Simple binary classification
                mw = Chem.Descriptors.MolWt(mol)
                labels.append(1 if mw > 200 else 0)
        
        labels = np.array(labels)
        
        # Reduce dimensionality
        print(f"Computing {method.upper()} projection...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(smiles_list)//4))
        else:
            reducer = PCA(n_components=2, random_state=42)
            
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='RdYlGn', alpha=0.7, s=20)
        plt.colorbar(scatter, label='Molecular Weight > 200')
        plt.title(f'{method.upper()} of Molecule Embeddings - {self.tokenizer.name}')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def load_data_and_tokenizers():
    """Load data and tokenizers using your existing structure"""
    # Load SMILES data (adjust path as needed)
    data_path = "../data/sample_all_8k_smi.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please update the data_path in the script.")
        return None, None
    
    df = pd.read_csv(data_path)
    if 'SMILES' not in df.columns:
        raise ValueError("Expected column 'SMILES' in CSV")

    smiles_list = df['SMILES'].dropna().tolist()
    
    # Validate SMILES
    valid_smiles = []
    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is not None:
            valid_smiles.append(smiles)
    
    print(f"Loaded {len(valid_smiles)} valid SMILES")
    
    # Initialize tokenizers
    try:
        tok1_hf = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        tokenizer1 = TokenizerWrapper(tok1_hf, name="ChemBERTa", 
                                    bos_token="<s>", eos_token="</s>", 
                                    pad_token="<pad>", unk_token="<unk>")
    except Exception as e:
        print(f"Failed to load ChemBERTa tokenizer: {e}")
        tokenizer1 = None
    
    try:
        tok2_fast = FastChemTokenizer.from_pretrained("../smitok")
        tokenizer2 = TokenizerWrapper(tok2_fast, name="FastChemTokenizer", 
                                    bos_token="[BOS]", eos_token="[EOS]", 
                                    pad_token="[PAD]", unk_token="[UNK]")
    except Exception as e:
        print(f"Failed to load FastChemTokenizer: {e}")
        tokenizer2 = None
    
    tokenizers = [t for t in [tokenizer1, tokenizer2] if t is not None]
    
    return valid_smiles, tokenizers

def create_latent_visualizations():
    """Main function to create latent space visualizations"""
    
    # Load data and tokenizers
    smiles_list, tokenizers = load_data_and_tokenizers()
    if smiles_list is None or not tokenizers:
        print("Failed to load data or tokenizers. Please check your setup.")
        return
    
    # Use a subset for faster visualization
    viz_smiles = smiles_list[:1000]  # Adjust size as needed
    
    # Model paths
    model_paths = {
        'ChemBERTa': './checkpoints/ChemBERTa/best_model_ChemBERTa.pt',
        'FastChemTokenizer': './checkpoints/FastChemTokenizer/best_model_FastChemTokenizer.pt'
    }
    
    # Create output directory
    os.makedirs('latent_space_plots', exist_ok=True)
    
    for tokenizer in tokenizers:
        model_path = model_paths.get(tokenizer.name)
        if model_path is None or not os.path.exists(model_path):
            print(f"Model not found for {tokenizer.name}: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Creating visualizations for {tokenizer.name}")
        print(f"{'='*60}")
        
        try:
            # Create visualizer
            visualizer = LatentSpaceVisualizer(model_path, tokenizer, device)
            
            # Create latent space interpolation plots
            print("Creating latent space interpolation plots...")
            save_path = f'latent_space_plots/{tokenizer.name}_latent_interpolation.png'
            visualizer.plot_latent_space_interpolation(save_path=save_path)
            
            # Create molecule embedding plots
            print("Creating t-SNE embedding plot...")
            save_path = f'latent_space_plots/{tokenizer.name}_embeddings_tsne.png'
            visualizer.plot_molecule_embeddings(viz_smiles, method='tsne', save_path=save_path)
            
            print("Creating PCA embedding plot...")
            save_path = f'latent_space_plots/{tokenizer.name}_embeddings_pca.png'
            visualizer.plot_molecule_embeddings(viz_smiles, method='pca', save_path=save_path)
            
        except Exception as e:
            print(f"Error processing {tokenizer.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Visualization complete! Check the 'latent_space_plots' directory for results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Import RDKit descriptors for molecular property calculation
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        print("RDKit Descriptors not available. Using simpler classification.")
        # Fallback to simple classification if descriptors not available
        Descriptors = None
        rdMolDescriptors = None
    
    create_latent_visualizations()
