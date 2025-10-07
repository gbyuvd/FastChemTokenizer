
# 🧪 FastChemTokenizer — A High-Performance SMILES Tokenizer built via Info-Theoretic Motif Mining

> **Optimized for chemical language modeling. 2x faster, 50% shorter sequences, minimal memory. Built with entropy-guided n-gram selection.**


## 🚀 Overview

`FastChemTokenizer` is a **trie-based, longest-match-first tokenizer** specifically designed for efficient tokenization of **SMILES and SELFIES strings** in molecular language modeling. The tokenizer is built from scratch for speed and compactness, it outperforms popular tokenizers like [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/)'s while maintaining 0% UNK rate on ~2.7M dataset and compatibility with Hugging Face `transformers`. In n-grams building, this project uses [seyonec/ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/)'s as early tokenizer for determining n-grams using its token_ids, then uses information-theoretic filtering (entropy reduction, PMI, internal entropy) to extract meaningful statistical chemical motifs — then balances 391 backbone (functional) and 391 tail fragments for structural coverage.

Trained on ~2.7M valid SMILES and SELFIES built and curated from ChemBL34 (Zdrazil _et al._ 2023), COCONUTDB (Sorokina _et al._ 2021), and Supernatural3 (Gallo _et al._ 2023) dataset; from resulting 76K n-grams -> pruned to **1,238 tokens**, including backbone/tail motifs and special tokens.

The "comb_smi.csv" dataset can be downloaded [here](https://huggingface.co/datasets/gbyuvd/bioactives-naturals-smiles-molgen).

A tentative technical report can be read [here](https://amachinewithorgans.wordpress.com/2025/09/27/fastchemtokenizer-a-new-approach-to-chemical-language-processing-via-statistical-info-theoretic-motif-mining/).

## ⚡ Performance Highlights

#### SMILES
| Metric                          | FastChemTokenizer | [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/) Tokenizer | [gen-mlm-cismi-bert](https://huggingface.co/smostafanejad/gen-mlm-cismi-bert-wordpiece) |
|--------------------------------|-------------------|----------------------|---------------------|
| **Avg time per SMILES**        | **0.0692 ± 0.0038 ms**  | 0.1279 ± 0.0090 ms   | 0.1029 ± 0.0038  ms |
| **Avg sequence length**        | **21.61 ± 0.70 tokens**| 42.23 ± 1.55 tokens  | 50.86 ± 1.90 tokens |
| **Throughput**                 | **14,448/sec**    | 7,817/sec            | 9,720/sec           |
| **Peak memory usage**          | **12.92 MB**      | 258.00 MB            | 387.73 MB           |
| **UNK token rate**             | **0.0000%**       | 0.0000%              | ~0.0000% (non-zero) |
| **1000 encodes (benchmark)**   | **0.0029s**       | 1.6598s              | 0.5491s             |

✅ **1.97x faster** than ChemBERTa  
✅ **1.50x faster** than gen-mlm-cismi-bert  
✅ **~19x memory saving** compared to both of the above tokenizer  
✅ **No indexing errors** (avoids >512 token sequences)  
✅ **Zero unknown tokens** on validation set

_Note: Syntax-aware SMILES tokenizer is still in development and hasn't properly optimized and evaluated yet_

#### SELFIES
```
Core's vocab length = 781 (after pruning) 
        with tails = 1161 (after pruning) 
```
| Metric                         | FastChemTokenizer-WTails | FastChemTokenizer-Core | [opti-chemfie-experiment-1](https://huggingface.co/gbyuvd/bionat-selfies-gen-tokenizer-wordlevel) |
|--------------------------------|-------------------|----------------------|---------------------|
| **Avg time per SMILES**        | 0.1882 ± 0.0140 ms| 0.1674 ± 0.0093 ms   | **0.1157 ± 0.0095 ms**|
| **Avg sequence length**        | **20.46 ± 1.21 tokens**  | 33.41  ± 1.80 tokens | 54.29 ± 3.08 tokens |
| **Throughput**                 | 5,313/sec         | 5,973/sec            | **8,642 /sec**      |
| **Peak memory usage**          | **9.32 MB**       | 20.16 MB             | 490.13 MB           |
| **UNK token rate**             | **0.0000%**       | 0.0000%              | 0.0000%             |
| **1000 encodes (benchmark)**   | **0.0081s**       | 2.9020s              | 2.9020s             |

✅ Even though 1.32x slower, it produces **2.65x less tokens**   
        - this slow down could be related with searching based on a lot of whitespaces between the formated SELFIES strings
✅ **~61x memory saving with tails** and **~25x** with core

## 🧩 Vocabulary (SMILES)

- **Final vocab size**: 1,238 tokens
- **Includes**: 391 backbone motifs + 391 tail motifs + special tokens (`<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`)
- **Pruned**: 270 unused tokens (e.g., `'²'`, `'C@@H](O)['`, `'È'`)
- **Training corpus**: ~119M unigrams from ~3M SMILES sequences
- **Entropy-based filtering**: Internal entropy > 0.5, entropy reduction < 0.95


## 🛠️ Implementation

- **Algorithm**: Trie-based longest-prefix-match 
- **Caching**: `@lru_cache` for repeated string encoding
- **HF Compatible**: Implements `__call__`, `encode_plus`, `batch_encode_plus`, `save_pretrained`, `from_pretrained`
- **Memory Efficient**: Trie traversal and cache

**for SMILES (core backbone vocabs without tails)** 

for with tails, use `./smitok` 

if you want to use HF compat tokenizer (still in devel), please use `FastChemTokenizerHF` 

_Note: Syntax-aware SMILES tokenizer is still in development and hasn't properly optimized and evaluated yet_

**Syntax-aware but slower method:**

```python
from FastChemTokenizerHF import FastChemTokenizerSmiles

tokenizer = FastChemTokenizerSmiles.from_pretrained("./smitok_syntax")

test_smiles = "O=C1N=C(O)C(Cl)=C1Cl"
test_ids = tokenizer.encode(test_smiles)
decoded = tokenizer.decode(test_ids)
print(f"🧪 Test encode/decode: '{test_smiles}' → {test_ids} → '{decoded}'")
encoded = tokenizer.encode(test_smiles)
decoded = tokenizer.decode_with_trace(encoded)
print(decoded)

# ✅ Special tokens bound: 0 1 2 3 4
# 🧪 Test encode/decode: 'O=C1N=C(O)C(Cl)=C1Cl' → [0, 114, 183, 204, 195, 205, 197, 204, 208, 205, 183, 200, 208, 1] → '<s>O=C1N=C(O)C(Cl)=C1Cl</s>'
# 
# 🔍 Decoding 14 tokens:
#   [000] ID=    0 → '<s>'
#   [001] ID=  114 → 'O=C1N'
#   [002] ID=  183 → '=C'
#   [003] ID=  204 → '('
#   [004] ID=  195 → 'O'
#   [005] ID=  205 → ')'
#   [006] ID=  197 → 'C'
#   [007] ID=  204 → '('
#   [008] ID=  208 → 'Cl'
#   [009] ID=  205 → ')'
#   [010] ID=  183 → '=C'
#   [011] ID=  200 → '1'
#   [012] ID=  208 → 'Cl'
#   [013] ID=    1 → '</s>'
```

**Non-syntax-aware but faster method:**

```python
from FastChemTokenizer import FastChemTokenizer

tokenizer = FastChemTokenizer.from_pretrained("../smitok_core")
benzene = "c1ccccc1"
encoded = tokenizer.encode(benzene)
print("✅ Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("✅ Decoded:", decoded)
tokenizer.decode_with_trace(encoded)

# ✅ Encoded: [271, 474, 840]
# ✅ Decoded: c1ccccc1
# 
# 🔍 Decoding 3 tokens:
#   [000] ID=  271 → 'c1ccc'
#   [001] ID=  474 → 'cc'
#   [002] ID=  840 → '1'

```

**for SELFIES**

Please don't use the old `FastChemTokenizer` for SELFIES, use the HF one

```python
from FastChemTokenizerHF import FastChemTokenizerSelfies

tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core") # change to *_core for w/o tails
benzene = "[C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1]" # please make sure whitespaced input
encoded = tokenizer.encode(benzene)
print("✅ Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("✅ Decoded:", decoded)
tokenizer.decode_with_trace(encoded)

# ✅ Encoded: [0, 257, 640, 693, 402, 1]
# ✅ Decoded: <s> [C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1] </s>

# 🔍 Decoding 6 tokens:
#  [000] ID=    0 → '<s>'
#  [001] ID=  257 → '[C] [=C] [C] [=C] [C]'
#  [002] ID=  640 → '[=C]'
#  [003] ID=  693 → '[Ring1]'
#  [004] ID=  402 → '[=Branch1]'
#  [005] ID=    1 → '</s>'
```

#### BigSMILES (experimental)
```python
from FastChemTokenizer import FastChemTokenizer

tokenizer = FastChemTokenizer.from_pretrained("./bigsmiles-proto") 
testentry = "*CC(*)c1ccccc1C(=O)OCCCCCC"
encoded = tokenizer.encode(testentry)
print("✅ Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("✅ Decoded:", decoded)
tokenizer.decode_with_trace(encoded)

# ✅ Encoded: [186, 185, 723, 31, 439]
# ✅ Decoded: *CC(*)c1ccccc1C(=O)OCCCCCC
# 
# 🔍 Decoding 5 tokens:
#   [000] ID=  186 → '*CC(*)'
#   [001] ID=  185 → 'c1cccc'
#   [002] ID=  723 → 'c1'
#   [003] ID=   31 → 'C(=O)OCC'
#   [004] ID=  439 → 'CCCC'
```

## 📦 Installation & Usage

0. Make sure you have all the reqs packages, possibly can be run with different versions
1. Clone this repository to a directory
2. Load with:
```python
from FastChemTokenizer import FastChemTokenizer

tokenizer = FastChemTokenizer.from_pretrained("./smitok_core")
```
3. Use like any Hugging Face tokenizer:
```python
outputs = tokenizer.batch_encode_plus(smiles_list, padding=True, truncation=True, max_length=512)
```

## 📚 Models using this tokenizer:
- [ChemMiniQ3-HoriFIE](https://github.com/gbyuvd/ChemMiniQ3-HoriFIE)


## 📚 Early VAE Evaluation (vs. ChemBERTa's) [WIP for Scaling]
Using `benchmark_simpler.py`: 1st Epoch, on ~13K samples of len(token_ids)<=25; embed_dim=64, hidden_dim=128, latent_dim=64, num_layers=2; batch_size= 16 * 4 (grad acc) 

Latent Space Visualization based on SMILES Interpolation Validity   

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/sfzBvmJR-ovjpe5F7vNR4.png)

using smitok (with tails)

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/-TusjDSYv9J3K-pfb0hqu.png)

```text
Train: 13017
Val:   1627
Test:  1628

=== Benchmarking ChemBERTa ===
vocab_size                         : 767
avg_tokens_per_mol                 : 25.0359
compression_ratio                  : 1.3766
percent_unknown                    : 0.0000
encode_throughput_smiles_per_sec   : 4585.2022
decode_throughput_smiles_per_sec   : 18168.2779
decode_reconstruction_accuracy     : 100.0000

=== Benchmarking FastChemTokenizerHF ===
vocab_size                         : 1238
avg_tokens_per_mol                 : 13.5668
compression_ratio                  : 2.5403
percent_unknown                    : 0.0000
encode_throughput_smiles_per_sec   : 32005.8686
decode_throughput_smiles_per_sec   : 29807.3610
decode_reconstruction_accuracy     : 100.0000
```

## 🔧 Contributing

This project is an ongoing **experiment** — all contributions are welcome!

- 🧠 Have a better way to implement the methods?
- 📊 Want to add evaluation metrics?
- ✨ Found a bug? Please open an issue!

👉 Please:
- Keep changes minimal and focused.
- Add comments if you change core logic.

## ⚠️ Disclaimer

> **This is NOT a production ready tokenizer.**  
>  
> - Built during late-night prototyping sessions 🌙  
> - Not yet validated on downstream task
> - Some methods in fragment building are heuristic and unproven, the technical report and code for them will be released soon!
> - I’m still learning ML/AI~ 
> 

## ✍️ On-going
- [x] Redo evaluation with proper metrics and CI
- [>] Validation on VAE and Causal LM Transformer
- [x] Finish vocab construction on SELFIES
- [ ] Write technical report on methods, results

## 📄 License

Apache 2.0


## 🙏 Credits

- Inspired by [ChemFIE project](https://huggingface.co/gbyuvd/bionat-selfies-gen-tokenizer-wordlevel), [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/), [gen-mlm-cismi-bert](https://huggingface.co/smostafanejad/gen-mlm-cismi-bert-wordpiece), and [Tseng _et al_. 2024](https://openreview.net/forum?id=eR9C6c76j5)
- Built for efficiency
- Code & fragments vocab by gbyuvd

## References
### BibTeX
#### COCONUTDB
```bibtex
@article{sorokina2021coconut,
  title={COCONUT online: Collection of Open Natural Products database},
  author={Sorokina, Maria and Merseburger, Peter and Rajan, Kohulan and Yirik, Mehmet Aziz and Steinbeck, Christoph},
  journal={Journal of Cheminformatics},
  volume={13},
  number={1},
  pages={2},
  year={2021},
  doi={10.1186/s13321-020-00478-9}
}
```

#### ChemBL34
```bibtex
@article{zdrazil2023chembl,
  title={The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods},
  author={Zdrazil, Barbara and Felix, Eloy and Hunter, Fiona and Manners, Emma J and Blackshaw, James and Corbett, Sybilla and de Veij, Marleen and Ioannidis, Harris and Lopez, David Mendez and Mosquera, Juan F and Magarinos, Maria Paula and Bosc, Nicolas and Arcila, Ricardo and Kizil{\"o}ren, Tevfik and Gaulton, Anna and Bento, A Patr{\'i}cia and Adasme, Melissa F and Monecke, Peter and Landrum, Gregory A and Leach, Andrew R},
  journal={Nucleic Acids Research},
  year={2023},
  volume={gkad1004},
  doi={10.1093/nar/gkad1004}
}

@misc{chembl34,
  title={ChemBL34},
  year={2023},
  doi={10.6019/CHEMBL.database.34}
}
```

#### SuperNatural3
```bibtex
@article{Gallo2023,
  author = {Gallo, K and Kemmler, E and Goede, A and Becker, F and Dunkel, M and Preissner, R and Banerjee, P},
  title = {{SuperNatural 3.0-a database of natural products and natural product-based derivatives}},
  journal = {Nucleic Acids Research},
  year = {2023},
  month = jan,
  day = {6},
  volume = {51},
  number = {D1},
  pages = {D654-D659},
  doi = {10.1093/nar/gkac1008}
}
```
---

































