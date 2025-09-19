
# 🧪 FastChemTokenizer — A High-Performance SMILES Tokenizer built via Info-Theoretic Motif Mining

> **Optimized for chemical language modeling. 2x faster, 50% shorter sequences, minimal memory. Built with entropy-guided n-gram selection.**


## 🚀 Overview

`FastChemTokenizer` is a **trie-based, longest-match-first tokenizer** specifically designed for efficient tokenization of **SMILES strings** in molecular language modeling. The tokenizer is built from scratch for speed and compactness, it outperforms popular tokenizers like [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/)'s while maintaining 0% UNK rate on ~2.7M dataset and compatibility with Hugging Face `transformers`. In n-grams building, this project uses [seyonec/ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/)'s as early tokenizer for determining n-grams using its token_ids, then uses information-theoretic filtering (entropy reduction, PMI, internal entropy) to extract meaningful statistical chemical motifs — then balances 391 backbone (functional) and 391 tail fragments for structural coverage.

Trained on ~2.7M valid SMILES built and curated from ChemBL34 (Zdrazil _et al._ 2023), COCONUTDB (Sorokina _et al._ 2021), and Supernatural3 (Gallo _et al._ 2023) dataset; from resulting 76K n-grams -> pruned to **1,238 tokens**, including backbone/tail motifs and special tokens.

The "comb_smi.csv" dataset can be downloaded [here](https://huggingface.co/datasets/gbyuvd/bioactives-naturals-smiles-molgen).

## ⚡ Performance Highlights

| Metric                          | FastChemTokenizer | [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/) Tokenizer | [gen-mlm-cismi-bert](https://huggingface.co/smostafanejad/gen-mlm-cismi-bert-wordpiece) |
|--------------------------------|-------------------|----------------------|---------------------|
| **Avg time per SMILES**        | **0.0803 ms**     | 0.1581 ms            | 0.0938 ms           |
| **Avg sequence length**        | **21.49 tokens**  | 41.99 tokens         | 50.57 tokens        |
| **Throughput**                 | **12,448/sec**    | 6,326/sec            | 10,658/sec          |
| **Peak memory usage**          | **17.08 MB**      | 259.45 MB            | 387.43 MB           |
| **UNK token rate**             | **0.0000%**       | 0.0000%              | ~0.0000% (non-zero)             |
| **1000 encodes (benchmark)**   | **0.0029s**       | 1.6598s              | 0.5491s             |

✅ **1.97x faster** than ChemBERTa  
✅ **1.50x faster** than gen-mlm-cismi-bert  
✅ **No indexing errors** (avoids >512 token sequences)  
✅ **Zero unknown tokens** on validation set



## 🧩 Vocabulary

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

```python
from FastChemTokenizer import FastChemTokenizer

tokenizer = FastChemTokenizer.from_pretrained("./chemtok")
benzene = "c1ccccc1"
encoded = tokenizer.encode(benzene)
print("✅ Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("✅ Decoded:", decoded)
tokenizer.decode_with_trace(encoded)

# ✅ Encoded: [489, 640]
# ✅ Decoded: c1ccccc1

# 🔍 Decoding 2 tokens:
#  [000] ID=  489 → 'c1ccc'
#  [001] ID=  640 → 'cc1'
```


## 📦 Installation & Usage

1. Clone this repository to a directory
2. Load with:
```python
from FastChemTokenizer import FastChemTokenizer

tokenizer = FastChemTokenizer.from_pretrained("./chemtok")
```
3. Use like any Hugging Face tokenizer:
```python
outputs = tokenizer.batch_encode_plus(smiles_list, padding=True, truncation=True, max_length=512)
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
- [>] Validation on VAE and Causal LM Transformer
- [>] Finish vocab construction on SELFIES
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






