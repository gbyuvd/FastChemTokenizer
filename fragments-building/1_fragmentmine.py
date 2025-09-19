from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from collections import Counter
import polars as pl
from tqdm import tqdm
import pickle
import glob

# Copyright 2025 Genta Pramillean Bayu (@gbyuvd)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -------------------------
# 1. Setup
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


dataset = load_dataset("csv", data_files="./combined_smiles.csv", split="train", streaming=True)

MAX_N = 8
BATCH_SIZE = 1000_000
MIN_COUNT_BATCH = 200   # 25th percentile of top-k 10K
MIN_COUNT_GLOBAL = 800  # median of top-k 10K

# global dictionary
ngram2id, id2ngram = {}, []
next_id = 0

def get_ngram_id(ngram):
    global next_id
    if ngram not in ngram2id:
        ngram2id[ngram] = next_id
        id2ngram.append(ngram)
        next_id += 1
    return ngram2id[ngram]

# -------------------------
# 2. N-gram generator
# -------------------------
def generate_ngrams(seq, max_n):
    for n in range(1, max_n+1):
        for i in range(len(seq)-n+1):
            yield tuple(seq[i:i+n])

# -------------------------
# 3. Process batch
# -------------------------
def process_batch(tokenized_batch, batch_idx):
    ngram_counts = Counter()
    prefix_next_counts = Counter()

    for seq in tqdm(tokenized_batch, desc=f"Batch {batch_idx} sequences"):
        # n-grams
        for ng in generate_ngrams(seq, MAX_N):
            ngram_counts[ng] += 1

        # prefix -> next
        for i in range(len(seq)-1):
            prefix = tuple(seq[:i+1])
            next_tok = seq[i+1]
            prefix_next_counts[(prefix, next_tok)] += 1

    # --- per-batch filter
    ngram_counts = {ng:c for ng,c in ngram_counts.items() if c >= MIN_COUNT_BATCH}
    prefix_next_counts = {k:c for k,c in prefix_next_counts.items() if c >= MIN_COUNT_BATCH}

    # --- assign IDs
    ngram_ids = {get_ngram_id(ng): c for ng, c in ngram_counts.items()}
    prefix_ids = {(get_ngram_id(pref), nxt): c for (pref,nxt), c in prefix_next_counts.items()}

    # --- write parquet
    if ngram_ids:
        tbl = pa.table({
            "ngram_id": list(ngram_ids.keys()),
            "count": list(ngram_ids.values())
        })
        pq.write_table(tbl, f"counts_batch_{batch_idx}.parquet", compression="snappy")

    if prefix_ids:
        tbl = pa.table({
            "prefix_id": [k[0] for k in prefix_ids.keys()],
            "next_id": [k[1] for k in prefix_ids.keys()],
            "count": list(prefix_ids.values())
        })
        pq.write_table(tbl, f"prefix_batch_{batch_idx}.parquet", compression="snappy")

# -------------------------
# 4. Streaming loop
# -------------------------
batch, batch_idx = [], 0
dataset_iter = tqdm(dataset, desc="Streaming dataset")
for row in dataset_iter:
    selfies = row["SMILES"]
    
    # Skip if not a string or empty
    if not isinstance(selfies, str) or not selfies.strip():
        continue

    token_ids = tokenizer.encode(selfies, add_special_tokens=False)
    batch.append(token_ids)

    if len(batch) >= BATCH_SIZE:
        process_batch(batch, batch_idx)
        batch.clear()
        batch_idx += 1

# process remainder
if batch:
    process_batch(batch, batch_idx)

# -------------------------
# 5. Merge Phase (Polars)
# -------------------------
# Merge ngram counts
ngram_files = glob.glob("counts_batch_*.parquet")
df = pl.scan_parquet(ngram_files)
print("Merging n-grams...")
ngram_global = (
    df.group_by("ngram_id")
      .agg(pl.sum("count").alias("count"))
      .filter(pl.col("count") >= MIN_COUNT_GLOBAL)
      .collect()
)

# Merge prefix->next counts
prefix_files = glob.glob("prefix_batch_*.parquet")
df2 = pl.scan_parquet(prefix_files)
print("Merging prefix->next counts...")
prefix_global = (
    df2.group_by(["prefix_id", "next_id"])
       .agg(pl.sum("count").alias("count"))
       .filter(pl.col("count") >= MIN_COUNT_GLOBAL)
       .collect()
)

# -------------------------
# 6. Save results
# -------------------------
ngram_global.write_parquet("ngram_counts_final.parquet", compression="snappy")
prefix_global.write_parquet("prefix_counts_final.parquet", compression="snappy")

# save dictionary (id2ngram mapping)
with open("ngram_dictionary.pkl", "wb") as f:
    pickle.dump(id2ngram, f)

print("Done! Final n-grams:", len(ngram_global))
