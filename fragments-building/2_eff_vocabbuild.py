import polars as pl
import pickle
import math
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

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

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


MAX_N = 8
BATCH_SIZE = 1000_000
MIN_COUNT_BATCH = 200   # 25th percentile of top-k 10K; filter per batch
MIN_COUNT_GLOBAL = 800  # median of top-k 10K; filter per global

# -------------------------
# 1. Load merged tables and dictionary
# -------------------------
print("Loading data...")
ngram_table = pl.read_parquet("ngram_counts_final.parquet")
prefix_table = pl.read_parquet("prefix_counts_final.parquet")

with open("ngram_dictionary.pkl", "rb") as f:
    id2ngram = pickle.load(f)

print(f"Loaded {len(ngram_table)} n-grams and {len(prefix_table)} prefix->next patterns")

# -------------------------
# 2. Apply global count filtering FIRST
# -------------------------
print(f"Applying global count filter (>= {MIN_COUNT_GLOBAL})...")
ngram_table = ngram_table.filter(pl.col("count") >= MIN_COUNT_GLOBAL)
prefix_table = prefix_table.filter(pl.col("count") >= MIN_COUNT_GLOBAL)

print(f"After global count filtering: {len(ngram_table):,} n-grams and {len(prefix_table):,} prefix patterns remain")

# -------------------------
# 3. Compute frequency probabilities
# -------------------------
print("Computing frequency probabilities...")
unigram_mask = [len(id2ngram[row["ngram_id"]]) == 1 for row in ngram_table.to_dicts()]
unigram_total = sum(row["count"] for i, row in enumerate(ngram_table.to_dicts()) if unigram_mask[i])
print(f"Total unigram count (corpus size): {unigram_total:,}")

ngram_table = ngram_table.with_columns([
    (pl.col("count") / unigram_total).alias("p_ngram")
])

# -------------------------
# 4. Compute baseline entropy of the corpus
# -------------------------
print("Computing baseline corpus entropy...")
unigram_data = [(row["ngram_id"], row["count"]) 
                for i, row in enumerate(ngram_table.to_dicts()) 
                if unigram_mask[i]]

baseline_entropy = 0.0
for ngram_id, count in tqdm(unigram_data, desc="Computing baseline entropy"):
    p = count / unigram_total
    if p > 0:
        baseline_entropy -= p * math.log2(p)

print(f"Baseline corpus entropy: {baseline_entropy:.4f} bits")

# -------------------------
# 5. Compute conditional entropy for each prefix (entropy reduction)
# -------------------------
print("Computing conditional entropies and entropy reduction...")
prefix_totals = prefix_table.group_by("prefix_id").agg([
    pl.sum("count").alias("prefix_total")
])

prefix_probs = prefix_table.join(prefix_totals, on="prefix_id").with_columns([
    (pl.col("count") / pl.col("prefix_total")).alias("p_next_given_prefix")
])

prefix_probs = prefix_probs.with_columns([
    (-pl.col("p_next_given_prefix") * 
    pl.col("p_next_given_prefix").map_elements(lambda x: math.log2(max(x, 1e-12)), return_dtype=pl.Float64)
    ).alias("entropy_term")
])

prefix_entropy = prefix_probs.group_by("prefix_id").agg([
    pl.sum("entropy_term").alias("conditional_entropy")
])

prefix_entropy = prefix_entropy.with_columns([
    (baseline_entropy - pl.col("conditional_entropy")).alias("entropy_reduction")
])

print(f"Computed entropy reduction for {len(prefix_entropy)} prefixes")

# -------------------------
# 6. Compute PMI for n-grams
# -------------------------
print("Computing PMI scores...")

def compute_generalized_pmi():
    unigram_probs = {}
    for row in tqdm(ngram_table.to_dicts(), desc="Building unigram probability lookup"):
        ngram = id2ngram[row["ngram_id"]]
        if len(ngram) == 1:
            unigram_probs[ngram[0]] = row["p_ngram"]
    
    pmi_scores = []
    for row in tqdm(ngram_table.to_dicts(), desc="Computing PMI scores"):
        ngram_id = row["ngram_id"]
        ngram = id2ngram[ngram_id]
        p_ngram = row["p_ngram"]
        
        if len(ngram) == 1:
            pmi = 0.0
        else:
            product_prob = 1.0
            for token in ngram:
                product_prob *= unigram_probs.get(token, 1e-12)
            if product_prob > 0 and p_ngram > 0:
                pmi = math.log2(p_ngram / product_prob)
            else:
                pmi = 0.0
        pmi_scores.append(pmi)
    return pmi_scores

pmi_scores = compute_generalized_pmi()
ngram_table = ngram_table.with_columns([pl.Series("pmi", pmi_scores)])

# -------------------------
# 7. Join entropy reduction data
# -------------------------
print("Joining entropy reduction data...")
ngram_table = ngram_table.join(
    prefix_entropy.select(["prefix_id", "entropy_reduction"]).rename({"prefix_id": "ngram_id"}),
    on="ngram_id", 
    how="left"
).with_columns([
    pl.col("entropy_reduction").fill_null(0.0)
])

# -------------------------
# 8. Compute internal entropy
# -------------------------
print("Computing internal entropy and applying quality filters...")
ngram_table = ngram_table.join(
    prefix_entropy.select(["prefix_id", "conditional_entropy"]).rename({"prefix_id": "ngram_id"}),
    on="ngram_id",
    how="left"
).with_columns([
    pl.col("conditional_entropy").fill_null(0.0).alias("internal_entropy")
])

INTERNAL_ENTROPY_THRESHOLD = 0.5
print(f"Applying internal entropy filter > {INTERNAL_ENTROPY_THRESHOLD}...")
quality_filtered = ngram_table.filter(
    (pl.col("internal_entropy") > INTERNAL_ENTROPY_THRESHOLD) |
    (pl.col("internal_entropy") == 0.0)
)

print(f"After internal entropy filtering: {len(quality_filtered):,} n-grams remain")

# -------------------------
# 9. Normalize components
# -------------------------
print("Normalizing scoring components...")

def normalize_column(df, col_name):
    col_data = df[col_name].to_list()
    min_val, max_val = min(col_data), max(col_data)
    if max_val > min_val:
        return [(x - min_val) / (max_val - min_val) for x in col_data]
    else:
        return [0.0] * len(col_data)

log_freq = [math.log2(max(p, 1e-12)) for p in quality_filtered["p_ngram"].to_list()]
quality_filtered = quality_filtered.with_columns([pl.Series("log_frequency", log_freq)])

freq_norm = normalize_column(quality_filtered, "log_frequency")
pmi_norm = normalize_column(quality_filtered, "pmi") 
entropy_red_norm = normalize_column(quality_filtered, "entropy_reduction")
internal_ent_norm = normalize_column(quality_filtered, "internal_entropy")

quality_filtered = quality_filtered.with_columns([
    pl.Series("frequency_norm", freq_norm),
    pl.Series("pmi_norm", pmi_norm),
    pl.Series("entropy_reduction_norm", entropy_red_norm), 
    pl.Series("internal_entropy_norm", internal_ent_norm)
])

ENTROPY_REDUCTION_MAX = 0.95
print(f"Applying entropy reduction filter < {ENTROPY_REDUCTION_MAX}...")
final_filtered = quality_filtered.filter(
    pl.col("entropy_reduction_norm") < ENTROPY_REDUCTION_MAX
)

print(f"After entropy reduction filtering: {len(final_filtered):,} n-grams remain")

# -------------------------
# 10. Composite score
# -------------------------
print("Computing composite scores...")
ALPHA, BETA, GAMMA, DELTA = 0.25, 0.25, 0.35, 0.15
final_filtered = final_filtered.with_columns([
    (ALPHA * pl.col("frequency_norm") + 
     BETA * pl.col("pmi_norm") + 
     GAMMA * pl.col("entropy_reduction_norm") +
     DELTA * pl.col("internal_entropy_norm")
    ).alias("composite_score")
])

# -------------------------
# 11. Decode n-grams
# -------------------------
print("Adding decoded n-grams...")
def decode_ngram_id(ngram_id):
    ngram = id2ngram[ngram_id]
    return " ".join(map(str, ngram))

decoded_ngrams = [decode_ngram_id(row["ngram_id"]) for row in tqdm(final_filtered.to_dicts(), desc="Decoding n-grams")]
final_filtered = final_filtered.with_columns([pl.Series("ngram_decoded", decoded_ngrams)])

# -------------------------
# 12. Rank results + Backbone vs Tails
# -------------------------
print("Ranking final results and splitting backbone/tails...")
motifs_ranked = final_filtered.sort("composite_score", descending=True)

backbone = motifs_ranked.filter(pl.col("internal_entropy") > 0.5).sort("composite_score", descending=True)
tails = motifs_ranked.filter(pl.col("internal_entropy") == 0.0).sort("composite_score", descending=True)

TOP_BACKBONE = len(backbone)
TOP_TAILS = TOP_BACKBONE  # mirror size

backbone_top = backbone.head(TOP_BACKBONE)
tails_top = tails.head(TOP_TAILS)

# recombine balanced vocab
balanced_vocab = pl.concat([backbone_top, tails_top])

print(f"Backbone motifs: {len(backbone_top)}")
print(f"Tail motifs: {len(tails_top)}")
print(f"Balanced vocab: {len(balanced_vocab)}")

# -------------------------
# 13. Save results
# -------------------------
print("Saving results...")

# Save ranked (all candidates)
motifs_ranked.write_parquet("motifs_ranked_fixed.parquet", compression="snappy")

# Save top-K readable view (balanced)
TOP_K = 10000
motifs_top = balanced_vocab.head(TOP_K)
motifs_top.write_parquet("motifs_top_fixed.parquet", compression="snappy")

motifs_top.select([
    "ngram_decoded", "count", "p_ngram", "pmi", "entropy_reduction", 
    "internal_entropy", "frequency_norm", "pmi_norm", 
    "entropy_reduction_norm", "internal_entropy_norm", "composite_score"
]).write_csv("top_motifs_readable.csv")

# Save tokenizer vocab
balanced_vocab.write_parquet("tokenizer_vocab.parquet", compression="snappy")
balanced_vocab.select(["ngram_decoded"]).write_csv("tokenizer_vocab.csv")

# -------------------------
# 14. Summary
# -------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
original_ngrams = pl.read_parquet("ngram_counts_final.parquet")
print(f"Total n-grams originally: {len(original_ngrams):,}")
print(f"After global count filter: {len(ngram_table):,}")
print(f"After internal entropy filter: {len(quality_filtered):,}")
print(f"After entropy reduction filter: {len(final_filtered):,}")
print(f"Baseline corpus entropy: {baseline_entropy:.4f} bits")
print(f"\nBackbone motifs: {len(backbone_top)}")
print(f"Tail motifs: {len(tails_top)}")
print(f"Balanced tokenizer vocab: {len(balanced_vocab)}")
print("\n✅ Completed successfully!")
