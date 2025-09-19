import torch
import json
import os
from typing import List, Union, Optional, Tuple
from transformers.tokenization_utils_base import BatchEncoding
from functools import lru_cache

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

class TrieNode:
    __slots__ = ['children', 'token_id']
    def __init__(self):
        self.children = {}
        self.token_id = None  # If set, this node completes a valid token


class FastChemTokenizer:
    def __init__(self, token_to_id, model_max_length=512):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        # No more self.token_set — replaced by trie
        self.model_max_length = model_max_length

        # Precompute max token length for possible use & clarity
        self.max_token_len = max(len(t) for t in token_to_id.keys())

        # Build trie for fast longest-match lookup
        self.trie_root = self._build_trie(token_to_id)

        # Validate required special tokens
        required_special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        for tok in required_special_tokens:
            if tok not in token_to_id:
                raise KeyError(f"Required special token '{tok}' not found in vocab.")

        # Special token IDs
        self.bos_token_id = token_to_id["<s>"]
        self.eos_token_id = token_to_id["</s>"]
        self.pad_token_id = token_to_id["<pad>"]
        self.unk_token_id = token_to_id["<unk>"]
        self.mask_token_id = token_to_id["<mask>"]

        # Special tokens for convenience
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

    def _build_trie(self, token_to_id):
        root = TrieNode()
        for token, tid in token_to_id.items():
            node = root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.token_id = tid
        return root

    def __len__(self):
        """Return vocab size — REQUIRED for HF compatibility."""
        return len(self.token_to_id)

    def __call__(self, text: Union[str, List[str]], text_pair: Optional[Union[str, List[str]]] = None, **kwargs) -> BatchEncoding:
        if isinstance(text, list):
            batch = [(t, p) if p is not None else t for t, p in zip(text, text_pair)] if text_pair else text
            return self.batch_encode_plus(batch, **kwargs)
        else:
            return self.encode_plus(text=text, text_pair=text_pair, **kwargs)

    @lru_cache(maxsize=10000)
    def _cached_encode_str(self, s: str) -> Tuple[int, ...]:
        return tuple(self._encode_core(s))

    def _encode_core(self, text: str) -> List[int]:
        """Core encoding logic using Trie — no caching."""
        tokens = text
        result_ids = []
        i = 0
        n = len(tokens)

        while i < n:
            node = self.trie_root
            j = i
            last_match_id = None
            last_match_end = i

            # Traverse trie while characters match
            while j < n and tokens[j] in node.children:
                node = node.children[tokens[j]]
                j += 1
                if node.token_id is not None:
                    last_match_id = node.token_id
                    last_match_end = j  # Remember end of valid token

            if last_match_id is not None:
                result_ids.append(last_match_id)
                i = last_match_end
            else:
                # Fallback: encode single char
                tok = tokens[i]
                result_ids.append(self.token_to_id.get(tok, self.unk_token_id))
                i += 1

        return result_ids

    def encode(self, text: str) -> List[int]:
        """Public encode method — strips input and uses cache."""
        return list(self._cached_encode_str(text.strip()))

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            special_ids = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.mask_token_id,  
            }
        else:
            special_ids = set()

        tokens = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            token = self.id_to_token.get(tid, self.unk_token)
            tokens.append(token)

        return "".join(tokens)

    def decode_with_trace(self, token_ids: List[int]) -> None:
        print(f"\n🔍 Decoding {len(token_ids)} tokens:")
        for i, tid in enumerate(token_ids):
            token = self.id_to_token.get(tid, self.unk_token)
            print(f"  [{i:03d}] ID={tid:5d} → '{token}'")

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.model_max_length

        ids_a = self.encode(text)

        if text_pair is not None:
            ids_b = self.encode(text_pair)
        else:
            ids_b = None

        input_ids = []
        token_type_ids = []

        if add_special_tokens:
            input_ids.append(self.bos_token_id)
            token_type_ids.append(0)
            if ids_b is not None:
                input_ids.extend(ids_a)
                token_type_ids.extend([0] * len(ids_a))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(0)

                input_ids.extend(ids_b)
                token_type_ids.extend([1] * len(ids_b))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(1)
            else:
                input_ids.extend(ids_a)
                token_type_ids.extend([0] * len(ids_a))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(0)
        else:
            input_ids = ids_a
            token_type_ids = [0] * len(input_ids)
            if ids_b is not None:
                input_ids.extend(ids_b)
                token_type_ids.extend([1] * len(ids_b))

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]

        if padding:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids.extend([self.pad_token_id] * pad_len)
                token_type_ids.extend([0] * pad_len)

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        encoded_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_token_type_ids:
            encoded_dict["token_type_ids"] = token_type_ids

        if return_tensors == "pt":
            output = {}
            for k, v in encoded_dict.items():
                tensor = torch.tensor(v, dtype=torch.long)  #  Fixed: use torch.tensor, not as_tensor
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                output[k] = tensor
        else:
            output = encoded_dict

        return BatchEncoding(output, tensor_type=return_tensors)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        **kwargs
    ) -> BatchEncoding:
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []

        for item in batch_text_or_text_pairs:
            if isinstance(item, tuple):
                text, text_pair = item
            else:
                text, text_pair = item, None

            encoded = self.encode_plus(
                text=text,
                text_pair=text_pair,
                **kwargs
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            if "token_type_ids" in encoded:
                all_token_type_ids.append(encoded["token_type_ids"])

        batched = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }
        if all_token_type_ids:
            batched["token_type_ids"] = all_token_type_ids

        if kwargs.get("return_tensors") == "pt":
            def to_tensor_list(lst):
                # Use torch.tensor for safety — avoids "copy construct from tensor" warning
                return [torch.tensor(item, dtype=torch.long) for item in lst]

            batched = {
                k: torch.nn.utils.rnn.pad_sequence(
                    to_tensor_list(v),
                    batch_first=True,
                    padding_value=self.pad_token_id if k == "input_ids" else 0
                )
                for k, v in batched.items()
            }

        return BatchEncoding(batched, tensor_type=kwargs.get("return_tensors"))

    # Save vocab to directory
    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer vocab as `vocab.json` in target directory.
        Mimics Hugging Face convention.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(save_directory, "vocab.json")

        # Keys are strings, values are ints — JSON-safe
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        print(f"✅ Tokenizer vocab saved to: {vocab_file}")

    # Load from pretrained directory
    @classmethod
    def from_pretrained(cls, pretrained_directory: str, model_max_length=512):
        """
        Load tokenizer from directory containing `vocab.json`.
        """
        vocab_file = os.path.join(pretrained_directory, "vocab.json")

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)

        # Convert keys to str (JSON loads as str anyway), values to int
        token_to_id = {str(k): int(v) for k, v in token_to_id.items()}

        return cls(token_to_id=token_to_id, model_max_length=model_max_length)

class FastChemTokenizerSelfies:
    def __init__(self, token_to_id, model_max_length=512):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        # No more self.token_set — replaced by trie
        self.model_max_length = model_max_length

        # Precompute max token length for possible use & clarity
        self.max_token_len = max(len(t) for t in token_to_id.keys())

        # Build trie for fast longest-match lookup
        self.trie_root = self._build_trie(token_to_id)

        # Validate required special tokens
        required_special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        for tok in required_special_tokens:
            if tok not in token_to_id:
                raise KeyError(f"Required special token '{tok}' not found in vocab.")

        # Special token IDs
        self.bos_token_id = token_to_id["<s>"]
        self.eos_token_id = token_to_id["</s>"]
        self.pad_token_id = token_to_id["<pad>"]
        self.unk_token_id = token_to_id["<unk>"]
        self.mask_token_id = token_to_id["<mask>"]

        # Special tokens for convenience
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

    def _build_trie(self, token_to_id):
        root = TrieNode()
        for token, tid in token_to_id.items():
            node = root
            for char in token:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.token_id = tid
        return root

    def __len__(self):
        """Return vocab size — REQUIRED for HF compatibility."""
        return len(self.token_to_id)

    def __call__(self, text: Union[str, List[str]], text_pair: Optional[Union[str, List[str]]] = None, **kwargs) -> BatchEncoding:
        if isinstance(text, list):
            batch = [(t, p) if p is not None else t for t, p in zip(text, text_pair)] if text_pair else text
            return self.batch_encode_plus(batch, **kwargs)
        else:
            return self.encode_plus(text=text, text_pair=text_pair, **kwargs)

    @lru_cache(maxsize=10000)
    def _cached_encode_str(self, s: str) -> Tuple[int, ...]:
        return tuple(self._encode_core(s))

    def _encode_core(self, text: str) -> List[int]:
        """Core encoding logic using Trie — skips whitespace if not part of a token."""
        result_ids = []
        i = 0
        n = len(text)

        while i < n:
            if text[i].isspace():  # ← Skip whitespace unless part of a token
                i += 1
                continue

            node = self.trie_root
            j = i
            last_match_id = None
            last_match_end = i

            # Traverse trie while characters match
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.token_id is not None:
                    last_match_id = node.token_id
                    last_match_end = j

            if last_match_id is not None:
                result_ids.append(last_match_id)
                i = last_match_end
            else:
                # Fallback: encode single char
                result_ids.append(self.token_to_id.get(text[i], self.unk_token_id))
                i += 1

        return result_ids


    def encode(self, text: str) -> List[int]:
        """Public encode method — strips input and uses cache."""
        return list(self._cached_encode_str(text.strip()))

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            special_ids = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.mask_token_id,
            }
        else:
            special_ids = set()

        tokens = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            token = self.id_to_token.get(tid, self.unk_token)
            tokens.append(token)

        # ✅ Join with SPACE between tokens — this reconstructs original format
        return " ".join(tokens)

    def decode_with_trace(self, token_ids: List[int]) -> None:
        print(f"\n🔍 Decoding {len(token_ids)} tokens:")
        for i, tid in enumerate(token_ids):
            token = self.id_to_token.get(tid, self.unk_token)
            print(f"  [{i:03d}] ID={tid:5d} → '{token}'")

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, self.unk_token) for i in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.model_max_length

        ids_a = self.encode(text)

        if text_pair is not None:
            ids_b = self.encode(text_pair)
        else:
            ids_b = None

        input_ids = []
        token_type_ids = []

        if add_special_tokens:
            input_ids.append(self.bos_token_id)
            token_type_ids.append(0)
            if ids_b is not None:
                input_ids.extend(ids_a)
                token_type_ids.extend([0] * len(ids_a))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(0)

                input_ids.extend(ids_b)
                token_type_ids.extend([1] * len(ids_b))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(1)
            else:
                input_ids.extend(ids_a)
                token_type_ids.extend([0] * len(ids_a))
                input_ids.append(self.eos_token_id)
                token_type_ids.append(0)
        else:
            input_ids = ids_a
            token_type_ids = [0] * len(input_ids)
            if ids_b is not None:
                input_ids.extend(ids_b)
                token_type_ids.extend([1] * len(ids_b))

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]

        if padding:
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids.extend([self.pad_token_id] * pad_len)
                token_type_ids.extend([0] * pad_len)

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        encoded_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_token_type_ids:
            encoded_dict["token_type_ids"] = token_type_ids

        if return_tensors == "pt":
            output = {}
            for k, v in encoded_dict.items():
                tensor = torch.tensor(v, dtype=torch.long)  #  Fixed: use torch.tensor, not as_tensor
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                output[k] = tensor
        else:
            output = encoded_dict

        return BatchEncoding(output, tensor_type=return_tensors)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        **kwargs
    ) -> BatchEncoding:
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []

        for item in batch_text_or_text_pairs:
            if isinstance(item, tuple):
                text, text_pair = item
            else:
                text, text_pair = item, None

            encoded = self.encode_plus(
                text=text,
                text_pair=text_pair,
                **kwargs
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            if "token_type_ids" in encoded:
                all_token_type_ids.append(encoded["token_type_ids"])

        batched = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
        }
        if all_token_type_ids:
            batched["token_type_ids"] = all_token_type_ids

        if kwargs.get("return_tensors") == "pt":
            def to_tensor_list(lst):
                # Use torch.tensor for safety — avoids "copy construct from tensor" warning
                return [torch.tensor(item, dtype=torch.long) for item in lst]

            batched = {
                k: torch.nn.utils.rnn.pad_sequence(
                    to_tensor_list(v),
                    batch_first=True,
                    padding_value=self.pad_token_id if k == "input_ids" else 0
                )
                for k, v in batched.items()
            }

        return BatchEncoding(batched, tensor_type=kwargs.get("return_tensors"))

    # Save vocab to directory
    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer vocab as `vocab.json` in target directory.
        Mimics Hugging Face convention.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(save_directory, "vocab.json")

        # Keys are strings, values are ints — JSON-safe
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        print(f"✅ Tokenizer vocab saved to: {vocab_file}")

    # Load from pretrained directory
    @classmethod
    def from_pretrained(cls, pretrained_directory: str, model_max_length=512):
        """
        Load tokenizer from directory containing `vocab.json`.
        """
        vocab_file = os.path.join(pretrained_directory, "vocab.json")

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)

        # Convert keys to str (JSON loads as str anyway), values to int
        token_to_id = {str(k): int(v) for k, v in token_to_id.items()}

        return cls(token_to_id=token_to_id, model_max_length=model_max_length)