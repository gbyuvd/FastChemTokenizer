import torch
import json
import os
from typing import List, Union, Optional, Tuple, Dict, Any
from functools import lru_cache
from collections.abc import Mapping


# ------------------------------
# BatchEncoding
# ------------------------------
class BatchEncoding(dict, Mapping):
    """Minimal BatchEncoding compatible wrapper."""

    def __init__(self, data: dict, tensor_type: Optional[str] = None):
        data = {} if data is None else {k: v for k, v in data.items()}
        super().__init__(data)
        self.data = data
        self.tensor_type = tensor_type
        for k, v in data.items():
            setattr(self, k, v)

    def __getitem__(self, key): return self.data[key]
    def __iter__(self): return iter(self.data)
    def __len__(self): return len(self.data)
    def keys(self): return self.data.keys()
    def values(self): return self.data.values()
    def items(self): return self.data.items()
    def get(self, key, default=None): return self.data.get(key, default)

    def to(self, device):
        if self.tensor_type in ("pt", "torch"):
            for k, v in list(self.data.items()):
                if torch.is_tensor(v):
                    self.data[k] = v.to(device)
                    setattr(self, k, self.data[k])
        return self

    def cpu(self): return self.to("cpu")
    def cuda(self): return self.to("cuda")
    def detach(self):
        if self.tensor_type in ("pt", "torch"):
            for k, v in list(self.data.items()):
                if torch.is_tensor(v):
                    self.data[k] = v.detach()
                    setattr(self, k, self.data[k])
        return self

    def __repr__(self):
        keys = ", ".join(list(self.data.keys())[:10])
        return f"BatchEncoding(keys=[{keys}], tensor_type={self.tensor_type})"


# ------------------------------
# Base class
# ------------------------------
class PreTrainedTokenizerBase:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key.endswith('_token'):
                setattr(self, f"_{key}", value)
                setattr(self, f"{key}_id", None)
        self.model_max_length = kwargs.get('model_max_length', 512)
        self.padding_side = kwargs.get('padding_side', 'right')
        self.truncation_side = kwargs.get('truncation_side', 'right')
        self.chat_template = kwargs.get('chat_template')


# ------------------------------
# Trie node
# ------------------------------
class TrieNode:
    __slots__ = ['children', 'token_id']
    def __init__(self):
        self.children = {}
        self.token_id = None


# ------------------------------
# FastChemTokenizer
# ------------------------------

class FastChemTokenizer(PreTrainedTokenizerBase):
    def __init__(self, token_to_id=None, vocab_file=None, **kwargs):
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                token_to_id = json.load(f)
                token_to_id = {str(k): int(v) for k, v in token_to_id.items()}

        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

        # Build trie
        self.trie_root = self._build_trie(self.token_to_id)

        # ✅ Call parent (sets token *strings*, may reset *_id to None)
        super().__init__(
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            model_max_length=kwargs.get("model_max_length", 512),
            padding_side=kwargs.get("padding_side", "right"),
            truncation_side=kwargs.get("truncation_side", "right"),
            **kwargs,
        )

        # ✅ Re-map token strings → IDs from vocab
        self.bos_token_id  = self.token_to_id.get("<s>", 0)
        self.eos_token_id  = self.token_to_id.get("</s>", 1)
        self.pad_token_id  = self.token_to_id.get("<pad>", 2)
        self.unk_token_id  = self.token_to_id.get("<unk>", 3)
        self.mask_token_id = self.token_to_id.get("<mask>", 4)

        # Ensure reverse mapping always valid
        self.id_to_token[self.bos_token_id]  = "<s>"
        self.id_to_token[self.eos_token_id]  = "</s>"
        self.id_to_token[self.pad_token_id]  = "<pad>"
        self.id_to_token[self.unk_token_id]  = "<unk>"
        self.id_to_token[self.mask_token_id] = "<mask>"

        # Debug
        print("✅ Special tokens bound:",
              self.bos_token_id, self.eos_token_id, self.pad_token_id,
              self.unk_token_id, self.mask_token_id)
        
        # ✅ Ensure token *strings* also exist (for decode fallback)
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

    @property
    def vocab_size(self): return len(self.token_to_id)
    def __len__(self): return len(self.token_to_id)
    def get_vocab(self) -> Dict[str, int]: return self.token_to_id.copy()

    @lru_cache(maxsize=10000)
    def _cached_encode_str(self, s: str) -> Tuple[int, ...]:
        return tuple(self._encode_core(s))

    def _encode_core(self, text: str) -> List[int]:
        tokens, result_ids = text, []
        i, n = 0, len(tokens)
        while i < n:
            node, j = self.trie_root, i
            last_match_id, last_match_end = None, i
            while j < n and tokens[j] in node.children:
                node = node.children[tokens[j]]
                j += 1
                if node.token_id is not None:
                    last_match_id, last_match_end = node.token_id, j
            if last_match_id is not None:
                result_ids.append(last_match_id)
                i = last_match_end
            else:
                tid = self.token_to_id.get(tokens[i], self.unk_token_id)
                result_ids.append(tid)
                i += 1
        return result_ids

    # ------------------------------
    # Converters
    # ------------------------------
    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)
    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]):
        if isinstance(tokens, str): return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]):
        if isinstance(ids, int): return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str: return "".join(tokens)

    # ------------------------------
    # Encoding / Decoding
    # ------------------------------
        # ------------------------------
    # Convenience wrappers
    # ------------------------------
    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> List[int]:
        encoded = self.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)
            input_ids = input_ids.tolist()
        return input_ids

    def __call__(
        self, 
        text: Union[str, List[str]], 
        text_pair: Optional[Union[str, List[str]]] = None, 
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, Any]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """HuggingFace-compatible: one string → encode_plus, list → batch_encode_plus"""
        if return_token_type_ids is None:
            return_token_type_ids = True
        if return_attention_mask is None:
            return_attention_mask = True

        if isinstance(text, list):
            if text_pair is not None:
                batch = [(t, p) for t, p in zip(text, text_pair)]
            else:
                batch = text
            return self.batch_encode_plus(
                batch,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs
            )
        else:
            return self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs
            )

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, Any]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        if max_length is None: max_length = self.model_max_length
        ids_a = list(self._cached_encode_str(text.strip()))
        ids_b = list(self._cached_encode_str(text_pair.strip())) if text_pair else None

        input_ids, token_type_ids = [], []
        if add_special_tokens:
            input_ids.append(self.bos_token_id); token_type_ids.append(0)
            input_ids.extend(ids_a); token_type_ids.extend([0] * len(ids_a))
            input_ids.append(self.eos_token_id); token_type_ids.append(0)
            if ids_b is not None:
                input_ids.extend(ids_b); token_type_ids.extend([1] * len(ids_b))
                input_ids.append(self.eos_token_id); token_type_ids.append(1)
        else:
            input_ids = ids_a.copy(); token_type_ids = [0] * len(input_ids)
            if ids_b is not None:
                input_ids.extend(ids_b); token_type_ids.extend([1] * len(ids_b))

        if truncation and len(input_ids) > max_length:
            input_ids, token_type_ids = input_ids[:max_length], token_type_ids[:max_length]

        encoded_dict = {"input_ids": input_ids}
        if return_attention_mask:
            if padding == True or padding == "max_length":
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    if self.padding_side == "right":
                        input_ids.extend([self.pad_token_id] * pad_len)
                        token_type_ids.extend([0] * pad_len)
                    else:
                        input_ids = [self.pad_token_id] * pad_len + input_ids
                        token_type_ids = [0] * pad_len + token_type_ids
            attention_mask = [0 if tid == self.pad_token_id else 1 for tid in input_ids]
            encoded_dict["attention_mask"] = attention_mask
        if return_token_type_ids: encoded_dict["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_dict["special_tokens_mask"] = [
                1 if tid in {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.mask_token_id} else 0
                for tid in input_ids
            ]
        if return_length:
            encoded_dict["length"] = len([tid for tid in input_ids if tid != self.pad_token_id])

        if return_tensors in ["pt", "torch"]:
            out = {}
            for k, v in encoded_dict.items():
                if isinstance(v, list):
                    tensor = torch.tensor(
                        [self.unk_token_id if x is None else int(x) for x in v], dtype=torch.long
                    ).unsqueeze(0)
                    out[k] = tensor
                else:
                    out[k] = v
            return BatchEncoding(out, tensor_type=return_tensors)
        return BatchEncoding(encoded_dict, tensor_type=None)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, Any]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        if padding is True: padding = "longest"
        if padding == "max_length" and max_length is None: max_length = self.model_max_length

        all_input_ids, all_token_type_ids, all_attention_masks = [], [], []
        all_special_masks, all_lengths = [], []
        for item in batch_text_or_text_pairs:
            t, tp = item if isinstance(item, tuple) else (item, None)
            enc = self.encode_plus(
                text=t, text_pair=tp, add_special_tokens=add_special_tokens,
                padding=False, truncation=truncation, max_length=max_length,
                return_tensors=None, return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length, **kwargs
            )
            ids, tt, am = enc["input_ids"], enc.get("token_type_ids", [0]*len(enc["input_ids"])), enc.get("attention_mask",[1]*len(enc["input_ids"]))
            sm, ln = enc.get("special_tokens_mask",[0]*len(ids)), enc.get("length", len([x for x in ids if x != self.pad_token_id]))
            all_input_ids.append(ids); all_token_type_ids.append(tt); all_attention_masks.append(am)
            all_special_masks.append(sm); all_lengths.append(ln)

        pad_to = max(len(x) for x in all_input_ids) if padding == "longest" else (max_length if padding == "max_length" else None)
        batched = {
            "input_ids": all_input_ids,
            "token_type_ids": all_token_type_ids if return_token_type_ids else None,
            "attention_mask": all_attention_masks if return_attention_mask else None,
            "special_tokens_mask": all_special_masks if return_special_tokens_mask else None,
            "length": all_lengths if return_length else None,
        }
        if pad_to is not None:
            for key in ["input_ids","token_type_ids","attention_mask","special_tokens_mask"]:
                if batched.get(key) is None: continue
                padded = []
                for seq in batched[key]:
                    pad_len = pad_to - len(seq)
                    pad_val = self.pad_token_id if key=="input_ids" else 0
                    if pad_len > 0:
                        seq = seq+[pad_val]*pad_len if self.padding_side=="right" else [pad_val]*pad_len+seq
                    padded.append(seq)
                batched[key] = padded

        if return_tensors in ["pt", "torch"]:
            def to_tensor(lst, pad_val=0):
                return torch.tensor([[self.unk_token_id if x is None else int(x) for x in row] for row in lst], dtype=torch.long)
            out = {}
            if batched.get("input_ids") is not None: out["input_ids"] = to_tensor(batched["input_ids"], self.pad_token_id)
            if batched.get("attention_mask") is not None: out["attention_mask"] = to_tensor(batched["attention_mask"],0)
            if batched.get("token_type_ids") is not None: out["token_type_ids"] = to_tensor(batched["token_type_ids"],0)
            if batched.get("special_tokens_mask") is not None: out["special_tokens_mask"] = to_tensor(batched["special_tokens_mask"],0)
            if return_length and batched.get("length") is not None: out["length"] = torch.tensor([int(x) for x in batched["length"]], dtype=torch.long)
            return BatchEncoding(out, tensor_type=return_tensors)
        return BatchEncoding({k:v for k,v in batched.items() if v is not None}, tensor_type=None)

    # ------------------------------
    # Decoding
    # ------------------------------
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        special_ids = {self.bos_token_id,self.eos_token_id,self.pad_token_id,self.mask_token_id} if skip_special_tokens else set()
        tokens = [self.id_to_token.get(tid,self.unk_token) for tid in token_ids if tid not in special_ids]
        return "".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        if isinstance(sequences, torch.Tensor): sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) for seq in sequences]

    def decode_with_trace(self, token_ids: List[int]):
        print(f"\n🔍 Decoding {len(token_ids)} tokens:")
        for i, tid in enumerate(token_ids):
            token = self.id_to_token.get(tid, self.unk_token)
            tid_str = "None" if tid is None else f"{tid:5d}"
            print(f"  [{i:03d}] ID={tid_str} → '{token}'")

    # ------------------------------
    # Save / Load
    # ------------------------------
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory): os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory,(filename_prefix+"-" if filename_prefix else "")+"vocab.json")
        with open(vocab_file,"w",encoding="utf-8") as f: json.dump(self.token_to_id,f,ensure_ascii=False,indent=2)
        return (vocab_file,)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], filename_prefix: Optional[str]=None, **kwargs):
        if not os.path.exists(save_directory): os.makedirs(save_directory)
        self.save_vocabulary(save_directory, filename_prefix)
        config_file = os.path.join(save_directory,"tokenizer_config.json")
        with open(config_file,"w",encoding="utf-8") as f:
            json.dump({
                "tokenizer_class": self.__class__.__name__,
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
                "special_tokens": {
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "mask_token": self.mask_token,
                }
            },f,ensure_ascii=False,indent=2)
        return (save_directory,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path,"vocab.json")
            config_file = os.path.join(pretrained_model_name_or_path,"tokenizer_config.json")
            config = {}
            if os.path.exists(config_file):
                with open(config_file,"r",encoding="utf-8") as f: config=json.load(f)
            return cls(vocab_file=vocab_file, **{**config,**kwargs})
        else:
            raise NotImplementedError("Loading from Hub not implemented yet")


# ------------------------------
# SELFIES variant
# ------------------------------
class FastChemTokenizerSelfies(FastChemTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # ✅ ensures BOS/EOS etc. are set

    """SELFIES variant that handles whitespace-separated tokens."""

    def _encode_core(self, text: str) -> List[int]:
        result_ids, i, n = [], 0, len(text)
        while i < n:
            if text[i].isspace(): i += 1; continue
            node, j = self.trie_root, i
            last_match_id, last_match_end = None, i
            while j < n and text[j] in node.children:
                node = node.children[text[j]]; j += 1
                if node.token_id is not None:
                    last_match_id, last_match_end = node.token_id, j
            if last_match_id is not None:
                result_ids.append(last_match_id); i = last_match_end
            else:
                result_ids.append(self.token_to_id.get(text[i], self.unk_token_id)); i += 1
        return result_ids

    def convert_tokens_to_string(self, tokens: List[str]) -> str: return " ".join(tokens)
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, torch.Tensor): token_ids = token_ids.tolist()
        special_ids = {self.bos_token_id,self.eos_token_id,self.pad_token_id,self.mask_token_id} if skip_special_tokens else set()
        tokens = [self.id_to_token.get(tid,self.unk_token) for tid in token_ids if tid not in special_ids]
        return " ".join(tokens)
