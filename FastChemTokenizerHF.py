import torch
import json
import os
from typing import List, Union, Optional, Tuple, Dict, Any
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy, TensorType
from functools import lru_cache


class TrieNode:
    __slots__ = ['children', 'token_id']
    def __init__(self):
        self.children = {}
        self.token_id = None  # If set, this node completes a valid token


class FastChemTokenizer(PreTrainedTokenizerBase):
    """
    Fully HuggingFace API compatible tokenizer for chemical representations.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self,
        token_to_id=None,
        vocab_file=None,
        model_max_length=512,
        padding_side="right",
        truncation_side="right",
        chat_template=None,
        **kwargs
    ):
        # Handle vocab loading
        if token_to_id is None and vocab_file is None:
            raise ValueError("Either token_to_id or vocab_file must be provided")

        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                token_to_id = json.load(f)
                token_to_id = {str(k): int(v) for k, v in token_to_id.items()}

        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

        # Precompute max token length for possible use & clarity
        self.max_token_len = max(len(t) for t in token_to_id.keys()) if token_to_id else 0

        # Build trie for fast longest-match lookup
        self.trie_root = self._build_trie(token_to_id)

        # Validate required special tokens
        required_special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        for tok in required_special_tokens:
            if tok not in token_to_id:
                raise KeyError(f"Required special token '{tok}' not found in vocab.")

        # ✅ Assign special token IDs explicitly
        self.bos_token_id = token_to_id["<s>"]
        self.eos_token_id = token_to_id["</s>"]
        self.pad_token_id = token_to_id["<pad>"]
        self.unk_token_id = token_to_id["<unk>"]
        self.mask_token_id = token_to_id["<mask>"]

        # Special tokens
        bos_token = "<s>"
        eos_token = "</s>"
        pad_token = "<pad>"
        unk_token = "<unk>"
        mask_token = "<mask>"

        # Initialize parent class with all required parameters
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=None,
            pad_token=pad_token,
            cls_token=None,
            mask_token=mask_token,
            additional_special_tokens=[],
            model_max_length=model_max_length,
            padding_side=padding_side,
            truncation_side=truncation_side,
            chat_template=chat_template,
            **kwargs,
        )

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
    def vocab_size(self):
        return len(self.token_to_id)

    def __len__(self):
        return len(self.token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return self.token_to_id.copy()

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

            while j < n and tokens[j] in node.children:
                node = node.children[tokens[j]]
                j += 1
                if node.token_id is not None:
                    last_match_id = node.token_id
                    last_match_end = j

            if last_match_id is not None:
                result_ids.append(last_match_id)
                i = last_match_end
            else:
                tok = tokens[i]
                result_ids.append(self.token_to_id.get(tok, self.unk_token_id))
                i += 1

        return result_ids

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        token_ids = self._encode_core(text.strip())
        return [self.id_to_token[tid] for tid in token_ids]

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    # ✅ Public methods
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(tok) for tok in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """SMILES-style decoding: no spaces between tokens."""
        return "".join(tokens)

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

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> str:
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
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> List[str]:
        """Batch decode sequences."""
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return [
            self.decode(
                seq, 
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs
            ) 
            for seq in sequences
        ]

    def decode_with_trace(self, token_ids: List[int]) -> None:
        print(f"\n🔍 Decoding {len(token_ids)} tokens:")
        for i, tid in enumerate(token_ids):
            token = self.id_to_token.get(tid, self.unk_token)
            print(f"  [{i:03d}] ID={tid:5d} → '{token}'")

    def __call__(
        self, 
        text: Union[str, List[str]], 
        text_pair: Optional[Union[str, List[str]]] = None, 
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main callable method that handles both single and batch inputs.
        """
        # Handle defaults
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
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.model_max_length

        ids_a = list(self._cached_encode_str(text.strip()))

        if text_pair is not None:
            ids_b = list(self._cached_encode_str(text_pair.strip()))
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
            input_ids = ids_a.copy()
            token_type_ids = [0] * len(input_ids)
            if ids_b is not None:
                input_ids.extend(ids_b)
                token_type_ids.extend([1] * len(ids_b))

        # Handle truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            token_type_ids = token_type_ids[:max_length]

        # Handle padding
        if padding == True or padding == "max_length":
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                if self.padding_side == "right":
                    input_ids.extend([self.pad_token_id] * pad_len)
                    token_type_ids.extend([0] * pad_len)
                else:
                    input_ids = [self.pad_token_id] * pad_len + input_ids
                    token_type_ids = [0] * pad_len + token_type_ids

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        encoded_dict = {
            "input_ids": input_ids,
        }
        
        if return_attention_mask:
            encoded_dict["attention_mask"] = attention_mask
        
        if return_token_type_ids:
            encoded_dict["token_type_ids"] = token_type_ids
            
        if return_special_tokens_mask:
            special_tokens_mask = [
                1 if tid in {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.mask_token_id} else 0 
                for tid in input_ids
            ]
            encoded_dict["special_tokens_mask"] = special_tokens_mask
            
        if return_length:
            encoded_dict["length"] = len([tid for tid in input_ids if tid != self.pad_token_id])

        if return_tensors == "pt":
            output = {}
            for k, v in encoded_dict.items():
                tensor = torch.tensor(v, dtype=torch.long)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                output[k] = tensor
        else:
            output = encoded_dict

        return BatchEncoding(output, tensor_type=return_tensors)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_special_tokens_masks = []
        all_lengths = []

        for item in batch_text_or_text_pairs:
            if isinstance(item, tuple):
                text, text_pair = item
            else:
                text, text_pair = item, None

            encoded = self.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=False,  # We'll handle batch padding later
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=None,  # Don't convert to tensors yet
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs
            )
            
            all_input_ids.append(encoded["input_ids"])
            if "attention_mask" in encoded:
                all_attention_masks.append(encoded["attention_mask"])
            if "token_type_ids" in encoded:
                all_token_type_ids.append(encoded["token_type_ids"])
            if "special_tokens_mask" in encoded:
                all_special_tokens_masks.append(encoded["special_tokens_mask"])
            if "length" in encoded:
                all_lengths.append(encoded["length"])

        batched = {
            "input_ids": all_input_ids,
        }
        
        if all_attention_masks:
            batched["attention_mask"] = all_attention_masks
        if all_token_type_ids:
            batched["token_type_ids"] = all_token_type_ids
        if all_special_tokens_masks:
            batched["special_tokens_mask"] = all_special_tokens_masks
        if all_lengths:
            batched["length"] = all_lengths

        # Handle batch padding
        if padding == True or padding == "longest":
            max_len = max(len(ids) for ids in all_input_ids)
            for key in batched:
                if key in ["input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"]:
                    padded_seqs = []
                    for seq in batched[key]:
                        pad_len = max_len - len(seq)
                        if pad_len > 0:
                            if key == "input_ids":
                                padding_value = self.pad_token_id
                            else:
                                padding_value = 0
                            
                            if self.padding_side == "right":
                                padded_seq = seq + [padding_value] * pad_len
                            else:
                                padded_seq = [padding_value] * pad_len + seq
                        else:
                            padded_seq = seq
                        padded_seqs.append(padded_seq)
                    batched[key] = padded_seqs

        if return_tensors == "pt":
            def to_tensor_list(lst):
                return [torch.tensor(item, dtype=torch.long) for item in lst]
            
            for key in ["input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"]:
                if key in batched:
                    batched[key] = torch.nn.utils.rnn.pad_sequence(
                        to_tensor_list(batched[key]),
                        batch_first=True,
                        padding_value=self.pad_token_id if key == "input_ids" else 0
                    )
            
            # Handle non-sequence data
            if "length" in batched:
                batched["length"] = torch.tensor(batched["length"], dtype=torch.long)

        return BatchEncoding(batched, tensor_type=return_tensors)

    def pad(
        self,
        encoded_inputs,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """Pad encoded inputs."""
        # This is a simplified version - full implementation would be more complex
        return encoded_inputs

    # Save/Load methods
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to files."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
            
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)

    def save_pretrained(
        self, 
        save_directory: Union[str, os.PathLike],
        legacy_format: bool = True,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs
    ):
        """Save tokenizer to directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save vocabulary
        vocab_files = self.save_vocabulary(save_directory, filename_prefix)
        
        # Save tokenizer config
        tokenizer_config = {
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
        }
        
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

        print(f"✅ Tokenizer saved to: {save_directory}")
        
        return (save_directory,)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs
    ):
        """Load tokenizer from pretrained directory or hub."""
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
            config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
            
            # Load config if available
            config = {}
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            
            # Merge config with kwargs
            merged_config = {**config, **kwargs}
            
            return cls(vocab_file=vocab_file, **merged_config)
        else:
            raise NotImplementedError("Loading from HuggingFace Hub not implemented yet")

    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get special tokens mask."""
        if already_has_special_tokens:
            return [
                1 if tid in {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.mask_token_id} 
                else 0 for tid in token_ids_0
            ]
        
        mask = [1]  # BOS
        mask.extend([0] * len(token_ids_0))  # Token sequence
        mask.append(1)  # EOS
        
        if token_ids_1 is not None:
            mask.extend([0] * len(token_ids_1))  # Second sequence
            mask.append(1)  # EOS
            
        return mask

    def create_token_type_ids_from_sequences(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs for sequences."""
        sep = [self.eos_token_id]
        cls = [self.bos_token_id]
        
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build inputs with special tokens."""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        
        return ([self.bos_token_id] + token_ids_0 + [self.eos_token_id] + 
                token_ids_1 + [self.eos_token_id])


class FastChemTokenizerSelfies(FastChemTokenizer):
    """
    SELFIES variant that handles whitespace-separated tokens.
    Uses trie-based longest-match encoding (same as original working version).
    """

    def _encode_core(self, text: str) -> List[int]:
        """Trie-based encoding for SELFIES with fragment + atom vocab."""
        result_ids = []
        i = 0
        n = len(text)

        while i < n:
            if text[i].isspace():  # skip literal whitespace
                i += 1
                continue

            node = self.trie_root
            j = i
            last_match_id = None
            last_match_end = i

            # Traverse trie character by character (including spaces if part of vocab key)
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
                # Fallback: encode one char as unk or atom
                result_ids.append(self.token_to_id.get(text[i], self.unk_token_id))
                i += 1

        return result_ids

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """SELFIES decoding: join tokens with spaces (preserve original format)."""
        return " ".join(tokens)

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> str:
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

        return " ".join(tokens)   # ✅ preserve spaces
