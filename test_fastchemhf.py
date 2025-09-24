"""
HF-compatibility test-suite for FastChemTokenizer & FastChemTokenizerSelfies
Author : gbyuvd
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BatchEncoding

from FastChemTokenizerHF import FastChemTokenizer, FastChemTokenizerSelfies


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _dummy_vocab() -> dict[str, int]:
    """Return a toy chemical vocab with all HF special tokens."""
    return {
        "<s>": 0,
        "</s>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "<mask>": 4,
        "C": 5,
        "O": 6,
        "N": 7,
        "=": 8,
        "(": 9,
        ")": 10,
        "[C@@H]": 11,   # multi-char token
        "c1ccccc1": 12, # ring token
    }


def _smiles() -> str:
    return "c1ccccc1[C@@H](O)N"


def _selfies() -> str:
    return "[C] [C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1] [C@H1] [Branch1] [C] [O] [N]"


# ------------------------------------------------------------------
# Test-cases for SMILES
# ------------------------------------------------------------------
class TestFastChemTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        vocab_path = Path(cls.tmpdir.name) / "vocab.json"
        vocab_path.write_text(json.dumps(_dummy_vocab()))
        cls.tokenizer = FastChemTokenizer(vocab_file=str(vocab_path))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    # --------------- HF core contracts ----------------
    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, len(_dummy_vocab()))

    def test_len(self):
        self.assertEqual(len(self.tokenizer), len(_dummy_vocab()))

    def test_get_vocab(self):
        self.assertEqual(self.tokenizer.get_vocab(), _dummy_vocab())

    def test_convert_tokens_to_ids(self):
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("C"), 5)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("[C@@H]"), 11)

    def test_convert_ids_to_tokens(self):
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(5), "C")
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(11), "[C@@H]")

    def test_convert_tokens_to_string(self):
        tokens = ["c1ccccc1", "[C@@H]", "(", "O", ")", "N"]
        self.assertEqual(self.tokenizer.convert_tokens_to_string(tokens),
                         "c1ccccc1[C@@H](O)N")

    # --------------- Encoding / decoding ---------------
    def test_encode_single(self):
        ids = self.tokenizer.encode(_smiles(), add_special_tokens=False)
        self.assertEqual(ids, [12, 11, 9, 6, 10, 7])  # golden

    def test_encode_plus_defaults(self):
        out: BatchEncoding = self.tokenizer(_smiles())
        self.assertIn("input_ids", out)
        self.assertIn("attention_mask", out)
        self.assertIn("token_type_ids", out)
        # should start with <s> and end with </s>
        self.assertEqual(out.input_ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(out.input_ids[-1], self.tokenizer.eos_token_id)

    def test_decode(self):
        ids = [0, 12, 11, 9, 6, 10, 7, 1]  # with <s> & </s>
        txt = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(txt, _smiles())

    def test_batch_encode_decode(self):
        smiles = [_smiles(), "CO"]
        batch: BatchEncoding = self.tokenizer(smiles, padding=True,
                                              truncation=True,
                                              return_tensors="pt")
        self.assertEqual(batch.input_ids.dim(), 2)
        self.assertEqual(batch.input_ids.shape[0], 2)
        # round-trip
        decoded = self.tokenizer.batch_decode(batch.input_ids,
                                              skip_special_tokens=True)
        self.assertEqual(decoded, smiles)

    # --------------- Special tokens ------------------
    def test_special_tokens_mask(self):
        ids = self.tokenizer.encode(_smiles(), add_special_tokens=True)
        mask = self.tokenizer.get_special_tokens_mask(ids,
                                                     already_has_special_tokens=True)
        self.assertEqual(sum(mask), 2)  # <s> & </s>

    # --------------- Persistence ---------------------
    def test_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.tokenizer.save_pretrained(tmp)
            self.assertTrue(Path(tmp, "vocab.json").exists())
            self.assertTrue(Path(tmp, "tokenizer_config.json").exists())

            tok2 = FastChemTokenizer.from_pretrained(tmp)
            self.assertEqual(tok2.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(tok2.encode(_smiles()),
                           self.tokenizer.encode(_smiles()))

    # --------------- Misc extras ---------------------
    def test_decode_with_trace(self):
        # simply ensure no crash
        ids = self.tokenizer.encode(_smiles())
        self.tokenizer.decode_with_trace(ids)

    def test_max_token_len_property(self):
        self.assertEqual(self.tokenizer.max_token_len, 8)  # "c1ccccc1"


# ------------------------------------------------------------------
# Test-cases for SELFIES
# ------------------------------------------------------------------
class TestFastChemTokenizerSelfies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        vocab_path = Path(cls.tmpdir.name) / "vocab.json"
        # Build a tiny SELFIES vocab
        selfies_vocab = _dummy_vocab()
        selfies_vocab.update({
            "[C]": 13,
            "[=C]": 14,
            "[Ring1]": 15,
            "[=Branch1]": 16,
            "[C@H1]": 17,
            "[Branch1]": 18,
            "[O]": 19,
            "[N]": 20,
        })
        vocab_path.write_text(json.dumps(selfies_vocab))
        cls.tokenizer = FastChemTokenizerSelfies(vocab_file=str(vocab_path))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_selfies_whitespace_handling(self):
        selfies = _selfies()
        ids = self.tokenizer.encode(selfies, add_special_tokens=False)
        # Ensure whitespace does not cause <unk>
        self.assertNotIn(self.tokenizer.unk_token_id, ids)

    def test_selfies_roundtrip(self):
        selfies = _selfies()
        ids = self.tokenizer.encode(selfies)
        reconstructed = self.tokenizer.decode(ids, skip_special_tokens=True)
        # Roundtrip should preserve whitespace-separated format
        self.assertEqual(reconstructed, selfies)


# ------------------------------------------------------------------
# Allow `python test_fast_chem_tokenizer.py`
# ------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
