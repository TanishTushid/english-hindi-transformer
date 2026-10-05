"""
Inference using the custom scratch transformer + SentencePiece tokenizers.
Model: transformer_trained.weights.h5
Tokenizers: models/english_sp.model, models/hindi_sp.model
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import sentencepiece as spm

from model import load_transformer, create_masks


class ScratchTranslator:
    """
    Runs your hand-built Transformer model from scratch.
    Requires SentencePiece tokenizers trained on the same data.
    """

    def __init__(self, weights_path: str, config_path: str,
                 eng_sp_model: str, hin_sp_model: str):
        print(f"[Scratch] Loading model from {weights_path} ...")
        self.model, self.config = load_transformer(weights_path, config_path)

        print(f"[Scratch] Loading English tokenizer from {eng_sp_model} ...")
        self.en_tok = spm.SentencePieceProcessor(model_file=eng_sp_model)

        print(f"[Scratch] Loading Hindi tokenizer from {hin_sp_model} ...")
        self.hi_tok = spm.SentencePieceProcessor(model_file=hin_sp_model)

        self.PAD_ID    = self.config["pad_id"]
        self.BOS_ID    = self.config["bos_id"]
        self.EOS_ID    = self.config["eos_id"]
        self.MAX_ENG   = self.config["max_english_len"]
        self.MAX_HIN   = self.config["max_hindi_len"]

        print(f"[Scratch] Ready. max_eng={self.MAX_ENG}, max_hin={self.MAX_HIN}")

    def _pad(self, tokens: list, max_len: int) -> list:
        tokens = tokens[:max_len]
        tokens += [self.PAD_ID] * (max_len - len(tokens))
        return tokens

    def translate(self, text: str) -> str:
        # Encode English input
        enc_tokens = self.en_tok.encode(text, out_type=int)
        enc_tokens = self._pad(enc_tokens, self.MAX_ENG)
        enc_inp = tf.constant([enc_tokens], dtype=tf.int32)

        # Greedy autoregressive decode
        dec_tokens = [self.BOS_ID]

        for _ in range(self.MAX_HIN - 1):
            dec_inp = tf.constant([dec_tokens], dtype=tf.int32)
            enc_mask, dec_mask, cross_mask = create_masks(enc_inp, dec_inp)

            logits, _, _, _ = self.model(
                enc_inp, dec_inp, training=False,
                encoder_padding_mask=enc_mask,
                decoder_look_ahead_mask=dec_mask,
                decoder_padding_mask=cross_mask,
            )

            next_token = int(tf.argmax(logits[0, -1]).numpy())

            if next_token == self.EOS_ID:
                break
            dec_tokens.append(next_token)

        output_ids = dec_tokens[1:]  # Remove BOS
        return self.hi_tok.decode(output_ids)
