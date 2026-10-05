"""
Custom Transformer Model (from scratch) - TensorFlow/Keras
Reconstructed from: 03_transformer_from_scratch.ipynb
English -> Hindi translation
"""

import tensorflow as tf
import numpy as np
import json


# ──────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────

def positional_encoding(max_length: int, d_model: int):
    positions = np.arange(max_length)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
    angle_radians = positions * angle_rates
    encoding = np.zeros((max_length, d_model))
    encoding[:, 0::2] = np.sin(angle_radians[:, 0::2])
    encoding[:, 1::2] = np.cos(angle_radians[:, 1::2])
    return tf.cast(encoding, dtype=tf.float32)


# ──────────────────────────────────────────────
# Layers
# ──────────────────────────────────────────────

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x


def scaled_dot_product_attention(query, key, value, mask=None):
    scores = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    scores = scores / tf.math.sqrt(depth)
    if mask is not None:
        scores += mask * -1e9
    weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(weights, value)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.output_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        bs = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), bs)
        k = self.split_heads(self.wk(k), bs)
        v = self.split_heads(self.wv(v), bs)
        attn, weights = scaled_dot_product_attention(q, k, v, mask)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (bs, -1, self.d_model))
        return self.output_dense(attn), weights


def feed_forward_network(d_model, d_ff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        out, w = self.self_attention(x, x, x, mask)
        out = self.dropout1(out, training=training)
        x = self.norm1(x + out)
        f = self.dropout2(self.ffn(x), training=training)
        return self.norm2(x + f), w


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_length, dropout_rate):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        x = self.dropout(x, training=training)
        attention_maps = []
        for layer in self.encoder_layers:
            x, w = layer(x, training=training, mask=mask)
            attention_maps.append(w)
        return x, attention_maps


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, d_ff)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_out, training=False, look_ahead_mask=None, padding_mask=None):
        s, sw = self.self_attention(x, x, x, look_ahead_mask)
        s = self.dropout1(s, training=training)
        out1 = self.norm1(x + s)
        c, cw = self.cross_attention(out1, enc_out, enc_out, padding_mask)
        c = self.dropout2(c, training=training)
        out2 = self.norm2(out1 + c)
        f = self.dropout3(self.ffn(out2), training=training)
        return self.norm3(out2 + f), sw, cw


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_length, dropout_rate):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, enc_out, training=False, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        x = self.dropout(x, training=training)
        self_att_maps, cross_att_maps = [], []
        for layer in self.decoder_layers:
            x, sw, cw = layer(x, enc_out, training=training,
                              look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            self_att_maps.append(sw)
            cross_att_maps.append(cw)
        return x, self_att_maps, cross_att_maps


class Transformer(tf.keras.Model):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads, d_ff,
                 input_vocab_size, target_vocab_size, input_max_length, target_max_length, dropout_rate):
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff,
                               input_vocab_size, input_max_length, dropout_rate)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff,
                               target_vocab_size, target_max_length, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, enc_inp, dec_inp, training=False, encoder_padding_mask=None,
             decoder_look_ahead_mask=None, decoder_padding_mask=None):
        enc_out, enc_att = self.encoder(enc_inp, training=training, mask=encoder_padding_mask)
        dec_out, dec_self_att, dec_cross_att = self.decoder(
            dec_inp, enc_out, training=training,
            look_ahead_mask=decoder_look_ahead_mask, padding_mask=decoder_padding_mask)
        logits = self.final_layer(dec_out)
        return logits, enc_att, dec_self_att, dec_cross_att


# ──────────────────────────────────────────────
# Mask Utilities
# ──────────────────────────────────────────────

def create_padding_mask(sequence, pad_id=0):
    mask = tf.cast(tf.equal(sequence, pad_id), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_masks(encoder_inputs, decoder_inputs):
    enc_pad = create_padding_mask(encoder_inputs)
    dec_pad = create_padding_mask(encoder_inputs)
    dec_target_pad = create_padding_mask(decoder_inputs)
    tgt_len = tf.shape(decoder_inputs)[1]
    look_ahead = create_look_ahead_mask(tgt_len)
    combined = tf.maximum(dec_target_pad, look_ahead[tf.newaxis, tf.newaxis, :, :])
    return enc_pad, combined, dec_pad


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

def load_transformer(weights_path, config_path):
    with open(config_path) as f:
        config = json.load(f)
    model = Transformer(
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        input_vocab_size=config["english_vocab_size"],
        target_vocab_size=config["hindi_vocab_size"],
        input_max_length=config["max_english_len"],
        target_max_length=config["max_hindi_len"],
        dropout_rate=config["dropout_rate"],
    )
    # Build with dummy data
    dummy_enc = tf.zeros((1, config["max_english_len"]), dtype=tf.int32)
    dummy_dec = tf.zeros((1, config["max_hindi_len"]), dtype=tf.int32)
    enc_mask, dec_mask, cross_mask = create_masks(dummy_enc, dummy_dec)
    model(dummy_enc, dummy_dec, training=False,
          encoder_padding_mask=enc_mask,
          decoder_look_ahead_mask=dec_mask,
          decoder_padding_mask=cross_mask)
    model.load_weights(weights_path)
    return model, config
