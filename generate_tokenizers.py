"""
Generate SentencePiece tokenizer models from processed parquet data.
Run ONCE from the project root before using the scratch transformer:
    python generate_tokenizers.py
"""

import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DIR = os.path.join(BASE, "processed")
MODELS_DIR    = os.path.join(BASE, "models")

ENG_SP_OUT    = os.path.join(MODELS_DIR, "english_sp")
HIN_SP_OUT    = os.path.join(MODELS_DIR, "hindi_sp")
ENG_TXT       = os.path.join(MODELS_DIR, "_english_tokenizer.txt")
HIN_TXT       = os.path.join(MODELS_DIR, "_hindi_tokenizer.txt")
TRAIN_PARQUET = os.path.join(PROCESSED_DIR, "train_processed.parquet")


def main():
    print("=" * 58)
    print("  SentencePiece Tokenizer Generator")
    print("=" * 58)

    if os.path.exists(ENG_SP_OUT + ".model") and os.path.exists(HIN_SP_OUT + ".model"):
        print("  Tokenizer models already exist — skipping.")
        print(f"  {ENG_SP_OUT}.model")
        print(f"  {HIN_SP_OUT}.model")
        return

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pip install pandas pyarrow")
        sys.exit(1)

    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: pip install sentencepiece")
        sys.exit(1)

    if not os.path.exists(TRAIN_PARQUET):
        print(f"ERROR: {TRAIN_PARQUET} not found")
        sys.exit(1)

    print(f"\n[1/4] Loading {TRAIN_PARQUET} ...")
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"      {len(df):,} pairs. Columns: {df.columns.tolist()}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[2/4] Writing English text ...")
    df["english"].dropna().to_csv(ENG_TXT, index=False, header=False, encoding="utf-8")

    print("[3/4] Writing Hindi text ...")
    df["hindi"].dropna().to_csv(HIN_TXT, index=False, header=False, encoding="utf-8")

    print("[4a/4] Training English BPE tokenizer (16k vocab) ...")
    spm.SentencePieceTrainer.train(
        input=ENG_TXT, model_prefix=ENG_SP_OUT,
        vocab_size=16000, model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    print("[4b/4] Training Hindi BPE tokenizer (16k vocab) ...")
    spm.SentencePieceTrainer.train(
        input=HIN_TXT, model_prefix=HIN_SP_OUT,
        vocab_size=16000, model_type="bpe", character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    os.remove(ENG_TXT)
    os.remove(HIN_TXT)

    en = spm.SentencePieceProcessor(model_file=ENG_SP_OUT + ".model")
    hi = spm.SentencePieceProcessor(model_file=HIN_SP_OUT + ".model")
    print(f"\n[OK] EN vocab={en.get_piece_size()}, HI vocab={hi.get_piece_size()}")
    print("    Tokenizers saved to models/")
    print("\n Run: python start.py")

if __name__ == "__main__":
    main()
