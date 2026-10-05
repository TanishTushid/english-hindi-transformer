"""
Inference using fine-tuned NLLB model stored locally in models/nllb_final/
Works with transformers 5.x and uses your locally fine-tuned weights.
"""

import os

# Suppress verbose TF/tokenizer warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from transformers import AutoTokenizer


class NLLBTranslator:
    """
    Loads your fine-tuned NLLB model from the local nllb_final directory.
    Uses PyTorch backend via HuggingFace transformers.
    """

    def __init__(self, model_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        from transformers import AutoModelForSeq2SeqLM

        print(f"[NLLB] Loading tokenizer from {model_path} ...")
        # NllbTokenizer is the correct class for NLLB models
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load NLLB tokenizer: {e}")

        print(f"[NLLB] Loading fine-tuned model weights from {model_path} ...")
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load NLLB model: {e}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"[NLLB] Model ready on {self.device}")

        # NLLB language codes
        self.src_lang = "eng_Latn"
        self.tgt_lang = "hin_Deva"

    def translate(self, text: str, max_new_tokens: int = 256) -> str:
        self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        forced_bos = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if forced_bos is None or forced_bos == self.tokenizer.unk_token_id:
            raise RuntimeError(
                f"Target language token {self.tgt_lang!r} is not in the tokenizer."
            )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
