"""Export the latest complete scratch-Transformer training checkpoint for inference."""

import re
from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.model import load_transformer


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CONFIG_PATH = MODELS_DIR / "transformer_config.json"
INITIAL_WEIGHTS_PATH = MODELS_DIR / "transformer_initial.weights.h5"
OUTPUT_PATH = MODELS_DIR / "transformer_trained.weights.h5"
TRAINABLE_VARIABLE_TEMPLATE = (
    "optimizer/_trainable_variables/{index}/.ATTRIBUTES/VARIABLE_VALUE"
)


def checkpoint_number(index_path: Path) -> int:
    match = re.fullmatch(r"ckpt-(\d+)\.index", index_path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint index filename: {index_path.name}")
    return int(match.group(1))


def main() -> None:
    model, _ = load_transformer(str(INITIAL_WEIGHTS_PATH), str(CONFIG_PATH))
    candidates = []
    for checkpoint_dir in MODELS_DIR.glob("training_v2-*/training_v2"):
        candidates.extend(checkpoint_dir.glob("ckpt-*.index"))
    candidates.sort(key=checkpoint_number, reverse=True)

    for index_path in candidates:
        checkpoint_path = index_path.with_suffix("")
        reader = tf.train.load_checkpoint(str(checkpoint_path))
        checkpoint_shapes = {
            name: shape for name, shape in tf.train.list_variables(str(checkpoint_path))
        }
        variable_keys = [
            TRAINABLE_VARIABLE_TEMPLATE.format(index=index)
            for index in range(len(model.trainable_variables))
        ]
        if not all(key in checkpoint_shapes for key in variable_keys):
            continue
        if any(
            checkpoint_shapes[key] != variable.shape.as_list()
            for key, variable in zip(variable_keys, model.trainable_variables)
        ):
            continue

        values = [reader.get_tensor(key) for key in variable_keys]
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError(f"Checkpoint contains non-finite weights: {checkpoint_path}")
        for variable, value in zip(model.trainable_variables, values):
            variable.assign(value)

        model.save_weights(str(OUTPUT_PATH))
        print(f"Exported complete checkpoint {checkpoint_path} to {OUTPUT_PATH}")
        print(f"Restored {len(variable_keys)} trainable tensors.")
        return

    raise FileNotFoundError(
        "No complete training checkpoint matching this model configuration was found "
        f"under {MODELS_DIR}."
    )


if __name__ == "__main__":
    main()
