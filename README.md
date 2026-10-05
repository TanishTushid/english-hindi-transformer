<div align="center">

# Bhashantara

### English → Hindi Translation with Two Transformer Models

A local web application for translating English text into Hindi. Compare a locally saved, fine-tuned NLLB/M2M-style model with a custom encoder-decoder Transformer trained from scratch.

</div>

## Screenshots

| Transformer from scratch | Fine-tuned NLLB |
| --- | --- |
| ![Bhashantara using the Transformer from scratch](./assests/Screenshot%20(308).png) | ![Bhashantara using the fine-tuned NLLB model](./assests/Screenshot%20(309).png) |

> The screenshot directory is currently named `assests/` in this project. The links above use that existing spelling.

## Features

- English-to-Hindi translation from a browser-based interface.
- Choose between two local models:
  - **Fine-tuned NLLB/M2M-style model** using Hugging Face Transformers and PyTorch.
  - **Transformer from scratch** implemented in TensorFlow/Keras.
- FastAPI backend with a JSON translation API and interactive API documentation.
- Model availability is checked at startup and exposed through health/model endpoints.
- Example sentences, character count, clipboard paste/copy, and browser speech input/output where supported.
- Local model loading: inference does not require downloading model weights from Hugging Face.

## Models

### Fine-tuned NLLB

The saved model and tokenizer are loaded from `models/nllb_final/`. Inference uses the English (`eng_Latn`) source language and Hindi (`hin_Deva`) target language codes, with beam search.

The checkpoint is large (the current `model.safetensors` file is about 2.46 GB). Loading it also requires additional memory for the model and runtime. A machine with sufficient RAM or a supported GPU is recommended. The API loads this model only when a translation is requested.

The multi-gigabyte `model.safetensors` file is not committed to this GitHub repository. To use NLLB after cloning, obtain the matching fine-tuned checkpoint separately and place it at `models/nllb_final/model.safetensors`. The scratch Transformer can be used without that file.

### Transformer trained from scratch

The scratch model is a custom TensorFlow/Keras sequence-to-sequence Transformer:

| Setting | Value |
| --- | ---: |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Model dimension | 128 |
| Attention heads | 4 |
| Feed-forward dimension | 512 |
| English vocabulary | 16,000 |
| Hindi vocabulary | 16,000 |
| Maximum English input length | 56 tokens |
| Maximum Hindi decoding length | 64 tokens |

English and Hindi use separate SentencePiece BPE tokenizers. The current application uses `models/transformer_trained.weights.h5`, `models/transformer_config.json`, `models/english_sp.model`, and `models/hindi_sp.model`.

#### Training and checkpoint note

The notebooks in `notebook/` cover preprocessing, tokenization, model construction, and training. Tokenized `.npz` chunks contain integer token IDs; they do not contain the SentencePiece vocabulary/model files needed to tokenize new text at inference time. The tokenizer models must use the same token-to-ID mappings used when producing the training chunks.

The training checkpoints are TensorFlow checkpoint pairs (`.index` and `.data-*` files) and are not committed. To export a complete checkpoint into the HDF5 weights file used by the app, first obtain and place the checkpoint folder under `models/training_v2-*/training_v2/`, then run:

```powershell
.\.venv\Scripts\python.exe .\export_training_checkpoint.py
```

The exporter checks that a checkpoint contains all model tensors with matching shapes; it skips incomplete checkpoints rather than exporting partial weights. The current exported weights were produced from `ckpt-6`; in the supplied checkpoint set, `ckpt-7` through `ckpt-9` do not contain the complete model weights.

## Project layout

```text
translator/
├── README.md
├── requirements.txt
├── start.py
├── generate_tokenizers.py
├── export_training_checkpoint.py
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── translate_nllb.py
│   └── translate_scratch.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── notebook/
│   ├── 01_dataset_preprocessing.ipynb
│   ├── 02_tokenization.ipynb
│   ├── 03_transformer_from_scratch.ipynb
│   ├── 04_training.ipynb
│   └── english-hindi-nllb-translation-ipynb.ipynb
├── processed/                 # Processed train/validation/test parquet files
├── tokenized/                 # Integer-encoded training chunks (.npz)
├── models/
│   ├── nllb_final/            # Fine-tuned model and tokenizer
│   ├── english_sp.model       # English SentencePiece model
│   ├── hindi_sp.model        # Hindi SentencePiece model
│   ├── transformer_config.json
│   ├── transformer_trained.weights.h5
│   └── training_v2-*/         # TensorFlow training checkpoints
└── assests/                   # UI screenshots (directory name as supplied)
```

## Requirements

- Windows, Linux, or macOS. Python **3.12** is the tested setup for this project.
- Python packages listed in `requirements.txt`.
- Disk space for the virtual environment, datasets, tokenized chunks, and local model files.
- Enough RAM for the selected model. The NLLB checkpoint alone is about 2.46 GB; its runtime memory use is higher.

On native Windows, TensorFlow runs on CPU with the standard `tensorflow-cpu` package. TensorFlow GPU support requires a suitable Linux/WSL setup or a separately configured compatible backend.

## Setup

Run these commands from the project root in PowerShell:

```powershell
# Create a Python 3.12 virtual environment
py -3.12 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install CPU-only PyTorch, then the project dependencies
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If your environment already has Python 3.12 selected, `python -m venv .venv` can be used instead of `py -3.12 -m venv .venv`.

### Verify model files

For the **fine-tuned model**, check that `models/nllb_final/` contains its config, tokenizer files, and `model.safetensors`.

For the **scratch model**, check that these files exist:

```text
models/transformer_trained.weights.h5
models/transformer_config.json
models/english_sp.model
models/hindi_sp.model
```

If the SentencePiece models are missing but the original processed training data is present in `processed/train_processed.parquet`, generate them with:

```powershell
python .\generate_tokenizers.py
```

Only regenerate tokenizers when necessary, and use the same training data and SentencePiece settings as the original tokenization run. If the token-to-ID mapping changes, the existing scratch-model weights and tokenized chunks will no longer correspond to the new tokenizer.

## Run the application

From the project root:

```powershell
python .\start.py
```

Or use the virtual environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe .\start.py
```

Then open:

- Web application: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
- Health and model status: <http://localhost:8000/api/health>

The server reports which models are available at startup. A model is marked unavailable if required packages or files are missing.

## Translation API

### `POST /api/translate`

Request:

```json
{
  "text": "Hello, how are you?",
  "model": "scratch",
  "max_tokens": 128
}
```

- `text`: English input, required, maximum 1,000 characters.
- `model`: `"nllb"` or `"scratch"`; defaults to `"nllb"`.
- `max_tokens`: positive integer from 1 to 1,024; defaults to 256. It controls NLLB generation. The scratch model uses its configured decoding length.

Example using PowerShell:

```powershell
$body = @{
  text = "Hello, how are you?"
  model = "scratch"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/translate" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Successful response:

```json
{
  "translation": "आप कैसे हैं?",
  "model_used": "scratch",
  "time_ms": 1250.0,
  "input_text": "Hello, how are you?"
}
```

Other endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | API status and model availability |
| `GET` | `/api/models` | Model availability and expected artifact paths |
| `POST` | `/api/translate` | Translate text with a selected model |

Invalid input returns HTTP 400 or 422. A requested model that is unavailable returns HTTP 503.

## Data and notebooks

The notebooks document the model-development workflow:

1. `01_dataset_preprocessing.ipynb` — prepare English-Hindi data.
2. `02_tokenization.ipynb` — train separate SentencePiece tokenizers and create tokenized chunks.
3. `03_transformer_from_scratch.ipynb` — define and initialize the custom Transformer.
4. `04_training.ipynb` — train and checkpoint the scratch model.
5. `english-hindi-nllb-translation-ipynb.ipynb` — NLLB translation/fine-tuning notebook workflow.

The local `processed/`, `tokenized/`, and checkpoint data may be large. Keep the original datasets and tokenizer artifacts paired with the weights that were trained using them.

## GitHub and large files

The multi-gigabyte NLLB weights, training checkpoints, processed data, and tokenized chunks are excluded from this repository to keep GitHub clones manageable. The small scratch Transformer inference weights, tokenizer models, and configuration are included. Before publishing or expanding the repository:

- Do not commit `.venv/`.
- Consider Git LFS or an external model/data repository for large weights and datasets.
- Place separately distributed large model/data artifacts at the paths documented above.
- Do not publish dataset files unless their license permits redistribution.

## Troubleshooting

- **A model is unavailable:** Check the startup output and `/api/models`; it includes the missing artifact or dependency.
- **Scratch tokenizer missing:** Put the matching `english_sp.model` and `hindi_sp.model` in `models/`. Tokenized chunks alone are not tokenizer files.
- **Scratch weights missing:** Run `export_training_checkpoint.py` after placing a complete checkpoint in the expected `models/training_v2-*/training_v2/` directory.
- **NLLB load fails or the machine runs out of memory:** Free RAM, use a machine with more memory, or select the scratch model.
- **PowerShell blocks environment activation:** Run the virtual environment's interpreter directly with `.\.venv\Scripts\python.exe`.
- **The browser shows an outdated page/script:** Refresh the page after restarting the server.

## License

No license file is currently included. Add a `LICENSE` file before redistributing or accepting external contributions so the project's usage terms are clear.
