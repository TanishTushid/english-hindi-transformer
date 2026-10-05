"""
Start the English -> Hindi Translator.
Run from the project root:
    python start.py

Then open: http://localhost:8000
"""

import os, sys, subprocess
from pathlib import Path

ROOT        = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"

print("=" * 58)
print("  bhashantara — English to Hindi Translator")
print("=" * 58)
print(f"  Backend : {BACKEND_DIR}")
print(f"  Open    : http://localhost:8000")
print(f"  API docs: http://localhost:8000/docs")
print("=" * 58)
print()

# Check if tokenizers need to be generated
en_tok = ROOT / "models" / "english_sp.model"
hi_tok = ROOT / "models" / "hindi_sp.model"

if not en_tok.exists() or not hi_tok.exists():
    print("   SentencePiece tokenizers missing.")
    print("     The Scratch Transformer model requires them.")
    print("     Run this first to enable it:")
    print("       python generate_tokenizers.py")
    print("  (The NLLB model will still work without them)\n")

try:
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        check=True,
    )
except KeyboardInterrupt:
    print("\n  Server stopped.")
