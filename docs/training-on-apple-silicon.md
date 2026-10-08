# Training the bi-encoder on Apple Silicon (Mac Studio / MacBook Pro)

Every fine-tune in `benchmarks/results/router-eval-*.md` ran on a 4-core CPU container with a
15 to 60 minute budget, and validation was still rising when the budget ran out. The same scripts
run unchanged on an Apple Silicon Mac through PyTorch's MPS backend, which is roughly an order of
magnitude faster for these 33M to 335M parameter encoders. Unsloth is not an option here: it is
CUDA-only and targets decoder LLMs, not BERT-style encoders.

## One-time setup

```bash
git clone https://github.com/josaum/olaya && cd olaya
uv venv && uv pip install -e '.[dev]'          # torch wheels on macOS ship with MPS
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # must print True
```

The encoders are downloaded from the Hugging Face Hub on first use (about 2 GB for all seven). The
scripts never force offline mode; set `HF_HUB_OFFLINE=1` yourself once everything is cached if you want
runs to fail fast instead of touching the network. Pre-fetch them in one go:

```bash
.venv/bin/python - <<'EOF'
from transformers import AutoModel, AutoTokenizer
for m in ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
          "intfloat/e5-small-v2", "intfloat/e5-large-v2",
          "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]:
    AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
EOF
```

## Device selection

`olaya.embeddings.TransformerEncoder` takes `device=` (`cpu`, `mps`, `cuda`, `auto`) and defaults to
the `OLAYA_ENCODER_DEVICE` environment variable, so every script below picks up MPS with:

```bash
export OLAYA_ENCODER_DEVICE=mps      # or auto
export OLAYA_TORCH_THREADS=8         # host-side tokenisation / numpy
```

An explicitly requested device that is not available is an error, never a silent CPU fallback.

## Runs worth doing

Each command writes its report under `benchmarks/results/` and its checkpoint under
`models/checkpoints/` (gitignored; `MANIFEST.json` records hashes and metrics). Commit the reports.

1. **Joint encoder, longer budget, bigger base.** The CPU run (`router-eval-joint.md`) stopped at
   epoch 4 with validation still improving. On MPS, 60 minutes covers 20 epochs of bge-base:

   ```bash
   .venv/bin/python scripts/finetune_joint_encoder.py --base bge-base --minutes 60 --epochs 20 --mine 3 --tag joint-base
   .venv/bin/python scripts/finetune_joint_encoder.py --base bge-large --minutes 90 --epochs 20 --mine 3 --tag joint-large
   ```

2. **The same on the class-disjoint split**, which is the honest transfer number:

   ```bash
   .venv/bin/python scripts/finetune_joint_encoder.py --base bge-base --minutes 60 --epochs 20 --mine 3 --tag joint-base --split class
   ```

3. **Per-domain reports with every encoder**, so the zero-shot table and the per-domain fine-tune are
   refreshed with the full budget:

   ```bash
   .venv/bin/python scripts/evaluate_routers.py --finetune-minutes 30 --finetune-base bge-base
   .venv/bin/python scripts/evaluate_routers.py --split class --finetune-minutes 30 --finetune-base bge-base
   ```

4. **Evaluate a checkpoint you already have** inside the per-domain protocol:

   ```bash
   .venv/bin/python scripts/evaluate_routers.py --encoders bge-small --encoder-checkpoint models/checkpoints/olaya_embed_joint-base.pt
   ```

5. **Commit the reports** (list the files; zsh rejects an unmatched brace glob):

   ```bash
   git add benchmarks/results/router-eval-joint-base.json benchmarks/results/router-eval-joint-base.md \
           benchmarks/results/router-eval-joint-base-classdisjoint.json benchmarks/results/router-eval-joint-base-classdisjoint.md
   git commit -m "results(router): joint bge-base on Apple Silicon (MPS)"
   ```

## What to expect

On the CPU runs the joint bge-small encoder with mined hard negatives reached full-ontology top-1
0.75 / 0.66 / 0.63 on incident / GS1 / Schema.org (batch split). The remaining headroom is in the
scorer's parent-versus-child discrimination (the `[routed]` engine rows show recall@5 rising while
top-1 stays flat), which is what more epochs, more mined negatives and a larger base address. Labels
are LLM-distilled, not human; see `decision-head-summary.md` for what that does and does not
establish.

## Running it from a Claude Code session on the Mac

Open the repository in the Claude Desktop app, or run `claude remote-control` in the repository
folder; the session then appears in the Claude Code app and can run the commands above directly.
