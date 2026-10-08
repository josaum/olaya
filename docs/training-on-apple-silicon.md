# Training the bi-encoder on Apple Silicon (Mac Studio / MacBook Pro)

Every fine-tune in `benchmarks/results/router-eval-*.md` ran on a 4-core CPU container with a
15 to 60 minute budget, and validation was still rising when the budget ran out. The same scripts
run unchanged on an Apple Silicon Mac through PyTorch's MPS backend, which is roughly an order of
magnitude faster for these 33M to 335M parameter encoders. Unsloth is not an option here: it is
CUDA-only and targets decoder LLMs, not BERT-style encoders.

## One-time setup

```bash
git clone https://github.com/josaum/olaya && cd olaya
uv venv && uv pip install -e '.[dev,gemma]'    # torch wheels on macOS ship with MPS; gemma = the default fine-tune base
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

## EmbeddingGemma 2 (Google, 270M text backbone) is the default fine-tune

`google/embeddinggemma-2` is a prompted model served through `sentence-transformers`, so it needs the
`gemma` extra and a newer `transformers`:

```bash
uv pip install -e '.[dev,gemma]'      # sentence-transformers[image] >= 6.1, transformers >= 5.19, peft >= 0.21 (the audio extra pulls kenlm, which does not build on Python 3.14)
```

`olaya.embeddings.SentenceTransformerEncoder` loads it text-only (vision and audio encoders dropped,
271M parameters), applies the `query` prompt to states and the `document` prompt to class texts, and
exposes the differentiable forward the fine-tune uses. It is the default `--base` of
`scripts/finetune_joint_encoder.py` and the default `--finetune-base` of `scripts/evaluate_routers.py`,
and the joint script picks the device automatically (CUDA, then MPS, then CPU), so the two protocol
runs need no flags:

```bash
.venv/bin/python scripts/finetune_joint_encoder.py --tag joint-gemma
.venv/bin/python scripts/finetune_joint_encoder.py --tag joint-gemma --split class
```

### The recipe (`olaya.embedding_router.finetune_bi_encoder`)

- **LoRA rank 16** on the attention and MLP projections (5M trainable parameters of 271M), merged
  into the weights before saving, so the checkpoint keeps the plain `olaya-st-encoder-v1` format.
  `--lora-rank 0` trains the full weights instead.
- **float32 weights.** The Hub checkpoint is bf16; AdamW steps at these learning rates are smaller
  than bf16 can represent, so most of them would round away. The dtype is stored in the checkpoint
  and restored on reload, otherwise the fine-tuned deltas would be rounded away a second time.
- **AdamW 3e-5**, weight decay 0.01 on matrices only, linear warmup over 10% of the planned steps,
  linear decay to zero at `--epochs` (default 8).
- **Early stopping** after `--patience 3` epochs without a validation gain, and the zero-shot weights
  are the candidate every epoch has to beat: a run that only hurts returns the base model
  (`best epoch 0`).
- **Gradient checkpointing on MPS.** One step is 16 queries plus up to ~96 class texts through 24
  layers; without recomputation it needs ~35 GB on MPS and swaps instead of training. With it a
  step takes ~11 GB and 3.8 s, about 3.5 minutes per epoch.

### How the recipe was chosen

Class-disjoint validation (mean full-ontology top-1 over the three domains, 819 train pairs, zero-shot
0.564), one run per row, selected on validation only:

| recipe | epoch 1 | epoch 2 | epoch 3 | best |
|---|---|---|---|---|
| full weights, 2e-5 | 0.480 | 0.397 | 0.472 | base kept |
| full weights, 2e-5, token embeddings frozen | 0.485 | 0.375 | 0.451 | base kept |
| full weights, 5e-6 | 0.657 | 0.482 | 0.461 | 0.657 |
| full weights, 5e-6, token embeddings frozen | 0.659 | 0.504 | 0.434 | 0.659 |
| full weights, 2e-6 | 0.674 | 0.636 | 0.617 | 0.674 |
| LoRA 16, 3e-4 | 0.672 | 0.557 | 0.527 | 0.672 |
| LoRA 16, 1e-4 | **0.724** | 0.642 | 0.539 | 0.724 |
| LoRA 16, 3e-5 (default) | 0.670 | **0.722** | 0.714 | 0.722 (0.700 and 0.673 at epochs 4 and 5) |
| LoRA 8, 3e-5 | 0.655 | 0.693 | 0.711 | 0.723 at epoch 4 (0.719, 0.715, 0.703 after) |
| LoRA 32, 3e-5 | 0.691 | 0.695 | 0.669 | 0.695 (0.563 and 0.536 at epochs 4 and 5) |

Full fine-tuning memorises the 819 pairs within two epochs at any rate that moves the model: the peak
is always epoch 1 and a lower rate only raises it and slows the decline. Freezing the 134M-parameter
token embedding matrix changes nothing. LoRA raises the ceiling, and at 3e-5 it reaches it on a
plateau rather than a one-epoch spike, which is why it is the default even though 1e-4 ties its peak.
The earlier bf16 full fine-tune that reported 0.676 at epoch 1 was this effect by accident: most of
its updates rounded to zero.

Rank does not move the ceiling: rank 8 reaches the same 0.72 two epochs later, and rank 32 peaks at
0.695 and then collapses the way full fine-tuning does. Rank 16 is the smallest adapter that reaches
the ceiling by epoch 2.

### Results with the default recipe

- Batch-disjoint (`router-eval-joint-gemma.md`): best epoch 8 of 8, mean validation top-1 0.780 (zero-shot 0.593); test top-1 gs1 0.797, incident 0.812, schemaorg 0.750. The bf16 full fine-tune this replaces scored 0.797 / 0.812 / 0.783, so seen-class accuracy is unchanged within noise and the curve is now monotone instead of a spike at epoch 5.
- Class-disjoint (`router-eval-joint-gemma-classdisjoint.md`): best epoch 2, stopped on patience after 5, mean validation top-1 0.722 (zero-shot 0.564); test top-1 gs1 0.656, incident 0.826, schemaorg 0.606, against 0.607 / 0.652 / 0.591 for the bf16 full fine-tune and 0.410 / 0.696 / 0.402 for the joint bge-small encoder.

- Per-domain fine-tunes with the same default (`scripts/evaluate_routers.py --finetune-minutes 60`,
  `router-eval-<domain>.md`): batch-disjoint test top-1 gs1 0.859, incident 0.812, schemaorg 0.642 (the
  bge-small and MiniLM per-domain references were 0.609 / 0.688 / 0.558); class-disjoint 0.557 / 0.739 / 0.598
  (were 0.459 / 0.696 / 0.417). gs1 does better per-domain than jointly; the incident class-disjoint
  fine-tune beats zero-shot on its 14-row validation set and loses to it on the 23-row test set, which is
  the noise floor of that domain. Each per-domain run takes 1 to 10 minutes on MPS.

Checkpoints are 1.08 GB in float32 and stay gitignored.

## What to expect

On the CPU runs the joint bge-small encoder with mined hard negatives reached full-ontology top-1
0.81 / 0.66 / 0.64 on incident / GS1 / Schema.org (batch split). The first Mac Studio run (joint
bge-base, MPS, 55 s per epoch, best epoch 3) reached 0.75 / 0.67 / 0.66 and, on the class-disjoint
split, 0.65 / 0.57 / 0.36: a clear transfer gain on GS1 over the per-domain fine-tune (0.46) and a
loss on Schema.org (0.42). See `router-eval-joint-base.md` and `router-eval-joint-base-classdisjoint.md`. The remaining headroom is in the
scorer's parent-versus-child discrimination (the `[routed]` engine rows show recall@5 rising while
top-1 stays flat), which is what more epochs, more mined negatives and a larger base address. Labels
are LLM-distilled, not human; see `decision-head-summary.md` for what that does and does not
establish.

## Running it from a Claude Code session on the Mac

Open the repository in the Claude Desktop app, or run `claude remote-control` in the repository
folder; the session then appears in the Claude Code app and can run the commands above directly.
