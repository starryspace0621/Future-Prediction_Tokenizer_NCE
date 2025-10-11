
# FPT-MVP: Future-Prediction Tokenizer vs VAE (BAIR)

Minimal PyTorch prototype to compare **VAE** vs **FPT (Future-Prediction Tokenizer)**.

## Steps
1) Prepare dataset and update `configs/*yaml` paths.
2) Train tokenizers: `python train_tokenizer.py --config configs/fpt_64.yaml` (or `vae_64.yaml`)
3) Evaluate models: `python -m eval.evaluate`
