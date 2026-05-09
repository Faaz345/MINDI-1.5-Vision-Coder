# MINDI Model Sources

This file records the working model links found while preparing the MI300X deployment.

## MINDI 1.5

- Hugging Face model/project repo: `Mindigenous/MINDI-1.5-Vision-Coder`
- GitHub repo: `https://github.com/Faaz345/MINDI-1.5-Vision-Coder.git`
- Hugging Face dataset: `Mindigenous/MINDI-1.5-training-data`
- Hugging Face Space: `Mindigenous/mindi-chat`
- Live Space URL: `https://mindigenous-mindi-chat.hf.space`
- Base LLM: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Vision encoder: `openai/clip-vit-large-patch14`

The old `context.md` says the final MINDI 1.5 checkpoint should be under:

```txt
checkpoints/phase3_final/
```

Expected checkpoint layout from the Space loader:

```txt
checkpoints/phase3_final/lora/
checkpoints/phase3_final/vision/
checkpoints/phase3_final/fusion/fusion.pt
data/tokenizer/mindi_tokenizer/
```

Current public repo metadata does not expose `checkpoints/phase3_final/**`.
Use `HF_TOKEN` with access to the private checkpoint files, or copy the checkpoint to
the droplet and set `MINDI_CHECKPOINT_DIR`.

## MINDI 1.0

- Loadable public model repo: `Mindigenous/MINDI-1.0-420M`
- Backup repo: `Mindigenous/mindi-backup`
- Backup archives: `Mindigenous/mindi-backups`

`Mindigenous/MINDI-1.0-420M` has a public `model.safetensors` file and is directly
loadable as a normal Hugging Face model. It is useful for API smoke tests, but it is
not MINDI 1.5.
