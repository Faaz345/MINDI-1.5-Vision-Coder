# AMD MI300X Deployment

This deploys MINDI on the AMD Developer Cloud `vLLM 0.17.1 on Ubuntu 24.04` image as an OpenAI-compatible API.

There are two paths:

- `setup_mindi15_openai_server.sh`: correct path for current MINDI 1.5 Vision-Coder checkpoints, because the model is Qwen2.5-Coder + LoRA + CLIP/fusion code, not a plain vLLM-native checkpoint.
- `setup_mi300x_vllm.sh`: use this only for a vLLM-compatible Hugging Face model path, a merged MINDI causal-LM export, or smoke tests with normal HF models.

See `MINDI_MODEL_SOURCES.md` for the repo links discovered from the old project docs.

## 1. SSH

```bash
ssh root@YOUR_DROPLET_PUBLIC_IP
```

Update once before serving:

```bash
apt-get update
apt-get upgrade -y
reboot
```

Reconnect after reboot.

## 2. MINDI 1.5 Checkpoint Access

The old MINDI 1.5 Space loader expects:

```txt
checkpoints/phase3_final/lora/
checkpoints/phase3_final/vision/
checkpoints/phase3_final/fusion/fusion.pt
data/tokenizer/mindi_tokenizer/
```

If those files are private on Hugging Face, set `HF_TOKEN` before setup:

```bash
export HF_TOKEN=hf_your_token_with_model_access
```

If you already have the checkpoint locally, copy it to the droplet:

```powershell
ssh root@YOUR_DROPLET_PUBLIC_IP "mkdir -p /shared-docker/models/mindi-1.5"
scp -r "D:\path\to\phase3_final_parent" root@YOUR_DROPLET_PUBLIC_IP:/shared-docker/models/mindi-1.5
```

Then run setup with:

```bash
sudo MINDI_CHECKPOINT_DIR=/shared-docker/models/mindi-1.5 ./setup_mindi15_openai_server.sh
```

## 3. Run The MINDI 1.5 OpenAI-Compatible Server

From your local project:

```powershell
scp -r "D:\Desktop 31st Jan 2026\MINDIGENOUS 2.0\mindi-core\deploy\amd-mi300x" root@YOUR_DROPLET_PUBLIC_IP:/opt/mindigenous-amd-mi300x
```

On the droplet:

```bash
cd /opt/mindigenous-amd-mi300x
chmod +x setup_mindi15_openai_server.sh smoke_test.sh
sudo HF_TOKEN="$HF_TOKEN" ./setup_mindi15_openai_server.sh
```

The script creates:

- `/etc/mindigenous/mindi-openai.env`
- `/etc/systemd/system/mindi-openai.service`
- `/shared-docker/mindigenous/mindi15_openai_server.py`

Check logs:

```bash
journalctl -u mindi-openai -f
```

## 4. Optional vLLM-Native Path

The AMD image mounts host `/shared-docker` into the `rocm` container. Put a vLLM-compatible model there.

Local model upload from Windows:

```powershell
ssh root@YOUR_DROPLET_PUBLIC_IP "mkdir -p /shared-docker/models"
scp -r "D:\path\to\MINDI-1.5" root@YOUR_DROPLET_PUBLIC_IP:/shared-docker/models/mindi-1.5
```

Or use a vLLM-compatible model repository id:

```bash
export HF_TOKEN=your_token_if_needed
export MINDI_MODEL_PATH=your-org/mindi-1.5
```

### Run vLLM Setup

From your local project:

```powershell
scp -r "D:\Desktop 31st Jan 2026\MINDIGENOUS 2.0\mindi-core\deploy\amd-mi300x" root@YOUR_DROPLET_PUBLIC_IP:/opt/mindigenous-amd-mi300x
```

On the droplet:

```bash
cd /opt/mindigenous-amd-mi300x
chmod +x setup_mi300x_vllm.sh smoke_test.sh
sudo MINDI_MODEL_PATH=/shared-docker/models/mindi-1.5 ./setup_mi300x_vllm.sh
```

The script creates:

- `/etc/mindigenous/mindi-vllm.env`
- `/etc/systemd/system/mindi-vllm.service`

Check logs:

```bash
journalctl -u mindi-vllm -f
```

## 5. Test On Droplet

```bash
source /etc/mindigenous/mindi-openai.env
./smoke_test.sh http://127.0.0.1:8000/v1 "$MINDI_API_KEY" mindi-1.5
```

For the optional vLLM service, source `/etc/mindigenous/mindi-vllm.env` instead.

## 6. Connect Local Backend Safely

Recommended for development: use an SSH tunnel instead of exposing port `8000`.

On Windows:

```powershell
ssh -L 9000:127.0.0.1:8000 root@YOUR_DROPLET_PUBLIC_IP
```

Then set `mindi-core/.env` locally:

```env
MINDI_API_URL=http://127.0.0.1:9000/v1
MINDI_API_KEY=the_key_printed_by_setup_script
MINDI_MODEL=mindi-1.5
```

Run local backend:

```bash
cd mindi-core
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

## Production Notes

- Do not expose `8000` publicly without firewall allowlisting or HTTPS proxy.
- Keep `MINDI_API_KEY` only in backend env files.
- For 1x MI300X keep tensor parallel at default `1`.
- Lower `VLLM_MAX_MODEL_LEN` if the model fails to allocate memory.
