#!/bin/bash
# Run this on your AMD GPU droplet (165.245.141.245) to find checkpoint files
# Usage: ssh root@165.245.141.245 'bash -s' < check_droplet_for_ckpts.sh

echo "=== Searching for checkpoint directories ==="
find / -type d -name "*checkpoint*" 2>/dev/null | grep -v "proc\|sys\|snap\|venv\|__pycache__"

echo ""
echo "=== Searching for .safetensors files ==="
find / -name "*.safetensors" 2>/dev/null | head -20

echo ""
echo "=== Searching for .pt / .bin model files ==="
find / -name "*.pt" -o -name "*.bin" 2>/dev/null | grep -v "/usr\|/snap\|/proc" | head -30

echo ""
echo "=== Common training output locations ==="
for d in /mnt/mindi /workspace /root /home/*/workspace /home/*/mindi /opt/mindi; do
    if [ -d "$d" ]; then
        echo "--- $d ---"
        ls -la "$d" 2>/dev/null
        find "$d" -maxdepth 4 -type d -name "*checkpoint*" -o -name "*phase*" -o -name "*lora*" 2>/dev/null
        du -sh "$d" 2>/dev/null
    fi
done

echo ""
echo "=== Docker/Modal volume inspection ==="
df -h 2>/dev/null
ls -la /mnt/ 2>/dev/null
