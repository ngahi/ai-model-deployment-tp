#!/usr/bin/env bash
set -e

mkdir -p artifacts/weights

echo "Downloading SAM vit_b checkpoint..."
wget -nc -O artifacts/weights/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "Done ✅"
