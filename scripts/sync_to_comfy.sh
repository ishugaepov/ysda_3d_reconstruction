#!/bin/sh
# Copy data/<name>/images into ComfyUI's input folder as input/<name>/.
# ComfyUI ignores symlinked folders in its pickers, so this is a real copy.
set -eu
name="$1"
src="$(cd "$(dirname "$0")/.." && pwd)/data/$name/images"
dst="$HOME/ComfyUI-Shared/input/$name"
[ -d "$src" ] || { echo "no such dataset: $src" >&2; exit 1; }
mkdir -p "$dst"
rsync -a --delete "$src/" "$dst/"
echo "synced $(ls "$dst" | wc -l | tr -d ' ') files -> $dst"
