# Data

Each capture lives in `data/<name>/` and is described by its journal entry. Images are **not committed** (see `.gitignore`); keep the original source path in the entry.

| Name | Images | Journal | Source |
|---|---|---|---|
| `toy` | 21 iPhone photos, plush rocket toy | [0002](../journal/0002_toy-dataset.md) | `~/Downloads/toy/jpg` |

To make a dataset visible to ComfyUI, copy it into ComfyUI's input folder. ComfyUI's folder picker ignores symlinks, so it has to be a real copy:

```bash
scripts/sync_to_comfy.sh toy   # → ~/ComfyUI-Shared/input/<name>/
```
