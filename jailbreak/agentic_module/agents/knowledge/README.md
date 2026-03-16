# Knowledge Folder

This folder contains tactic knowledge files for the jailbreaking pipeline.

## Structure

```
knowledge/
├── default_tactics.json                          # Read-only baseline tactics (do not modify)
├── exp_qwen8b_llama1b_curated_tactics.json      # Per-experiment curated tactics
├── exp_qwen06b_phi4_curated_tactics.json        # Per-experiment curated tactics
└── ...
```

## Files

### default_tactics.json
The baseline set of tactics that all experiments start with. This file should not be modified directly. It serves as the fallback when no experiment-specific curated tactics are available.

### {experiment_name}_curated_tactics.json
Per-experiment curated tactics files created during tactic curation (Stage 2 of the pipeline). These contain:
- Original default tactics
- Newly discovered tactics from successful jailbreaks

## Loading Logic

1. Check if experiment-specific curated tactics exist in this folder
2. If found, load curated tactics (includes defaults + newly discovered)
3. If not found, load default_tactics.json only

## Saving Logic

After tactic curation, new tactics are saved to this folder using the experiment name as prefix:
- `{experiment_name}_curated_tactics.json`
