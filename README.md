# TRACE
Transition-grounded Revision Agent for Closed-loop Electrolyte Optimization 

## How it works

1. **Rank directions.** The MLP prior scores eight actions for the current formulation and objective, and the top three become candidate directions.
2. **Retrieve evidence.** Relevant before-and-after precedents are retrieved from the literature database.
3. **Generate a proposal.** An OpenAI model combines this evidence with the component family tables to produce one concrete formulation change.
4. **Collect feedback.** Enter feedback in the terminal.
5. **Update and continue.**
   - A positive result applies the proposal to the current formulation.
   - Other results retry within the same direction, avoiding failed families and duplicates. After repeated non-positive results, the agent moves to the next direction.


## Repository structure

```
.
├── interactive_feedback_agent_terminal_top3.py   # Main program
├── prior_suggestions.py                          # MLP inference
├── literature_rag.py                             # Precedent retrieval and family tables
├── action_family_rag.py                          # Action-specific evidence and proposal generation
├── json2priorreward.py                           # Build MLP training labels
├── json_literature.json                          # Anonymized literature data
├── MLP_new/
│   ├── MLP3_smiles.py                            # Model definition and SMILES feature encoding
│   └── mlp_prior_state_new_v5.pt                 # Trained MLP checkpoint
├── supplementary/
│   ├── component_smiles_lookup.json              # Component name to SMILES
│   ├── component_aliases.json                    # Component name aliases
│   ├── salt_families.json                        # Salt family table
│   ├── solvent_families.json                     # Solvent family table
│   ├── additive_families.json                    # Additive family table
│   ├── learned_family_tables.json                # Families assigned by the LLM (updated at runtime)
│   ├── excel_solvent_vocabulary.json             # Solvent vocabulary from experts
│   ├── excel_codebook.json                       # Anonymized additive names per article from experts
│   └── unmapped_solvent_keys.json                # Additional solvent names compared with the one from experts
├── examples/
│   └── initial_state.json                        # Example input
└── software/                                     # Future work (see below)
```

## Requirements

- An OpenAI API key and network access. Using the API may incur charges.
- Python 3.9. The code was run on macOS with Python 3.9.6 and the package versions below.

## Installation

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 numpy==2.0.2 scikit-learn==1.6.1 tqdm==4.67.3 \
    rdkit==2025.9.2 openai==2.21.0 sentence-transformers==5.1.2
```

On first use, the retrieval step downloads the `sentence-transformers/all-MiniLM-L6-v2` text embedding model from Hugging Face. If the model is unavailable, retrieval falls back to TF-IDF only, which may change the ranking of results.

## Quick start

Run from the repository root, because the default model and database paths are relative:

```bash
export OPENAI_API_KEY="your-api-key"
python interactive_feedback_agent_terminal_top3.py --folder examples
```

After each proposal, the program prompts for feedback:

| Input | Meaning |
|---|---|
| `positive` (`pos`, `p`, `+`) | Apply the proposal to the current formulation |
| `negative` (`neg`, `n`, `-`) | Reject the proposal |
| `unchange` (`unchanged`, `same`, `u`) | The proposal made no meaningful difference |
| `quit` | Stop the session |

### Input

`--folder` must contain a file named `initial_state.json`:

```json
{
  "state0": {
    "composition": {
      "salt": "LiPF6",
      "salt_concentration": "1.0 M",
      "solvents": { "EMC": "0.5", "PC": "0.5" },
      "additives": {}
    }
  },
  "objective": "high_voltage"
}
```

### Output

These files are written to the same folder:

| File | Contents |
|---|---|
| `interactive_feedback_session.json` | Full session record |
| `interactive_feedback_attempts.json` | Every attempted proposal |
| `interactive_feedback_final_state.json` | Final formulation |
| `.prior_suggestions_cache.json` | Cached LLM responses |

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--folder` | Required | Folder containing `initial_state.json`; outputs are written here |
| `--prior` | `MLP_new/mlp_prior_state_new_v5.pt` | MLP checkpoint |
| `--data` | `json_literature.json` | Literature data |
| `--llm-mode` | `gpt-5` | OpenAI model for proposals |
| `--llm-temperature` | `0.0` | LLM temperature |
| `--top-k` | `5` | Top-k for literature retrieval |
| `--max-steps` | `20` | Maximum interaction steps |
| `--top-action-candidates` | `3` | Number of top MLP directions to consider |
| `--nonpositive-threshold` | `3` | Non-positive results tolerated for an action before it is blocked |
| `--duplicate-regeneration-limit` | `3` | Regeneration attempts after a duplicate or invalid proposal before the action is blocked |
| `--block-actions` | Empty | Comma-separated actions to skip, such as `additive_change` |
| `--known-molecules-only` | Off | Reject molecules that are not already in the family tables |
| `--verbose-terminal` | Off | Print the full ranking and proposal JSON |

## Data

`json_literature.json` contains 903 unique articles. 

## Rebuilding the MLP prior

The repository can run the provided checkpoint directly. To rebuild the MLP prior from the literature data, run these commands from the repository root:

```bash
# 1. Build the training labels
export OPENAI_API_KEY="your-api-key"
python json2priorreward.py --input json_literature.json --output prior_reward_dataset.json

# 2. Train the MLP prior
python MLP_new/MLP3_smiles.py --data prior_reward_dataset.json --out MLP_new/my_prior.pt --seed 223 --allow_missing_structures
```

Notes:

- **Differences from the provided checkpoint.** Rebuilt labels can differ from the data used to train the checkpoint, and so can a retrained model. SMILES come first from `supplementary/component_smiles_lookup.json`, and transitions the rules cannot classify are decided by a new LLM call.

## Future work: `software/`

`software/` is an experimental, self-contained version under continued development. All LLM calls default to `gpt-5`. It adds:

- A guided terminal program (`run.py`)
- Feedback memory that stores each result and makes it retrievable as new evidence
- Continual learning that applies a bounded adjustment to the frozen MLP prior
- Optional generation of GROMACS job folders when local MD tools are detected

It uses the same Python environment:

```bash
cd software
python run.py
```

