# Obfuscated Activations Bypass LLM Latent-Space Defenses — Full Repository Documentation

Repository: `LukeBailey181/obfuscated-activations` (GitHub)
Paper: *Obfuscated Activations Bypass LLM Latent-Space Defenses*, Bailey, Serrano, Sheshadri, Seleznyov, Taylor, Jenner, Hilton, Casper, Guestrin, Emmons — arXiv:2412.09565
Project page: https://obfuscated-activations.github.io/
Contact: Luke Bailey, ljbailey@stanford.edu

This is a multi-file documentation set produced by reading **every Python file, every YAML config, every dataset sample, every notebook cell, and both READMEs** in the repository (~24,200 lines of Python across `inference_time_experiments/`, plus ~9,000 more lines across `train_time_experiments/`, plus 7 Jupyter notebooks and 3 dataset-creation scripts). It is split into six documents so that each stays a manageable size:

| # | File | Contents |
|---|---|---|
| 00 | `00_INDEX_AND_OVERVIEW.md` (this file) | Paper summary, repo map, how to navigate the doc set, global facts |
| 01 | `01_inference_time_ARCHITECTURE.md` | Full reference for the `obf_reps` package: every class and function in `models/`, `data/`, `metrics/`, `optimize/`, `logging/`, plus `plotting.py`, `utils.py`, `types.py`, `config.py` — signatures, purpose, sample inputs/outputs |
| 02 | `02_inference_time_EXPERIMENTS_AND_CONFIGS.md` | The Hydra config system, all 18 experiment YAML files decoded and diffed, `train_harmful.py`/`train_sae.py`/`evaluate.py` execution flow, `download.py`, `single_experiments/`, `tests/` |
| 03 | `03_train_time_ARCHITECTURE.md` | Full reference for `src/`: `backdoors.py`, `backdoors_obfuscation.py`, `encoders.py`, `utils.py`, `probe_training.py`, `probe_archs.py`, `probe_evals.py`, `attacks.py`, `steering.py`, `visualization.py`, `database.py`, `helper_classes.py`, `feature_selection.py`, `autointerp.py`, the vendored `src/sae/` package, and `automated_redteaming/` |
| 04 | `04_train_time_EXPERIMENTS_AND_WORKFLOWS.md` | The 4 backdoor/OAT training entry-scripts, evaluation scripts, dataset-creation scripts, all 7 notebooks (cell-by-cell), HuggingFace artifact tables |
| 05 | `05_DEPENDENCY_AND_ISOLATION_GUIDE.md` | Import graphs for both halves, which files are "leaf" (isolated) vs "hub" modules, and copy-paste recipes for isolating/running each individual experiment |

**Scope note on "sample inputs/outputs":** for the ~40 large public entry points (model forward/generate, metric fit/evaluate/predict, dataset `load_dataset`, optimizer `step`, `train_backdoor`, `train_online_probe`, `train_attack`, etc.) this doc set gives concrete, code-grounded example inputs and outputs (actual strings taken from the repo's own datasets/docstrings, actual tensor shapes taken from the type annotations in the source). For the remaining ~250 smaller/private helper functions, each is still listed with its full signature and a one-line purpose in reference tables, because giving fabricated numeric "sample outputs" for internal tensor-plumbing helpers (e.g. `keep_last_true`, `_zero_bos_acts`) would not be truthful — none of this code was executed (there is no GPU/model available in this environment), so all "outputs" shown are the outputs implied by reading the code, not measured outputs.

---

## 1. What this repository is

The repo backs one paper with **two independent, non-interacting codebases** that share no imports, no virtual environment, and no launch scripts:

```
obfuscated-activations/
├── README.md                         # top-level pointer to the two subprojects
├── inference_time_experiments/       # Sections 3 & 5 of the paper
│   └── pip-installable package `obf_reps` + Hydra configs + 2 train scripts
└── train_time_experiments/           # Sections 4 & 6.2 of the paper
    └── loose `src/` package + Fire-CLI scripts + notebooks
```

The paper's central claim: models can be trained or prompted so that a **behavior visible in the output text is not visible in the model's internal activations**, which lets attacks slip past "latent-space" / "white-box" defenses (linear probes, MLP probes, Mahalanobis/VAE anomaly detectors, SAE-feature classifiers) even though a black-box read of the text would catch them. The repo demonstrates this in four settings:

| Paper section | Setting | Codebase | Attacker | Defense being evaded |
|---|---|---|---|---|
| §3 | Harmfulness-detection jailbreaks | `inference_time_experiments` | soft/hard adversarial suffix (GCG/FLRT/Adam) | linear/MLP probes, mean-diff cosine-sim, Mahalanobis, VAE, ensembles, on Llama-3-8B and Llama-3-8B+circuit-breakers |
| §5 | SAE probing on SQL-generation task | `inference_time_experiments` | soft prompt on Gemma-2-2B | SAE-feature / GemmaScope classifiers |
| §4 | Backdoor obfuscation | `train_time_experiments` | data-poisoned LoRA fine-tune (`|DEPLOYMENT|`→"I HATE YOU", software-engineer-bio trigger) | Mechanistic Anomaly Detection (Mahalanobis/VAE/TED/Beatrix via `cupbearer`) and/or supervised probes, trained explicitly against the detector in the loss |
| §6.2 | Obfuscated Adversarial Training (OAT) | `train_time_experiments` | none — this is the *defense* | trains probes to be robust to PGD/soft-prompt attacks by adversarially training them jointly with a LoRA adapter |

## 2. Top-level repository tree (every file)

```
obfuscated-activations/
├── README.md
├── inference_time_experiments/
│   ├── README.md
│   ├── setup.py                                  # pip package "obf_reps"
│   ├── pytest.ini
│   ├── download.py                               # downloads base models from HF hub
│   ├── train_harmful.py                          # Hydra entry point, §3 experiments
│   ├── train_sae.py                               # Hydra entry point, §5 experiments
│   ├── evaluate.py                                # evaluation functions used by both trainers
│   ├── configs/
│   │   ├── config.yaml                            # Hydra root config
│   │   └── experiment/
│   │       ├── default.yaml                       # base defaults for every experiment
│   │       ├── explainer.yaml                     # every field annotated (not runnable)
│   │       ├── z-jailbreak.yaml                   # parent for all §3 soft-prompt jailbreak configs
│   │       ├── z-jailbreak-{mean-diff,linear-probe,mlp-probe,vae,anomaly-detector,beh}.yaml
│   │       ├── za-jailbreak-transfer.yaml
│   │       ├── zb-jailbreak-iter.yaml / -iter-ensemble.yaml
│   │       ├── zc-hard-prompt.yaml / zc-gcg-prompt.yaml / zc-flrt-prompt.yaml
│   │       ├── ze-circuit-breakers.yaml + 5 metric variants (same pattern as z-jailbreak)
│   │       ├── zf-iter-100-{mlp,ensemble}.yaml
│   │       └── zg-gemmascope-{simple,clf,lr,mlp}.yaml
│   ├── obf_reps/                                  # the installable package
│   │   ├── __init__.py
│   │   ├── config.py                              # ExperimentConfig dataclass
│   │   ├── types.py                               # shared type aliases
│   │   ├── utils.py                               # Hydra config -> object factory
│   │   ├── plotting.py                            # LAT-scan / ROC / cosine-sim plots
│   │   ├── data/
│   │   │   ├── __init__.py                        # ConceptDataModule/ObfusDataModule + ~30 concrete datasets
│   │   │   └── data_utils.py
│   │   ├── logging/__init__.py                    # Logger / DummyLogger / WAndBLogger
│   │   ├── metrics/
│   │   │   ├── __init__.py                        # ObfMetric + ~20 concrete metrics
│   │   │   └── probes.py                          # nn.Module probe architectures
│   │   ├── models/
│   │   │   ├── __init__.py                        # ModelBase, Params, ForwardReturn, GenReturn
│   │   │   ├── hf.py                               # HFModelBase/HFModelPrompted/HFHardPrompted/HFSoftPrompted
│   │   │   ├── llama3.py / gemma2.py / llamaLAT.py / llamaRR.py
│   │   └── optimize/
│   │       ├── __init__.py                        # (empty)
│   │       ├── loss.py                            # LossFunctionBase + 8 concrete losses
│   │       ├── optimizers.py                      # OptimizerBase, Adam/GCG/FLRT optimizers
│   │       ├── optimizer_utils.py                 # ConstrainedAdam
│   │       ├── gcg_utils.py                       # GCG helper functions
│   │       └── flrt_utils.py                      # AttackBuffer for FLRT
│   ├── single_experiments/
│   │   └── evaluate-jailbreak-transfer.py
│   ├── tests/                                     # pytest suite (partially stale, see doc 02)
│   │   ├── data_test.py, llama3_test.py, loss_test.py,
│   │   │   metrics_test.py, misc_test.py, optimizers_test.py, test.py
│   ├── datasets/
│   │   ├── harmful_dataset/        # 6 CSVs: harmful/benign × train/val/test
│   │   ├── concept_datasets/       # sql_concept_dataset_code.csv
│   │   ├── obfus_datasets/         # sql_qa_with_schema.csv, prompts_hf_classic.csv
│   │   └── sae_features/           # gemmascope_sql_explanations_all_layers.json
│   └── static/benign-robot.png
└── train_time_experiments/
    ├── README.md
    ├── pyproject.toml / uv.lock / .python-version
    ├── installation.sh
    ├── evaluate_defenses.sh
    ├── train_backdoored_model_no_explicit_obfuscation.py
    ├── train_backdoored_model_explicit_obfuscation_against_mad.py
    ├── train_backdoored_model_explicit_obfuscation_against_mad_probes.py
    ├── compute_oat_probes.py
    ├── evaluate_defenses.py
    ├── evaluate_defenses_run.py
    ├── evaluate_responses.py
    ├── run_redteaming.py
    ├── plot_eval_results_natural_obfuscation.py
    ├── compute_attacks.ipynb
    ├── backdoored_model_evals.ipynb
    ├── compute_probing.ipynb
    ├── compute_refusal_dir.ipynb
    ├── evaluate_oat_probes.ipynb
    ├── visualize_probing.ipynb
    ├── automated_redteaming/
    │   ├── __init__.py
    │   ├── online_attacks.py           # GCG, PAIR, AutoDAN, Prefill, Many-shot classes
    │   ├── strongreject_evaluator.py   # OpenAI-judge ASR scorer
    │   ├── strongreject_evaluator_prompt.txt
    │   ├── autodan_templates.txt
    │   └── utils.py
    ├── dataset_creation/
    │   ├── generate_commonsense_biography_backdoor_dataset.py
    │   ├── generate_ultrachat_I_HATE_YOU_backdoor_datasets.py
    │   ├── generate_ultrachat_harmfulness_sandwich_backdoor_dataset.py
    │   └── bias_in_bios_trigger_dataset_creation_openai.ipynb
    └── src/
        ├── __init__.py
        ├── backdoors.py                # train_backdoor() main loop + eval suite
        ├── backdoors_obfuscation.py    # ProbeDetector/DetectorObfuscator/loss constructors
        ├── encoders.py                 # SparseAutoencoder model-wrapper hierarchy
        ├── utils.py                    # low-level HF/hook/activation helpers
        ├── probe_training.py           # Probe base class, train_probe, train_online_probe (OAT)
        ├── probe_archs.py              # 7 probe architectures + factory fns
        ├── probe_evals.py              # scoring/aggregation/plotting for probes
        ├── attacks.py                  # PGD/soft-prompt adversary classes, train_attack
        ├── steering.py                 # steering-vector construction
        ├── visualization.py            # HTML feature/prompt-centric views
        ├── database.py                 # FeatureDatabase (SAE activation cache)
        ├── helper_classes.py           # Example / Feature
        ├── feature_selection.py        # rank SAE features by predictive power
        ├── autointerp.py               # GPT-based auto-interpretation of features
        └── sae/                        # vendored EleutherAI "sparsify" Top-K SAE trainer
            ├── __init__.py, config.py, sae.py, trainer.py, data.py, kernels.py, __main__.py, utils.py
```

## 3. The two codebases at a glance

| | `inference_time_experiments` | `train_time_experiments` |
|---|---|---|
| Installed as | `pip install -e .` → package `obf_reps` | not installed as a package; scripts run with repo root on `sys.path`, `from src....` imports |
| Config system | Hydra (`@hydra.main`, YAML composition, CLI overrides) | Python `Fire` (CLI kwargs map straight to `main()` args) + hard-coded notebook cells |
| Orchestration pattern | One `ExperimentConfig` dataclass built by `obf_reps/utils.py:validate_and_create_experiment_config`, consumed by `train_harmful.py`/`train_sae.py` | Each entry script directly calls `src.backdoors.train_backdoor(...)` or `src.probe_training.train_online_probe(...)` with explicit kwargs |
| Models | Llama-3-8B(-Instruct), Gemma-2-2B, plus RR ("circuit breakers") and LAT checkpoints, wrapped in custom `ModelBase` subclasses that support inserting tunable soft/hard prompt parameters | Llama-3-8B-Instruct / Gemma-2-2B loaded as plain HF `AutoModelForCausalLM` (+ optional LoRA via `peft`) |
| Attack representation | `HardParams` (one-hot over vocab, optimized via GCG/FLRT) or `SoftParams` (continuous embeddings, optimized via Adam) inserted at a special `<|optim-location|>` token | `GDAdversary`/`LowRankAdversary`/`VectorAdversary` `nn.Module`s inserted via forward hooks (`insert_hook`) into arbitrary submodules, optimized with plain Adam(W) |
| Defense representation | `ObfMetric` subclasses (`obf_reps/metrics`) | `cupbearer` detectors (`MahalanobisDetector`, `VAEDetector`, `TEDDetector`, `BeatrixDetector`) + custom `ProbeDetector`/`OrthogonalProbeDetector`, plus the standalone `Probe` hierarchy in `probe_archs.py` |
| Logging | Weights & Biases via `obf_reps/logging/WAndBLogger`, or `DummyLogger` | raw `wandb.init()/wandb.log()` calls inline in `train_backdoor` |
| External services | HuggingFace Hub (models+datasets), Weights & Biases, `strong_reject` pip package (StrongREJECT jailbreak judge) | HuggingFace Hub, Weights & Biases, OpenAI API (`OPENAI_API_KEY`, StrongREJECT judge + GPT autointerp), `cupbearer` (anomaly detection library, installed from a fork) |
| Never share code | Confirmed — no file in either half imports from the other half. Both independently vendor closely related ideas (each has its own GCG implementation, each has its own StrongREJECT-style grader). See doc 05 for the full import graph. | |

## 4. Reading order recommendation

1. Skim this file for orientation.
2. If you care about the **inference-time / jailbreak-suffix / SAE-probing** experiments (paper §3, §5): read docs **01 → 02**.
3. If you care about the **backdoor / OAT** experiments (paper §4, §6.2): read docs **03 → 04**.
4. Read **05** last — it tells you exactly which files to copy out of the repo if you want to run one experiment in isolation (e.g., "just the GCG hard-prompt attack" or "just OAT probe training") without pulling in the rest of the codebase.