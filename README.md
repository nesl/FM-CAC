# FM-CAC: Carbon-Aware Control for Battery-Buffered Edge AI

[[arXiv]](https://arxiv.org/pdf/2604.16448)

Reference implementation of **FM-CAC** (Foundation-Model-driven Carbon-Aware
Control): a Time-Series Foundation Model (TSFM) forecast fed into a
receding-horizon Dynamic Programming (DP) solver that jointly picks the
inference pipeline, hardware state, and battery charge/discharge action to
cut carbon while meeting Quality-of-Service (QoS).

## Repository layout

```
.
├── main_sundial_mpc.py             # FM-CAC: TSFM + receding-horizon DP controller
├── env_simulator.py                # Battery + grid + inference simulator
├── env_config.py                   # Hardware profile registry
├── config_loader.py                # YAML scenario loader
├── evaluator.py                    # Episode runner and metric aggregator
├── run.sh                          # One-shot runner
├── configs/
│   ├── exp_heavy.yaml              # Scenario: CAISO, heavy tier, default QoS
│   └── heavy_hardware_config.yaml  # Hardware profile (YOLO on Jetson Orin Nano)
└── data/
    └── CAISO_3year_15min.csv       # 3-year CAISO carbon-intensity + price trace
```

## Environment setup

```bash
# 1. Create and activate a virtual environment
/usr/bin/python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (CUDA 12.1 build)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install numpy pandas pyyaml tqdm gymnasium matplotlib transformers==4.40.2
```

## Run

```bash
bash run.sh                          # default: configs/exp_heavy.yaml
bash run.sh configs/exp_heavy.yaml   # or explicit
```

Per-episode CSVs and per-seed / cross-seed JSON summaries are written to
`results/<config_name>/`.

## Forecaster

The default carbon-intensity forecaster is
[Sundial](https://huggingface.co/thuml/sundial-base-128m). The controller is
agnostic to the choice of Time-Series Foundation Model: any probabilistic
forecaster that exposes a `forecast(context, horizon)` interface returning
the predictive mean and standard deviation can be substituted. To replace
the default, provide a subclass of `SundialForecaster` in
`main_sundial_mpc.py` and override `forecast` accordingly. No modifications
to the remainder of the codebase are required.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
