# FM-CAC: Carbon-Aware Control for Battery-Buffered Edge AI

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

## Quick start

```bash
# 1. Environment (Python 3.10, CUDA 12.x GPU recommended)
/usr/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyyaml tqdm gymnasium matplotlib transformers==4.40.2

# 2. Run
bash run.sh                          # default: configs/exp_heavy.yaml
bash run.sh configs/exp_heavy.yaml   # or explicit
```

> **Note on `transformers==4.40.2`:** Sundial's `modeling_sundial.py` uses
> `past_key_values.seen_tokens`, which was removed in transformers 5.x.
> Pin transformers to 4.40.x–4.44.x for Sundial compatibility.

Per-episode CSVs are written to `results/<config_name>/`.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
