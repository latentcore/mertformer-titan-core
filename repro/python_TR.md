# Python Baseline (Review-Ready)

Bu repo, ekosistem sürprizlerini azaltmak için **Python 3.11** baseline’ına pinlenmiştir
(PyTorch/Transformers uyumu, stabil wheel’ler, `snake_demo.py` pygame demosu).

## Lokal Kurulum (macOS/Linux)

```bash
python3.11 -m venv .titan-venv
.titan-venv/bin/python -m pip install -U pip wheel setuptools
.titan-venv/bin/python -m pip install -r requirements.txt
.titan-venv/bin/python -m pip install -e ".[dev]"
```

Opsiyonel demo dependency’leri:

```bash
.titan-venv/bin/python -m pip install -e ".[demo]"
```

Ya da bootstrap helper:

```bash
bash scripts/bootstrap_venv.sh        # dev
bash scripts/bootstrap_venv.sh --demo # dev + demo (pygame)
```

## Notlar
- Offline-first runtime: `TITAN_OFFLINE=1` (default).
- Online mod için `HF_TOKEN` gerekir (ve `TITAN_WANDB=1` ise opsiyonel `WANDB_API_KEY`).
