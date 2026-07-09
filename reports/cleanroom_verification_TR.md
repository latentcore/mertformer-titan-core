# Clean-Room Doğrulaması (Build 30)

## Amaç
Yeni bir clone üzerinde, temel kalite kapılarının izole bir path'te tekrar üretilebilir olduğunu doğrulamak.

## Ortam
- Kaynak repo: `.`
- Temiz clone yolu: `/tmp/nihai_cleanroom_b27`
- Python: `3.11.14`
- Venv: `.cleanroom-venv`
- Test edilen commit: `a07a8c1`

## Çalıştırılan Komutlar
```bash
SOURCE_REPO="<repo-url-or-local-path>"
git clone "$SOURCE_REPO" /tmp/nihai_cleanroom_b27
cd /tmp/nihai_cleanroom_b27
python3.11 -m venv .cleanroom-venv
.cleanroom-venv/bin/python -m pip install -U pip
.cleanroom-venv/bin/python -m pip install -r requirements.txt
.cleanroom-venv/bin/python -m pip install -e .
.cleanroom-venv/bin/python -m pytest -q
.cleanroom-venv/bin/python -m pip install ruff
.cleanroom-venv/bin/python -m ruff check .
TITAN_PYTHON="$PWD/.cleanroom-venv/bin/python" TITAN_OFFLINE=1 bash scripts/verify_all.sh
```

## Sonuçlar
- `pytest`: `412 passed, 4 skipped`
- `ruff`: `All checks passed!`
- `verify_all`: `[verify] OK`

## Notlar
- Clean-room koşusu, yeni bir lokal clone ve ayrı sanal ortam üzerinde yapılmıştır.
- Bu rapor A17 (clean-room doğrulama) kapısı için build-time kanıtıdır.
