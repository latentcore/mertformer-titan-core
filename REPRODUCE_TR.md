# REPRODUCE — doğrulama & başlatma

Kanonik komut yüzeyi (`reports/final_commands.md` + `repro/`'dan birleştirildi).
English: [REPRODUCE.md](REPRODUCE.md).

## 0. Bootstrap (offline-first)
```bash
bash scripts/bootstrap_venv.sh        # pinli bağımlılıklarla .titan-venv oluşturur (Python 3.11)
```

## 1. Repo'yu doğrula (ağ yok, GPU yok) — CI `verify` gate'i
```bash
bash scripts/verify_all.sh            # secret-scan + pytest (428 passed, 4 skipped) +
                                      # preflight + operator-mode overfit smoke + gate'ler + manifestler
```
Bu, K4 drill'lerini (checkpoint save→restore→resume) ve CfC/MoE tolerans paritesini toy ölçekte koşar. Hiçbiri kanonik modeli eğitmez.

## 1b. Öğretmensiz smoke (70B teacher indirmesi yok, GPU gerekmez)
```bash
.titan-venv/bin/python scripts/train_smoke.py --steps 50 --cleanup   # kanonik modelde gerçek fwd/bwd/opt; teacher yok
.titan-venv/bin/python scripts/checkpoint_restore_drill.py           # save → reload → allclose
.titan-venv/bin/python scripts/resume_compat_check.py                # kayıtlı adımdan resume
.titan-venv/bin/python scripts/cfc_moe_tolerance_check.py            # MoE seq↔parallel + Liquid parite
```
70B teacher yalnız `distill_alpha > 0` iken yüklenir (`train.py`). `train_smoke.py` teacher'ı hiç kurmaz; tam eğitim girişinden öğretmensiz geçmek için `TITAN_DISTILL_ALPHA=0` (KD kapalı, saf cross-entropy) — bu yol teacher'ı asla yüklemez/indirmez. `HF_TOKEN`'i unset bırak (+ `TITAN_SKIP_PHASE0=1`) ki opsiyonel phase-0 precompute da indiremesin.

## 2. Eğitim hazırlığını kontrol et (eğitim yok)
```bash
bash zero_touch_start.sh --check-only   # train_allowed + reason code + engelleri basar
```

## 3. Gerçek koşuyu başlat (hedef donanım: H100/H200)
```bash
bash zero_touch_start.sh                # kanonik 45K sahipli lane: readiness verdict,
                                        # run lock, resume policy, post-train autorun
```
- Adım sınırı `cfg.max_steps`'tir, env `TITAN_MAX_STEPS` ile kontrol edilir (varsayılan 45000). Kanonik mimarinin birkaç-adımlık **öğretmensiz** smoke'u: `TITAN_DISTILL_ALPHA=0 TITAN_SKIP_PHASE0=1 TITAN_MAX_STEPS=2 bash zero_touch_start.sh`. `TITAN_DISTILL_ALPHA`'yı yalnız gerçekten 70B teacher / KD istediğinde kaldır (o zaman yükler/indirir).
- Online-teacher distilasyonu `HF_TOKEN` ister; sıkı offline lane önceden hesaplanmış top-k logit ister (engeller için bkz. [STATUS_TR.md](STATUS_TR.md)).

## 4. Tam closure ladder (artefaktları yeniden üretir; opsiyonel, yavaş)
```bash
bash scripts/final_one_shot.sh          # zip + SBOM + manifest; free private repo'da GitHub-policy
                                        # adımı 403 verir, beklenir ve atlanır (visibility değişmez)
```
