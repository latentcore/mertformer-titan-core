# REPRODUCE — doğrulama & başlatma

Kanonik komut yüzeyi (`reports/final_commands.md` + `repro/`'dan birleştirildi).
English: [REPRODUCE.md](REPRODUCE.md).

## 0. Bootstrap (offline-first)
```bash
bash scripts/bootstrap_venv.sh        # pinli bağımlılıklarla .titan-venv oluşturur (Python 3.11)
```

## 1. Repo'yu doğrula (ağ yok, GPU yok) — CI `verify` gate'i
```bash
bash scripts/verify_all.sh            # secret-scan + pytest (370 passed, 4 skipped) +
                                      # preflight + operator-mode overfit smoke + gate'ler + manifestler
```
Bu, K4 drill'lerini (checkpoint save→restore→resume) ve CfC/MoE tolerans paritesini toy ölçekte koşar. Hiçbiri kanonik modeli eğitmez.

## 2. Eğitim hazırlığını kontrol et (eğitim yok)
```bash
bash zero_touch_start.sh --check-only   # train_allowed + reason code + engelleri basar
```

## 3. Gerçek koşuyu başlat (hedef donanım: H100/H200)
```bash
bash zero_touch_start.sh                # kanonik 45K sahipli lane: readiness verdict,
                                        # run lock, resume policy, post-train autorun
```
- Adım sınırı `cfg.max_steps`'tir, env `TITAN_MAX_STEPS` ile kontrol edilir (varsayılan 45000). Kanonik mimarinin birkaç-adımlık smoke'u (davranış-param değişikliği yok): `TITAN_MAX_STEPS=2 bash zero_touch_start.sh`.
- Online-teacher distilasyonu `HF_TOKEN` ister; sıkı offline lane önceden hesaplanmış top-k logit ister (engeller için bkz. [STATUS_TR.md](STATUS_TR.md)).

## 4. Tam closure ladder (artefaktları yeniden üretir; opsiyonel, yavaş)
```bash
bash scripts/final_one_shot.sh          # zip + SBOM + manifest; free private repo'da GitHub-policy
                                        # adımı 403 verir, beklenir ve atlanır (visibility değişmez)
```
