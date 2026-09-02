# Ablation: BitLinear Kapalı

**Amaç**: BitNet ve standart dense ağırlıkları karşılaştırmak.

**Config değişikliği**:
- `use_bitnet: false`

**[2026-09-02] Ön koşul bug'ı düzeltildi**: `use_bitnet: false` daha önce **modele hiç etki
etmiyordu** — `layers/ffn.py`, `layers/mla.py`, `layers/moe.py` ve `layers/liquid.py`
`BitLinear(...)`'ı koşulsuz çağırıyordu, yani bu ablasyon olduğu gibi koşulsaydı sessizce
anlamsız/sıfır-fark bir sonuç üretecekti. `layers/bitlinear.py::make_linear` ile düzeltildi,
dört dosya da artık `cfg.use_bitnet`'i gerçekten okuyor. Tam kayıt için `DECISIONS.md` ve
`BACKLOG.md`'ye bakın.

**Koşturucu**: `scripts/run_bitlinear_ablation.py` (Liquid ablasyonuyla aynı $0-pilot
metodolojisi — bkz. `no_liquid/README_TR.md`). Fix sonrası CPU-smoke ile doğrulandı (iki kol
artık gerçekten ayrışıyor; fix öncesi birebir aynıydı).

**Durum**: Gerçek bir sinyal için henüz koşulmadı — CPU smoke sadece anahtarın çalıştığını
kanıtlıyor, ölçülmüş bir sonuç değil. Eğitim donanımında çalıştırın (`python
scripts/run_bitlinear_ablation.py --steps 500 --device cuda` ya da daha fazla) ve sonuçları
`ablations/results.md` dosyasına kaydedin.
