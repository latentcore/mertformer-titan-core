# Uygulama Planı: High-Performance Edge-Native Intelligence

## Stratejik İlke
Sıkıcı istikrar ve öğrenme hızı. Başarı, yalnızca loss değil, sistem sağlığı üzerinden ölçülür.

## Phase -1: Safety & Failure Budget
- Kill Switch: Non-finite tespiti ve otomatik durdurma.
- Failure Budget: 72 saat anlamlı öğrenme sinyali yoksa pivot/debug tetikleme.
- Checkpoint Drill: Master run öncesi save/restore bütünlüğü.

## Phase 0: Reproducibility & Sanity Gates
- Reproducibility stamp, git hash + config + seed + dataset manifest loglar.
- Overfit gate, 1MB kodu ezberleme kabiliyeti doğrular.
- Observability layer, grad norm + router entropy + VRAM snapshot toplar.
- Golden samples, 50 promptluk stabil mantık kontrol seti.

## Phase 1: Telemetry-Driven Execution
- Expected vs Actual: tokens/sec, loss slope ve GPU kullanımının karşılaştırılması.
- Master training, telemetry threshold’ları ve failure budget ile yönetilir.
- Internal truth benchmarking, HumanEval/MBPP çıktıları ile yapılır.

## Phase 2: Asset Production
- Offline demo video senaryosu.
- One-pager ve technical snapshot.
- Founders Hub başvuru taslağı.

## Phase 3: Future Horizons
- White paper ve defense licensing (validation sonrası).

## Çalıştırma Sırası (Operator Mode)
1. `scripts/operator_mode_gate.py` (safe mod, lokal).
2. `scripts/operator_mode_gate.py --full` (eğitim donanımı).
3. Master run (2.6B) telemetry + failure budget ile.
4. Benchmark ve asset güncellemeleri.

## Kabul Kriterleri
- Kill switch sentetik NaN testi ile doğrulanır.
- Failure budget, no-learning koşulunda pivot tetikler.
- Checkpoint restore drill pass.
- Overfit gate hedef loss veya %80+ iyileşme.
- Golden sample seti 50 prompt içerir.
- Asset stack eksiksizdir.

## Acil Kapanis Protokolu (v27)
- Egitim/teacher akisi zorunlu olmadikca degistirilmez.
- Tum kernel yolları opt-in ve deneysel kalir.
- README/README_TR uyumu (Docs Index + Dizin) saglanir.
- Testler kosulur ve cache temizligi yapilir.

## QAT Plani (Ne Zaman / Nasil)
- Ne zaman: Stabil bir temel checkpoint alindiktan sonra.
- Nasil: Kucuk veri altkumesinde QAT dene, baseline ile karsilastir, sonra genislet.
- Hedef: Low-bit inference kalitesini artirirken stabiliteyi korumak.

## Turkce Tokenizer POC (Risk Kontrollu)
- Varsayilan ogretmen tokenizer kalir.
- Opt-in bayrak ile `tokenizer/tr` kullanilir.
- Kucuk dogrulama setinde token uzunlugu ve kalite karsilastirilir.
- Distillation kalitesi dusersen geri donulur.

## Kernel Deneysel + Tensor Core Opt-in
- Deneysel low-bit kernel opt-in (CUDA + Triton gerekli).
- Tensor-core yolu opt-in (`MERTFORMER_TENSORCORE=1`) ve dogruluk oncelikli.
- Performans iddialari icin gercek profil gerekir.
