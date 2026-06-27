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
- Expected vs Actual: tokens/sn, loss slope ve GPU kullanımının karşılaştırılması.
- Master training, telemetry threshold’ları ve failure budget ile yönetilir.
- Internal truth benchmarking, HumanEval/MBPP çıktıları ile yapılır.

## Phase 2: Asset Production
- One-pager ve technical snapshot.
- Founders Hub başvurusu: 2026-05-31'de yapıldı (taslak `private/commercial/` altında arşivlendi).

## Phase 3: Future Horizons
- White paper ve defense licensing (validation sonrası).

## Çalıştırma Sırası (Operator Mode)
1. `scripts/operator_mode_gate.py` (safe mod, yerel).
2. `scripts/operator_mode_gate.py --full` (eğitim donanımı).
3. Master run (2.64B tasarım hedefi) telemetry + failure budget ile.
4. Benchmark ve asset güncellemeleri.

## Kabul Kriterleri
- Kill switch sentetik NaN testi ile doğrulanır.
- Failure budget, no-learning koşulunda pivot tetikler.
- Checkpoint restore drill pass.
- Overfit gate hedef loss veya %80+ iyileşme.
- Golden sample seti 50 prompt içerir.
- Asset stack eksiksizdir.

## Acil Kapanış Protokolü (v1.0 (Build 30))
- Eğitim/teacher akışı zorunlu olmadıkça değiştirilmez.
- Tüm kernel yolları opt-in ve deneysel kalır.
- README/README_TR uyumu (Docs Index + Dizin) sağlanır.
- Testler koşulur ve cache temizliği yapılır.

## QAT Planı (Ne Zaman / Nasıl)
- Ne zaman: Stabil bir temel checkpoint alındıktan sonra.
- Nasıl:
  - Faz A (Pilot, 1-2 gün): 1-5% alt küme ile QAT aç, sabit seed kullan.
  - Faz B (Karşılaştırma, 1 gün): loss/throughput değerlerini baseline ile karşılaştır.
  - Faz C (Yaygınlaştırma, 2-3 gün): Faz B olumluysa tam müfredat için aç.
- Hedef: Low-bit inference kalitesini artırırken stabiliteyi korumak.
- Çıkış kriteri: Val loss gerilemesin, stabilite spike olmasın.

## Türkçe Tokenizer POC (Risk Kontrollü)
- Varsayılan öğretmen tokenizer kalır.
- Opt-in bayrak ile `tokenizer/tr` kullanılır.
- Pilot adımları:
  - Faz A (30-60 dk): 500-1,000 sample tokenize et, ortalama token uzunluğunu karşılaştır.
  - Faz B (1-2 saat): 200-step mini-train (CPU/MPS), loss trendini karşılaştır.
  - Faz C (aynı gün): loss bozulursa otomatik geri dönüş.
- Çıkış kriteri: %5'ten fazla token uzunluğu şişmesi yok ve loss stabil.

## Kernel Deneysel + Tensor Core Opt-in
- Deneysel low-bit kernel opt-in (CUDA + Triton gerekli).
- Tensor-core yolu opt-in (`MERTFORMER_TENSORCORE=1`) ve doğruluk öncelikli.
- Performans iddiaları için gerçek profil gerekir.
