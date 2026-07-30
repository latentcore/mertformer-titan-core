# MertFormer Titan (Build 30 V2)

Cihaz-içi (on-device) odaklı bir LLM araştırma yığını: **BitNet b1.58 ternary ağırlıklar**,
seyrek **Mixture-of-Experts (MoE)**, **Liquid/CfC mixer** ve **GQA attention** (grouped-query). Kanıt-önce: bu repo **pre-training,
closure-complete bir mühendislik PoC**'sidir — kanonik model **henüz eğitilmedi** ve gerçek
bir koşu checkpoint üretene kadar hiçbir yetenek/benchmark iddiası yapılmaz.

English: [README.md](README.md).

## Durum (kanonik: [STATUS_TR.md](STATUS_TR.md))
- **Build:** `726 passed, 5 skipped` (offline-first `pytest`).
- **Readiness:** `decision_reason_code = READY_REMOTE_BOOTSTRAP` · `recommended_path = remote_bootstrap` · `train_allowed = true`.
- **Koşu engelleri:** `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`.
- **Tek gerçek boşluk:** gerçek bir 45K GPU koşusu (H100/H200). Crash-sınıfı bug: yok.

## Mimari (ölçülen, benchmark değil)
- 18 katman · hidden 2048 · 16 head / 8 KV (**GQA attention (grouped-query, current implementation)**) · 8 expert top-2, her 3. katmanda MoE · [4, 10, 16] katmanlarında Liquid/CfC mixer · BitNet b1.58 ternary.
- **Ölçülen runtime param:** `3,672,982,022` (~3.67B). **Tasarım hedefi:** 2.64B. İki etiket de bilinçli — bkz. [reports/param_accounting_report.md](reports/param_accounting_report.md) ve [DECISIONS_TR.md](DECISIONS_TR.md).
- Tam mimari: [ARCHITECTURE.md](ARCHITECTURE.md).

## Ölçülen vs ölçülmeyen (dürüst sınır)
- **Ölçülen:** repo self-test'leri, offline smoke harness, 12-seed Liquid ablation.
- **Liquid ablation hükmü ([ABLATION_TR.md](ABLATION_TR.md)):** OFF %96.32 / ON %94.69 ID exact-accuracy (Δ−1.63 pp, p=0.305, Cohen's d=−0.43). **Ölçülen doğruluk faydası yok; ~%30 daha yavaş; toy ölçekte sonuçsuz.** Doğrulanmış 45K koşusuna kadar **hiçbir Liquid hız/gecikme iddiası** yapılmaz.
- **Ölçülmeyen (boşluk):** kanonik 3.67B model hiç eğitilmedi, yani yakınsama/genelleme **doğrulanmadı**. Bu donanım bağımlılığıdır, kod düzenlemesi değil.

## Claim boundary (iddia sınırı)
Bu repo bir **pre-training**, **proof-of-system** PoC'sidir; production-ready/sertifikalı bir platform iddiası taşımaz ve gerçek bir koşu checkpoint üretene kadar eğitilmiş yetenek için **NOT ELIGIBLE FOR CLAIM**'dir. Attention: **GQA dikkat bloğu (grouped-query, mevcut implementasyon)**.
- **Yönlendirme politikası: token-choice top-k.**
- Closure-matrix kapsamı: agi/asi satırları kapsam-dışı pending — `out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 53, 54, 55, 56, 57]` (bkz. [reports/closure_57_matrix.md](reports/closure_57_matrix.md)).
- Çıktı disiplini: **Varsayılan mod `verified`.** Kanıt olmadan hiçbir claim final dokümanlarda kalmaz.

## Hızlı başlangıç
```bash
bash scripts/bootstrap_venv.sh        # pinli bağımlılıklarla .titan-venv (Python 3.11)
bash scripts/verify_all.sh            # offline: 726 passed, 5 skipped + gate'ler (GPU/ağ yok)
bash zero_touch_start.sh --check-only # readiness verdict + engeller (eğitim yok)
```
Tam doğrulama/başlatma akışı: [REPRODUCE_TR.md](REPRODUCE_TR.md).

### Config örneği (mevcut varsayılanlar)
Birkaç temsili `config/config.py` varsayılanı (yalnız dokümantasyon — burada düzenleme yapmayın):
```python
use_torch_compile = False
moe_intermediate = 8192     # ölçülen 3.67B toplamı bu tutar
liquid_layers_idx = [4, 10, 16]
```

## Kanonik yüzeyler (önce bunları oku)
- [AGENTS.md](AGENTS.md) — proje-kapanış anayasası; bu reponun kendi source-of-truth sırasında 1. madde.
- [STATUS_TR.md](STATUS_TR.md) — aşama, readiness, tek boşluk.
- [TRUTH_MATRIX_TR.md](TRUTH_MATRIX_TR.md) — her iddia → kanıt sınıfı (measured/target/vision).
- [BACKLOG_TR.md](BACKLOG_TR.md) — 45K gate + ertelenen post-run bulgular.
- [GOVERNANCE_TR.md](GOVERNANCE_TR.md) — politika/kontrat indeksi + gizlilik duruşu.
- [REPRODUCE_TR.md](REPRODUCE_TR.md) — doğrulama & başlatma komutları.
- [DECISIONS_TR.md](DECISIONS_TR.md) — bilinçli kararlar (değiştirilmeyip-belgelenen bulgular dahil).
- [ARCHITECTURE.md](ARCHITECTURE.md) · [TECHNICAL_REPORT_TR.md](TECHNICAL_REPORT_TR.md) · [MODEL_CARD_TR.md](MODEL_CARD_TR.md).
- Master-truth dokümanları: [docs/PROJECT_MASTER_TRUTH_TR.md](docs/PROJECT_MASTER_TRUTH_TR.md) · [docs/CHESS_ONEFILE_MASTER_TRUTH_TR.md](docs/CHESS_ONEFILE_MASTER_TRUTH_TR.md).
- Dış-reviewer onboarding yolu (yukarıdaki listeden ayrı): [START_HERE.md](START_HERE.md) · [README_SUMMARY_TR.md](README_SUMMARY_TR.md).

## 📂 Proje Yapısı
### Kanonik Yerleşim (Build 30 V2)
```text
Tam tracked-dosya ağacı: docs/PROJECT_STRUCTURE.md
```

## 🚀 İş Birliği ve Ticari Fırsatlar

LLM ön-eğitimi, verimli mimariler (ternary/düşük-bit, MoE, yinelemeli mixer'lar) ve eğitim
altyapısı alanlarındaki rollere ve iş birliklerine açığım.

Bu deponun göstermeyi amaçladığı şey: açıkta kurulmuş ve denetlenmiş uçtan uca bir
ön-eğitim yığını — mimari, veri hattı, damıtma, değerlendirme koşumu, kapanış kapıları ve
iddia disiplini. Bugün neyin *ölçülmüş*, neyin *hedef* ya da *vizyon* olduğu için
[STATUS_TR.md](STATUS_TR.md)'ye bakın; bu ayrım bilinçlidir ve
[TRUTH_MATRIX_TR.md](TRUTH_MATRIX_TR.md) tarafından zorlanır.

**İletişim:** [mert.yunlu08@gmail.com](mailto:mert.yunlu08@gmail.com) — ya da bu depoda
bir issue açın.

## 🤝 Katkılar ve PR Kuralları

Katkılar Apache 2.0 lisansı altında memnuniyetle karşılanır.

Bir pull request açmadan önce:

1. **`bash scripts/verify_all.sh` sıfır regresyonla geçmelidir.** Bu komut tüm pytest
   suite'ini ve kapanış kapılarını (manifest senkronu, doküman-iddia tutarlılığı, secret
   scan, fact drift) koşar. Geçen test sayısını düşüren bir PR birleştirilmez.
2. **Davranış değişikliğiyle birlikte test ekleyin.** Suite sözleşmedir; bu depoda kapatılan
   birkaç bug tam olarak testin gerçek kodu import etmek yerine kopyalaması yüzünden aylarca
   hayatta kalmıştı.
3. **İddia sınırlarını gevşetmeyin.** Ölçülen/hedef/vizyon ayrımı, reason code'lar ve
   `TRUTH_MATRIX_TR.md` taşıyıcı unsurlardır. Ölçülmemiş bir sayıyı ölçülmüş gibi
   etiketlemeyin; gerçek bir koşuyla desteklenmeyen hiçbir yüzeye `trained`,
   `benchmark-verified` veya `production-ready` eklemeyin.
4. **EN/TR ikizlerini birlikte güncelleyin.** `README.md`/`README_TR.md`,
   `BACKLOG.md`/`BACKLOG_TR.md`, `DECISIONS.md`/`DECISIONS_TR.md`,
   `STATUS.md`/`STATUS_TR.md` ve kardeşleri birlikte değişir.
5. **`feature/` veya `fix/` dal öneki kullanın** ve yapay zekâ asistanı katkısını commit
   trailer'ında beyan edin — bkz. [CONTRIBUTING.md](CONTRIBUTING.md).

Tüm katılım [Davranış Kuralları](CODE_OF_CONDUCT_TR.md) kapsamındadır.

Güvenlik açıklarını asla public bir issue'da değil, [SECURITY_TR.md](SECURITY_TR.md) üzerinden
özel olarak bildirin.

## Lisans
Apache License 2.0 — bkz. [LICENSE](LICENSE) (Türkçe bilgilendirme: [LICENSE_TR](LICENSE_TR)). Telif Hakkı 2026 Mert Yunlu. Built with Llama — bkz. [NOTICE](NOTICE) ve [MODEL_LICENSE_TR.md](MODEL_LICENSE_TR.md).
