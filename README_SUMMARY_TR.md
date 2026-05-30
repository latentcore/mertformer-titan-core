![MertFormer Titan Header](assets/header.png)

Dil: [English](README_SUMMARY.md) | [Türkçe](README_SUMMARY_TR.md)

---

# MertFormer Titan - Dış Özet (Build 30 V2)

## Bu Proje Nedir?
MertFormer Titan; low-bit runtime altyapısı, yerel assistant foundation ve disiplinli evaluation yüzeyleri etrafında kurulan offline-first, denetlenebilir bir AI systems reposudur.

Uzun vadeli hedef: compute bütçesi sınırlı geliştiriciler, küçük ekipler ve yerel kurumlar için denetlenebilir AI eğitim/çıkarım maliyet bariyerini düşürmek. Bu, checkpoint-bound koşular ve hedef donanım ölçümleri oluşana kadar benchmark claim değil, target statüsündedir.

## Mevcut Exact Durum
- Aşama: `pilot-ready pre-training baseline`
- Repo-side readiness: `TRAIN_ALLOWED`
- Exact reason code: `READY_REMOTE_BOOTSTRAP`
- Önerilen repo-side lane: `remote_bootstrap`
- Sıkı yerel lane: `offline_clean`
- Tercih edilen ciddi doğrulama hedefi: `45K`
- Başvuru kapısı: gerçek owned training run + checkpoint-bound evidence
- Kalan non-winning blocker'lar: `offline_clean:PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`, `online_teacher:MISSING_HF_TOKEN`

## İnceleme İçin Kritik Noktalar
- Henüz trained checkpoint claim’i yapılmıyor.
- Trained checkpoint oluşmadan benchmark durumu `NOT ELIGIBLE FOR CLAIM` olarak kalır.
- Exact `45K` tercih edilir; ama başvuru readiness, meaningful real training run + checkpoint-bound evidence ile tanımlanır.
- Export/device evidence güçlü artıdır; hard blocker değildir.

## En Kısa İnceleme Yolu
1. [START_HERE.md](START_HERE.md)
2. [docs/PROJECT_MASTER_TRUTH.md](docs/PROJECT_MASTER_TRUTH.md)
3. [reports/final_truth_matrix.md](reports/final_truth_matrix.md)
4. [reports/known_limits_v1.md](reports/known_limits_v1.md)
5. [reports/systems_performance_case_study.md](reports/systems_performance_case_study.md)
6. [reports/offline_assistant_case_study.md](reports/offline_assistant_case_study.md)
7. [reports/chess_proof_teaching_case_study.md](reports/chess_proof_teaching_case_study.md)
8. [applications/anthropic/README.md](applications/anthropic/README.md)

## Kanonik Komutlar
```bash
bash scripts/bootstrap_venv.sh
bash scripts/verify_all.sh
bash zero_touch_start.sh --check-only
bash zero_touch_start.sh
bash scripts/final_one_shot.sh
```

## En Güçlü Sinyaller
- training efficiency ve systems-debugging disiplini
- backend routing ve fallback dürüstlüğü
- governance-gated, offline-first assistant foundation
- claim-safe verification ve repo truth sync
- düşük maliyetli denetlenebilir eğitim/çıkarım için compute erişilebilirliği konumlandırması; target statüsü measured evidence'dan ayrı tutulur

## Hâlâ Açık Post-Run Evidence Sınıfı
- trained final weights
- best/latest checkpoint proof
- checkpoint-bound benchmark outputs
- trained demo bundle
- trained export/device measurements
