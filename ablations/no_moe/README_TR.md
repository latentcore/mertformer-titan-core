# Ablation: MoE Kapalı

**Amaç**: MoE routing kapatılarak dense-only baseline oluşturmak.

**Config değişikliği**:
- `use_moe: false`
- `num_experts_per_tok: 0` (gerekirse)

**Durum**: Henüz koşulmadı. Sonuçları `ablations/results.md` dosyasına kaydedin.
