# Mimari Kararlar

- **Config disiplini**: `config.py` çalışma zamanı varsayılanı; isteğe bağlı YAML overlay desteği.
- **Swarm mimarisi**: Hedef mimari olarak dokümante; core training'e zorunlu değil.
- **Multimodal**: Metin çekirdeği kanıtlanana kadar ertelendi.
- **MLA → GQA yeniden adlandırma (2026-06-16, DONE)**: Attention sınıfı `layers/mla.py` içinde artık `GQA` (her zaman grouped-query'di; latent-MLA bottleneck değildi). Tam ayrıntı için `DECISIONS.md`.
- **Doğrulanmış bulgular (2026-06-16) — belgelendi, davranış değişikliği 45K sonrasına ertelendi**: z-loss efektif ağırlığı `2e-6` (çift çarpım: `1e-4 * 0.02`), Liquid `dt=1.0` sabit (gated RNN), `epoch_mode` varsayılan `False`, sequential MoE dispatch + `.item()` (yalnız GPU-perf), `mark_weights_updated()` uncalled (cache-invalidation kancası — silinmez). Hiçbiri bu turda değiştirilmedi (45K koşusunu confound etmemek için). Mekanizma + post-45K aksiyon: `DECISIONS.md`.
