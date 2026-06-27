# Feature Flag Yönetişimi

- generated_utc: `2026-06-27T22:17:44Z`
- kapsam: 45K mimari doğrulama koşusu için closure-mode operasyon yönetişimi
- iddia sınırı: flag'ler kontrol edilebilir kod yollarını tanımlar; koşu kanıtı olmadan benchmark, gecikme, enerji, deployment veya eğitilmiş model iddiası oluşturmaz

## Kanonik Ana Hat

- `zero_touch_start.sh` -> `scripts/final_orchestrator.py`
- Bu geçiş için önerilen eğitim hattı: `remote_bootstrap`
- `TITAN_OFFLINE=1`, `TITAN_REQUIRE_GATED_TEACHER=1` ve `TITAN_USE_PRECOMPUTED_LOGITS=1`, katı offline-clean hattını tanımlar. Bu hat teacher-tokenizer KD'dir ve `TITAN_USE_TR_TOKENIZER=1` AYARLAMAZ (burada TR'yi zorlamak tokenizer-identity drift'ine yol açar; bkz. scripts/build_train_readiness_contract.py).
- `remote_bootstrap`, `TITAN_OFFLINE=0` kullanır ve runtime credential injection ile hedef makinede dataset/bootstrap çalıştırmasını varsayar.

## Closure-Safe Hız Kontrolleri

Bu flag'ler yalnızca operatör kontrollü hedef makine ayarlarıdır. Muhafazakâr baseline, projection flag'leri kapalı tutularak ve `TITAN_LIQUID_TRAIN_IMPL=baseline` kullanılarak korunur.

| Flag | Varsayılan | Durum | Kapsam | Gate / rollback |
| --- | --- | --- | --- | --- |
| `TITAN_BATCH_SIZE` | `128` | opt-in override | `config/config.py` üzerinden global eğitim batch size | Ocean 2x H200 `1024` ile başlar; yalnız net OOM'da düşür. |
| `TITAN_BATCH_SIZE_FALLBACKS` | unset | opt-in launcher policy | `scripts/final_orchestrator.py` içinde net-OOM-only batch retry sırası | Ocean 2x H200 `1024,512,256` kullanır; OOM olmayan hata durur. |
| `TITAN_LOG_INTERVAL` | `1` | opt-in override | Eğitim log aralığı | Config varsayılanına dönmek için unset. |
| `TITAN_VAL_CHECK_INTERVAL` | `1000` | opt-in override | Validation aralığı | Config varsayılanına dönmek için unset. |
| `TITAN_SAVE_INTERVAL` | `1000` | opt-in override | Checkpoint aralığı | Config varsayılanına dönmek için unset. |
| `TITAN_DATALOADER_PIN` | `1` | varsayılan açık, runtime-safe | CUDA mevcutsa DataLoader pinned-memory transferi | Worker veya host memory kararsızsa `0` yap. |
| `TITAN_DATALOADER_NONBLOCKING` | `1` | varsayılan açık, runtime-safe | Açık tensor transferlerinde non-blocking davranış | Transfer davranışı debug edilecekse `0` yap. |
| `TITAN_FFN_PACK` | `0` | opt-in deney | FFN gate/up packed projection yolu | `tests/test_packed_projection_equivalence.py` gerekir; rollback için `0`. |
| `TITAN_MOE_PACK` | `0` | opt-in deney | MoE BitSwiGLU gate/up packed projection yolu | `tests/test_packed_projection_equivalence.py` gerekir; rollback için `0`. |
| `TITAN_MLA_KV_PACK` | `0` | opt-in deney | MLA K/V packed projection yolu | `tests/test_packed_projection_equivalence.py` gerekir; rollback için `0`. |
| `TITAN_LIQUID_FAST_PATH` | `1` | hedef makine seçici | Liquid fast path seçici | Ocean ilk uzun koşusunda `0`; ancak ölçülü smoke sonrası geri aç. |
| `TITAN_LIQUID_TRAIN_IMPL` | `baseline` | opt-in seçici | Liquid eğitim implementasyonu varyantı | `tests/test_liquid_safeguard.py` gerekir; rollback için `baseline`. |
| `ACCELERATE_CONFIG_FILE` | unset | opt-in runtime config | Accelerate dağıtık launch config'i | `repro/accelerate_8xgpu.yaml` yalnızca uyumlu 8 GPU hedef makinelerde kullanılır. |

Önemli sınırlar:
- Packed projection yolları varsayılan kapalıdır.
- Ocean 2x H200 launch profili, equivalence testleri sonrası packed projection yollarını açabilir; low-bit ve fused-backward kernel yolları kapalı kalır.
- Batch fallback yalnızca retry orchestration davranışıdır; teacher, tokenizer, dataset, loss veya model mimarisini değiştirmez.
- İlk uzun koşu için `packed_pair_compile` ertelenir.
- `MERTFORMER_LOWBIT_KERNEL=1` açıksa FFN/MoE/MLA packed projection yolları, deneysel low-bit inference kernel sınırını geçmemek için baseline yola düşer.
- `repro/accelerate_8xgpu.yaml`, stabil model config sözleşmesi değil, yeniden üretilebilirlik/koşu config'i olduğu için `repro/` altında durur.
- Bu flag'lerden gelen herhangi bir hız sayısı; hedef makine logu, komut, donanım, commit ve çıktı artefaktına bağlanmadan ölçülmüş iddia olarak yazılamaz.

## Opsiyonel Hız Flag'leri Öncesi Zorunlu Doğrulama

```bash
python3 -m pytest -q tests/test_packed_projection_equivalence.py tests/test_liquid_safeguard.py
bash scripts/verify_all.sh
bash zero_touch_start.sh --check-only
```

Hedef makine smoke için uzun 45K koşusundan önce `--dry-run` veya operatör onaylı kısa smoke tercih edilir. Geçen smoke logu olmadan tüm opsiyonel flag'leri kanonik uzun hatta açmayın.

## Kanonik Olmayan / Ertelenenler

- TPU/XLA, multimodal, TurboQuant, sequence packing, async checkpointing ve scale-up hatları ayrı test edilmiş karar ile yükseltilmedikçe phase-2, external veya post-evidence kapsamındadır.
- `run.sh` yalnızca yardımcı giriş olarak kalır; kanonik launcher'ın yerine geçmemelidir.
