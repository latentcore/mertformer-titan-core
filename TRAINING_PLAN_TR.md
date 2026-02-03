# MertFormer Titan: Stratejik Eğitim Yol Haritası

Bu belge, Azure A100 altyapısı üzerindeki Titan Eğitim Çalışması (Training Run) için uygulama planını ana hatlarıyla belirtir.

## Aşama 1: Prototip İnce Ayar (Bilgi Damıtma - Distillation)
**Hedef:** Llama-3-70B (Öğretmen) modelinden MertFormer-1.58bit (Öğrenci) modeline bilgi aktarımı.
- **Veri Seti:** 10 Milyar Token (Yüksek kaliteli kodlama talimat setleri).
- **Yöntem:** Çevrimdışı Logits Damıtma (Statik).
- **Altyapı:** 8x A100 (Founders Hub).
- **Sonuç:** Temel talimat takibi ve sözdizimi açısından kusursuz kodlama yapabilen bir model.

## Aşama 2: Ajan Entegrasyonu (Sürü - Swarm)
**Hedef:** Modeli çoklu ajan rolleri için özelleştirmek.
- **Veri Seti:** "Mert Arşivi" (15 Milyon Tokenlık Kişisel RAG) + Role Özel İnce Ayar (QA, Güvenlik, Mimar).
- **Yöntem:** Her bir ajan rolü için LoRA Adaptörleri.
- **Sonuç:** Aynı temel modelden doğan farklı ajan kişilikleri (Örn: "Paranoyak" Güvenlik Görevlisi ile "Yaratıcı" Tasarımcı).

## Aşama 3: Performans Optimizasyonu (Bilge Döngüsü - Sage Loop)
**Hedef:** Üretim zorlaştırması (hardened) ve kendi kendine gelişme.
- **Mekanizma:** "Wisdom Loop" - Başarılı proje günlükleri ve analiz raporları üzerinde eğitim.
- **Yöntem:** Derleyici Geri Bildiriminden Takviyeli Öğrenme (RLCF).
- **Sonuç:** Aynı hatayı iki kez yapmayan, kendi kendini iyileştiren bir sistem.

---
**Durum:** AŞAMA 1 BAŞLATILMAYA HAZIR.
