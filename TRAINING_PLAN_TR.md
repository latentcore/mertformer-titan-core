# MertFormer Titan: Stratejik Eğitim Yol Haritası

Bu belge, Azure A100 altyapısı üzerindeki Titan Eğitim Çalışması (Training Run) için uygulama planını ana hatlarıyla belirtir.

## Aşama 0: Preflight & Güvenlik Gate'leri
**Hedef:** Eğitim öncesi sistem bütünlüğünü ve hazır olma durumunu doğrulamak.
- **Yöntem:** `run.sh --test` + operator mode gate (eğitim donanımında tam mod).
- **Çıktılar:** Preflight logları + operator gate logları.

## Aşama 1: Damıtma Koşusu (Temel)
**Hedef:** **Llama-3.3-70B-Instruct** (Öğretmen) modelinden MertFormer (Öğrenci) modeline bilgi aktarımı.
- **Veri Seti:** ~24 Milyar token (yüksek kalite, KD odaklı müfredat).
- **Yöntem:** Çevrimdışı logits damıtma + **precomputed logits**.
- **Altyapı:** 8x A100 (Founders Hub veya eşdeğeri).
- **Ana ayarlar:** `max_steps=45000`, `max_seq_len=4096`.
- **Sonuç:** Stabil talimat takibi ve sözdizimi açısından sağlam kodlama baseline'ı.

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

## Aşama 4: Değerlendirme ve Benchmarklar
**Hedef:** Eğitim sonrası dahili benchmark çıktıları üretmek.
- **HumanEval/MBPP:** Checkpoint varsa eğitimden sonra otomatik çalışır.
- **Çıktılar:** `reports/benchmarks/` altında JSONL çıktıları.

---
**Durum:** AŞAMA 1 BAŞLATILMAYA HAZIR (eğitim donanımı bekleniyor).
