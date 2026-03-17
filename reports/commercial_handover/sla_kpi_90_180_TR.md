# SLA + KPI Planı (0-90 / 91-180 Gün)

## SLA Temeli
1. P1 incident ilk yanıt: <= 30 dk
2. P2 incident ilk yanıt: <= 4 saat
3. P3 incident ilk yanıt: <= 1 iş günü
4. Kritik güvenlik patch dağıtımı: <= 24 saat
5. Yüksek güvenlik patch dağıtımı: <= 7 gün

## 0-90 Gün KPI'ları
1. Gate pass oranı (`verify_all`): >= %95
2. Release doğrulama zinciri (manifest/checksum/signature): %100
3. Sev-1/Sev-2 incident rollback readiness drill: ayda 1 PASS
4. Lisans/hash audit completion: %100 (aylık)
5. Pilot acceptance paket teslim süresi: <= 2 iş günü

## 91-180 Gün KPI'ları
1. Regression kaçış oranı: <= %2
2. Mean time to recovery (MTTR): <= 2 saat (P1)
3. Doküman-claim tutarlılık gate başarısı: >= %98
4. Üretim benchmark sapması (p95 latency budget): <= %10
5. Change failure rate: <= %10

## İnceleme Frekansı
- Haftalık operasyon review
- Aylık risk/compliance review
- 90/180 gün yönetim raporu
