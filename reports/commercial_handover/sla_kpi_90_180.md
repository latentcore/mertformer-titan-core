# SLA + KPI Plan (0-90 / 91-180 Days)

## SLA Baseline
1. P1 incident first response: <= 30 dk
2. P2 incident first response: <= 4 saat
3. P3 incident first response: <= 1 iş günü
4. Critical security patch deployment: <= 24 saat
5. High security patch deployment: <= 7 gün

## 0-90 Days KPIs
1. Gate pass oranı (`verify_all`): >= %95
2. Release doğrulama zinciri (manifest/checksum/signature): %100
3. Sev-1/Sev-2 incident rollback readiness drill: ayda 1 PASS
4. Lisans/hash audit completion: %100 (aylık)
5. Pilot acceptance paket teslim süresi: <= 2 iş günü

## 91-180 Days KPIs
1. Regression kaçış oranı: <= %2
2. Mean time to recovery (MTTR): <= 2 saat (P1)
3. Doküman-claim tutarlılık gate başarısı: >= %98
4. Üretim benchmark sapması (p95 latency budget): <= %10
5. Change failure rate: <= %10

## Review Cadence
- Haftalık operasyon review
- Aylık risk/compliance review
- 90/180 gün yönetim raporu
