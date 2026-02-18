# KPI Paketi v1 (Build 30)

Bu paket, pilot go/no-go ve kurumsal teknik inceleme için 12 KPI sözleşmesini tanımlar.

## Çıktı Sözleşmesi
- CLI: `mertformer kpi-report --out reports/kpi_report_v1.json`
- Şema: `interfaces/kpi_report_v1.schema.json`
- Rapor Şema Adı: `kpi_report_v1`

## KPI Seti (12)
1. verify_all geçişi
2. secret scan geçişi
3. pytest geçişi
4. preflight geçişi
5. operator gate geçişi
6. pilot şeması mevcut
7. release artefakt varlığı (zip + locked age)
8. swarm omega hazır (45-agent)
9. onnx smoke geçişi
10. smoke benchmark mevcut
11. kaggle compare mevcut
12. claim eligibility gate

## Yorumlama
- `readiness_score` değeri `pass_count / total_count` olarak hesaplanır.
- `>= 0.90` kontrollü pilot için release-ready kabul edilir.
- `< 0.90` durumunda uyarı KPI'ları kapatılmalıdır.

## Notlar
- KPI tamamen kanıt odaklıdır; eğitilmiş checkpoint yoksa model kalite iddiası üretmez.
- ONNX kontrolü `--onnx-check` ile aktif edilir.
