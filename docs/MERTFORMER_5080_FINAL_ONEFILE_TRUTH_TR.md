# MertFormer 5080 Final Onefile Truth

Bu doküman, ana repoya promote edilen final 5080 onefile hattının repo-side truth boundary durumunu takip eder.

## Mevcut Konum

- Kanonik repo scripti: `scripts/mertformer_5080_final_onefile.py`
- Delivery helper: `scripts/build_mertformer_5080_final_delivery.py`
- Sonuç decrypt helper: `scripts/decrypt_mertformer_result_package.py`
- Varsayılan operatör profili: `safe_5080`
- Opsiyonel agresif profil: `challenge_5080`

## Aktif Model Yolu

- Aktif çalışma zamanı model sınıfı `RepoParityMertFormerModel` yoludur.
- Eski onecell iskeletinden kalan compatibility kodu `LegacyOnecellMertFormerTiny` olarak korunur.
- Compatibility kodu fallback/reference amacıyla tutulur; varsayılan aktif eğitim yolu değildir.
- Aktif mimari yol, `bitlinear`, `mla`, `moe`, `liquid`, `mertformer_block` ve `model.transformers` tabanlı embedded repo-backed stack'tir.

## Deneysel Politika

Deneysel/bilişsel bileşenler kod tabanında korunur; ama dürüst biçimde ele alınır:

- korunur, sessizce silinmez
- uygun yerlerde experimental/feature-flag olarak açıkça işaretlenir
- ölçülmüş benchmark kanıtı olmadan frontier kalite veya Gemma geçme iddiasının parçası yapılmaz

Bu özellikle şu katmanlar için geçerlidir:

- `GlobalWorkspaceBroadcast`
- `HebbianPlasticityLayer`
- `NeuroSymbolicLayer`
- `ContinuousLatentODEStateChannel`
- `NeuromodulatoryGainLayer`
- `LifelongSafetyLayer`
- `world_model_head`

## Claim Boundary

İzin verilen repo-side claim'ler:

- repo içinde kanonik genel amaçlı bir 5080 onefile hattı vardır
- bu hat syntax/test/parity ve delivery helper yüzeyine sahiptir
- smoke/evidence/package akışları yerelde çalıştırılabilir

Ölçülmüş kanıt olmadan kapalı kalan claim'ler:

- "Gemma-2B geçildi"
- frontier-grade kalite üstünlüğü
- yalnızca smoke koşusundan release-grade güç sonucu çıkarmak
- reverse engineering'in imkansız olduğu iddiası
