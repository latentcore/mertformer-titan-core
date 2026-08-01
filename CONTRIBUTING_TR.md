# Katkıda Bulunma

Bu depo **Apache License 2.0** ile lisanslanmıştır (bkz. [LICENSE](LICENSE), Türkçe
bilgilendirme: [LICENSE_TR](LICENSE_TR)) ve **dış katkılara açıktır**. Bir katkı
gönderdiğinizde, Apache 2.0'ın 5. maddesi uyarınca katkınızın aynı şartlarla
lisanslandığını kabul etmiş olursunuz.

Her pull request `bash scripts/verify_all.sh` komutunu sıfır regresyonla geçmelidir.
Tam liste için [README_TR.md](README_TR.md) içindeki "Katkılar ve PR Kuralları"
bölümüne bakın.

Geliştiren: Mert Yünlü. Uygulama için AI kod asistanları (Claude Code) kullanıldı; tüm mimari, tasarım kararları ve nihai inceleme yazarın kendisine aittir.

## Commit Mesajı Stili

Başlık: `<tip>: <kısa, emir kipi özet>` — tip şunlardan biri:
`fix|docs|feat|refactor|test|chore|harden|closure`.

Gövde:
1. Ne değişti ve neden (asıl bulgu/motivasyon).
2. Araştırma gerektiyse: nasıl bulunup kapsamlandı (özellikle kapsam
   makul şekilde sorgulanabilecekse).
3. Bilerek dokunulmayan bir şey varsa, ve neden.
4. Somut sayılar içeren bir `Verified:` satırı — "testler geçti" değil,
   gerçek `N passed, M skipped` ve hangi gate'lerin çalıştığı.
5. AI-destekliyse `Co-Authored-By: <model adı> <email>` trailer'ı,
   o commit'i gerçekte yapan modeli/aracı adlandırarak.

Ayrıntı seviyesi diff boyutuna değil, risk/belirsizliğe göre ölçeklenir.
Örnek (commit `0b4f79d6`):

```
fix: remove dead env var in check_overlay_validity.py + add cuda to chess GUI --device

scripts/check_overlay_validity.py::check_overlay(): removed a dead, unused
env dict (hardcoded Unix-only PATH: "/usr/bin:/bin:/usr/local/bin") that was
assigned but never passed to subprocess.run() (the call already uses
full_env instead) -- confirmed via `ruff --select F841` (repo's default lint
scope, pyproject.toml's select = ["E9", "F821", "F822", "F823"], deliberately
excludes this class to avoid low-signal churn across legacy scripts; this
one instance was fixed directly since it was already hand-identified).

apps/chess_gui/play_mertformer_chess_web.py: --device CLI flag's argparse
choices was missing "cuda" (only ["cpu", "mps"]) -- the auto-detect path
(choose_device()) already checks CUDA first, but a user could not force it
explicitly.

A repo-wide ruff/bandit discovery pass (130 unused-import/variable findings,
543 low/medium bandit findings) was run but deliberately not acted on beyond
the one instance above -- both fall inside the project's own documented
no-broad-cleanup lint policy and outside this pass's layers/model/train/
orchestrator/mertformer_sdk no-touch boundary.

Verified: tests/test_chess_gui_contract.py + tests/test_config_overlay_strict.py
(10 passed), scripts/check_overlay_validity.py run directly (4/4 overlays OK),
full suite 721 passed, 9 skipped, 1 xfailed (identical count, no regression),
doc-claim consistency OK (after reverting a locally-regenerated, never-committed
reports/train_readiness_decision.{json,md} back to canonical -- known
machine-local-state artifact, not a real doc bug).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
```

(Örnek commit mesajı, İngilizce reponun asıl commit metnini birebir koruyor — çeviri, gerçek kaydı bozmamak için yapılmadı.)

## Dahili İş Akışı
- Branch isimlerinde `feature/` ve `fix/` prefix'leri kullan.
- Tüm claim/metrikleri **pre-training / doğrulama bekliyor** statüsüyle hizalı tut.
- Büyük binary/dataset commit'leme; script ve yeniden üretilebilir config tercih et.
- Kullanıcıya dokunan bir davranış değiştiyse ilgili README/MD dosyasında kısaca belirt.

## Güvenlik
Güvenlik sorunlarını gizli bildir — bkz. [SECURITY.md](SECURITY.md). Public issue açma.
