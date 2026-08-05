# Chess Onefile Master Truth

Bu doküman, chess onefile hattının mevcut repo-side kapanış durumunu özetler.

Bilerek şu ayrımı korur:

- repo-side implementasyon tamamlığı
- gerçek eğitim/ölçüm kanıtı
- harici doğrulama
- release-grade kapanış

## Geçerli Kural

- `repo-side complete`, kodun, testlerin ve doküman yüzeyinin onefile closure zincirine gerçekten bağlandığı anlamına gelir.
- `real closure blocked`, gerçek koşu, ölçülmüş artefakt, harici imza/doğrulama veya yönetim kararı hâlâ eksik demektir.

## Toplulaştırılmış Master Tablo

| Lane | Repo-Side Complete | Real Closure Blocked | Hâlâ Neden Bloklu |
|---|---|---|---|
| `release_registry` | `true` | `true` | İç registry yüzeyleri var, ama release-grade kanıt yerel artefaktlarla verilmez. |
| `external_closure` | `true` | `true` | Harici reproducibility, hukuk, güvenlik ve pilot kapanışı hâlâ dış iş akışlarıdır. |
| `operational_closure` | `true` | `true` | Operatör handbook, DR kanıtı, retention policy ve blind handoff hâlâ gerçek prova ister. |
| `release_governance` | `true` | `true` | Release notes, freeze sign-off, changelog review ve maintenance policy hâlâ formel yönetişim ister. |
| `device_packaging` | `true` | `true` | Export truth, device validation, packaging validation ve installer validation hâlâ ölçülmüş koşu ister. |
| `benchmark_closure` | `true` | `true` | Raw output, compare report, benchmark summary ve locked manifest hâlâ gerçek benchmark kapanışı değildir. |
| `training_accounting` | `true` | `true` | Training report, token accounting, compute accounting ve cost reporting hâlâ gerçek ölçüm ister. |
| `trained_artifact_truth` | `true` | `true` | Final weights, best/latest checkpoint truth ve trained artifact registry hâlâ doğrulanmış trained output ister. |
| `management_closure` | `true` | `true` | Core-complete, research separation, maintenance-only ve final closure kararları hâlâ yönetim imzası ister. |
| `truth_docs_alignment` | `true` | `false` | Kanonik chess/project truth docs ile üretilen truth raporları şu anda senkron durumdadır. |
| `project_actionability` | `true` | `false` | Proje blocker’ları, repo-side truth zincirine bağlanmış kapsamlı bir actionability rapor kümesiyle (execution order, dependency graph, lane board, phase plan ve ~40 ilgili rapor yüzeyi; tam liste için aşağıdaki alt bölüme bakın) ilişkilendirilmiştir. |
| `generated_truth_consistency` | `true` | `false` | Üretilen summary raporları ile blocker/action-plan katmanları şu anda kendi içinde tutarlıdır. |

## `project_actionability` Rapor Yüzeyleri

Proje blocker’larını repo-side truth zincirine bağlayan actionability rapor kümesi:

- execution order, dependency graph, lane board, phase plan
- phase readiness scoreboard, owner accountability matrix, owner work queue
- critical path report, owner next-actions summary, ready-now board
- unlock-impact report, parallel workset report, phase-exit criteria report
- execution-wave report, evidence-backlog report, dependency-bottleneck report
- owner-phase-frontier report, evidence-criticality report, phase-transition matrix
- owner-load report, phase-dependency-pressure report, owner-bottleneck-alignment report
- evidence-phase-heatmap report, blocker-risk-register report, release-prereq matrix report
- foundation-run-dependency report, release-path report, external-closure-cluster report
- owner-evidence-gap report, release-gate-dependency report, external-signoff-queue report
- release-evidence-bridge report, training-run-readiness report, benchmark-closure-dependency report
- release-decision-queue report, external-validation-readiness report, artifact-lock-readiness report
- final-release-cutover report, real-run-execution-queue report, benchmark-evidence-lock report
- final-signoff-cutset report

## Repo-Side Complete Yüzeyler

- Feature-bundle ve feature-flag sistemi
- Yardımcı satranç head’leri
- Self-play, tournament ve replay-buffer rapor yüzeyleri
- Closure manifest zinciri
- Release/evidence registry
- Claim registry, known limits, support matrix, release gate summary
- RC/golden/handoff stub yüzeyleri
- External, pilot, security, legal stub yüzeyleri
- Operator handbook, DR, backup retention, blind handoff stub yüzeyleri
- Release-governance stub yüzeyleri
- Device/export/packaging stub yüzeyleri
- Benchmark-closure stub yüzeyleri
- Training/accounting stub yüzeyleri
- Trained-artifact-truth stub yüzeyleri
- Management-closure stub yüzeyleri
- Master summary ve aggregated truth yüzeyleri

## İlgili, ayrı paket: `ChessFormerAI/chessformer`

Bu hattın bağımsız bir yeniden yazımı (`ChessFormerAI/chessformer`, bu reponun `layers/`'ının
salt-okunur bir kopyasına karşı geliştirildi, bu reponun parçası değil), hedef tüketici
donanımında (RTX 5070) gerçek, checkpoint-bağlı bir eğitim koşusuna ve gerçek ölçülmüş
değerlendirmeye sahip: puzzle accuracy %45.78 (DeepMind'ın Searchless Chess'ine, arXiv:
2402.04494, doğrudan kıyaslanabilir) ve Stockfish'in kendi iç rating ölçeğinde 1509 Elo tahmini
(DeepMind'ın 2895 Lichess-blitz-insana-karşı rakamına kıyaslanamaz). Tam detay:
`evidence/2026-08-02-chess-searchless-5070/`, `BACKLOG_TR.md`, `DECISIONS_TR.md`.

Bu, yukarıdaki bu dokümanın kendi tablosunda **hiçbir şeyi değiştirmiyor** —
`scripts/chess_5080_onefile.py`'nin kendisi değişmedi, hâlâ hiç koşulmadı, ve iki paketin
farklı rapor şemaları, farklı governance apparatus'u var, ve (2026-08-06 itibarıyla) birleşmek
yerine ayrı kalma konusunda bilinçli bir karar var. `chessformer`, daha önceki bağımsız bir
incelemenin `scripts/chess_5080_onefile.py`'de bulduğu dört gerçek bug'ı düzeltti (`is_causal`
yanlış-etiketleme, Liquid clamp eğitim/değerlendirme uyuşmazlığı, MoE dispatch kapasite
drift'i, Liquid-state-atma bug'ı); bu bug'lar `chess_5080_onefile.py`'nin kendisinde hâlâ canlı.

## Gerçekte Kalan Çekirdek İşler

Repo-side kapanış güçlü olsa da aşağıdakiler gerçek blocker olarak durur:

- `external_strength_unproven`
- `real_training_outputs_pending`
- `external_reproduction_pending`
- `security_legal_pilot_pending`
- `operator_handoff_dr_pending`
- `rc_golden_final_release_pending`
- `export_device_packaging_pending`
- `benchmark_evidence_pending`
- `trained_artifact_truth_pending`
- `management_closure_pending`

## Alt Sonuç

- Chess onefile hattında artık güçlü bir repo-side closure framework var.
- Repo artık önceye göre çok daha denetlenebilir ve operasyonel durumda.
- Ama bu hattı tamamen bitmiş ilan etmek hâlâ dürüst olmaz.
- Eksiklerin ana sınıfı artık `kod yüzeyi eksik` değil.
- Eksiklerin ana sınıfı artık `gerçek kanıt, harici doğrulama ve final kapanış kararı eksik`.
