# YÖNETİŞİM (GOVERNANCE) — politikalar, kontratlar, değişim kontrolü

Repo'nun yönetişim yüzeyleri için tek indeks (önceden `reports/` içinde dağınıktı).
English: [GOVERNANCE.md](GOVERNANCE.md). Bilinçli kararlar: [DECISIONS_TR.md](DECISIONS_TR.md).

## Değişim kontrolü
- Commit'ler scoped + reason-coded; CI gate adı `verify` (bkz. [REPRODUCE.md](REPRODUCE.md)).
- Yok-etme yok politikası: superseded/toplu içerik **taşınır** (`archive/`, `private/`) ya da **gitignore** edilir (generated), asla hard-delete edilmez — git history kurtarma yüzeyidir.
- Davranış-değiştiren fix'ler ilk 45K koşusundan sonraya ertelenir ve [DECISIONS_TR.md](DECISIONS_TR.md) / [BACKLOG_TR.md](BACKLOG_TR.md)'de belgelenir (koşu confound olmasın).

## Repo politikaları (tam metin `reports/` altında)
- `reports/automation_boundary_policy.md` — otomasyon ne yapabilir/yapamaz.
- `reports/change_control_sop.md` — değişim-kontrol SOP.
- `reports/update_first_policy.md` — yeni-eklemeden-önce-mevcudu-güncelle.
- `reports/system_memory_policy.md` — kalıcı-hafıza politikası.
- `reports/folder_structure_policy.md` + `reports/repo_directory_contract.md` — yerleşim kuralları.
- `reports/backlog_operating_contract.md` — backlog işletim kuralları.
- `reports/code_truth_contract.md` — kod-doküman gerçek hizalaması.
- `reports/support_maintenance_policy.md` — destek/bakım sınırı.

## Kontratlar (makine-denetimli yüzeyler, tam metin `reports/` altında)
- `run_contract.md`, `checkpoint_contract.md`, `data_pipeline_contract.md`, `logger_contract.md`, `plot_contract.md`, `benchmark_contract.md`, `kpi_contract_build30.md`, `post_train_automation_contract.md`.

## Gizlilik / görünürlük
- Ana repo ve ayrı dealroom reposu **PRIVATE**'dır. Tek public yüzey README türevli Gist'tir.
- **Güncellendi (2026-07-30, Apache 2.0 relicensing pass'i):** README.md/README_TR.md artık
  public release hazırlığının bir parçası olarak kısa, bilinçli bir "Hiring & Commercial
  Opportunities" bölümü taşıyor (rol ilgisi + gerçek bir iletişim adresi) — bkz.
  [DECISIONS_TR.md](DECISIONS_TR.md). Daha derin yatırımcı/pitch/outreach materyali (deck'ler,
  one-pager'lar, sponsorluk mesajlaşması) teknik repo dışında, `private/commercial/` ve ayrı
  dealroom reposunda kalmaya devam ediyor.
