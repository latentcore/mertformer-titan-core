# Chess Teaching Contract Report

- generated_utc: `2026-06-11T17:46:43Z`
- contract_version: `1.0`
- all_green: `True`
- case_pass: `5/5`
- mode_pass: `5/5`
- level_monotonic_non_decreasing: `True`

## Case Results

| Case | Status | Mode | Level | Move | Tags | Problems |
| --- | --- | --- | --- | --- | --- | --- |
| `center_control` | `pass` | `teach` | `club` | `e2e4` | `center_control` | — |
| `development` | `pass` | `teach` | `club` | `g1f3` | `development` | — |
| `capture_check` | `pass` | `teach` | `club` | `h5f7` | `capture, check, activity` | — |
| `castle` | `pass` | `analyze` | `advanced` | `e1g1` | `castle` | — |
| `promotion` | `pass` | `turkish_teach` | `basic` | `a7a8q` | `promotion, check, checkmate` | — |

## Mode Results

| Mode | Status | Short Prefix |
| --- | --- | --- |
| `play` | `pass` | `Oyun modu` |
| `teach` | `pass` | `Öğretme modu` |
| `analyze` | `pass` | `Analiz modu` |
| `turkish_teach` | `pass` | `Türkçe öğretme modu` |
| `benchmark` | `pass` | `Benchmark modu` |

## Level Results

| Level | Reason Count | Reasons |
| --- | --- | --- |
| `basic` | `2` | `gelişimi hızlandırıp taş koordinasyonunu iyileştiriyor; merkez kareler üzerinde daha güçlü kontrol kuruyor` |
| `club` | `3` | `gelişimi hızlandırıp taş koordinasyonunu iyileştiriyor; merkez kareler üzerinde daha güçlü kontrol kuruyor; tek hamlede birden fazla tehdit hattını canlandırıyor` |
| `advanced` | `4` | `gelişimi hızlandırıp taş koordinasyonunu iyileştiriyor; merkez kareler üzerinde daha güçlü kontrol kuruyor; tek hamlede birden fazla tehdit hattını canlandırıyor; rakip vezire tempo kazandıran baskı uyguluyor` |

- level_monotonic_non_decreasing: `True`
- This is a local contract and explanation-faithfulness smoke layer. It does not replace trained-model benchmark evidence.
