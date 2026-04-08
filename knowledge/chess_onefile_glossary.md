# Chess Onefile Glossary

- `feature bundle`: named overlay that turns on a coordinated group of flags.
- `explicit override`: per-flag enable/disable applied after profile and bundle resolution.
- `auxiliary head`: optional chess-specific supervision head such as `phase_head`, `wdl_head`, or `legality_head`.
- `self-play report`: internal artifact generated from model-vs-itself games.
- `inference-mode tournament`: internal comparison between search-assisted and pure-policy inference modes of the same model.
- `replay-buffer manifest`: capped manifest of positions harvested from internal self-play.
- `artifact truth matrix`: machine-readable list of expected output artifacts and whether they exist.
- `run status manifest`: compact operational status snapshot for a finished run.
- `postrun analysis manifest`: compact summary of curated suite, stockfish, self-play, tournament, and replay-buffer surfaces.
- `run contract`: frozen operator-facing execution contract and claim boundary for one exact run.
- `release snapshot`: internal release-surface readiness summary for one exact run.
- `evidence pack stub`: compact list of present evidence versus still-missing external proof.
- `final truth registry`: claim list that labels each chess surface as measured, internal-only, partial, or not eligible.
- `claim registry`: per-claim mapping from classification to supporting evidence.
- `known limits`: explicit list of what the exact run still does not prove.
- `support matrix`: run-local matrix of supported modes, profiles, and artifact surfaces.
- `release gate summary`: compact pass/fail summary of internal and external release gates.
