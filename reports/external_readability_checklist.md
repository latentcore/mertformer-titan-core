# External Readability Checklist

Use this checklist for a 10-20 minute outside-reader test.

## Reader Should Be Able To Answer
1. What is this repository trying to build?
2. Which parts are measured and which are not?
3. What is the current exact readiness verdict?
4. What is the primary remaining blocker class?
5. What is the strongest systems/performance story?
6. What is the offline assistant story?
7. What is the chess lane for?
8. Which commands are canonical?
9. What still requires a real training run and checkpoint-bound evidence?
10. Which role is the application pack primarily targeting?

## Reader Should Find These Docs Without Search Fatigue
- `START_HERE.md`
- `README_SUMMARY.md`
- `docs/PROJECT_MASTER_TRUTH.md`
- `reports/final_truth_matrix.md`
- `reports/known_limits_v1.md`
- `reports/systems_performance_case_study.md`
- `reports/offline_assistant_case_study.md`
- `reports/chess_proof_teaching_case_study.md`

## Failure Conditions
The package is not externally readable yet if the reviewer says any of the following:
- "I still don't know what is actually measured."
- "I can't tell what the main product direction is."
- "I see lots of documents, but I don't know which ones are canonical."
- "I still don't know what would happen after the real training run."
- "I can't tell whether this is primarily a systems repo or a bundle of unrelated lanes."

## Pass Condition
A pass means the reviewer can summarize the repo in one paragraph without inventing claims the repo does not make.
