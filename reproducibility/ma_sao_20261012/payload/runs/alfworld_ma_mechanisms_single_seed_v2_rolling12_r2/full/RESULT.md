# Single-seed mechanism study: full

This is a new fixed-endpoint development experiment, not a rerun selected to recover97%.
One training seed; no training-seed robustness claim.

| Saved endpoint | Selected Harness + Solver | Solver without Harness | New valid proposals |
| ---: | ---: | ---: | ---: |
| 96 | 180/200 (90.0%) | 178/200 (89.0%) | 7/8 |
| 144 | 186/200 (93.0%) | 179/200 (89.5%) | 7/8 |

**Primary endpoint:144, fixed before training.**
The selected Harness is frozen on selection48 before its main200 evaluation.
A carried incumbent is not described as a new learned Harness.

Native training and actual work: [training audit](./TRAINING_AUDIT.json).
Machine-readable endpoint reports and bindings: [result](./RESULT.json).

Single training seed at user request; question bootstrap is not training-seed robustness.
Original200 and selection48 are reused development questions.
New full control does not reproduce the historical97% initialization with already-trained MA critics.
Always-joint matches generated/environment budgets, not consumed updates or total compute.
Historical97% also differs in the round48 Adam reset and24 versus16 proposal-selection opportunities.
No main200 checkpoint selection or performance-based retries.


Twelve-round paired native recovery was enabled. All interrupted attempts, speculative work and retained/deleted-generation receipts are preserved under `attempts/` and `recovery/`; inference restarts are not bitwise rollout replay.

Evaluation was interrupted and resumed without repeating completed episodes or generating replacement Harness candidates. Input-bound recovery evidence and additional incomplete episode costs are retained in RESULT.json/evaluation_recovery.
