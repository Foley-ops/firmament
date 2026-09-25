# Historical artifacts — Genesis v0.1

All runs made before 2026-09-25 used the **genesis-v0.1** ruleset (git tag
`genesis-v0.1-historical`). They are frozen: files were made read-only and their
SHA-256 hashes are recorded in `runs-v0.1-sha256.txt` (run from inside `runs/`).

To reproduce one, check out the tag; current code will refuse to load these
snapshots because the config hash now covers rule-file contents.

Status labels:
- `20260911-070117-shakeout-72d741` — M0 dead world, 1000 sim-days. Exploratory: ran with
  the v0.1 dead-era chemistry speed-up (rates ×10 while lifeless), so chemistry timing
  is not physical.
- `20260911-200738-genesis-57719a` — M1 attempt 1. INVALID as evidence of evolution
  (zero-length polymer artifact); see docs/M1_ATTEMPTS.md.
- All other runs — shakeout, crashed, or test runs.
