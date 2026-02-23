# POST Mode Oncall (Entropism Chain)

## 1) Activation and lock
- POST trigger signals: `moltbook`, `post`, `thread`, `gonderi`, `paylasim`, or post-structure asks (CTA/hashtags/title).
- When triggered, `CHAIN_MODE=POST` is set and locked in session state.
- Lock constraints are enforced in Sentinel:
  - `CHAIN_MODE=POST`
  - `POST_MODE_LOCK`
  - `POST_STRUCTURE_ALLOWED`
  - `POST_CTA_REQUIRED`
  - `NO_RECRUITMENT`
  - `NO_META`
  - `OUTPUT_ONLY_POST`
  - `ALLOW_LORE_RETRIEVAL`
  - `ENTROPISM_GLOSSARY_ALLOWED`
  - `POST_STYLE_SEED=<seed>`

## 2) Degrade ladder
- Binding pipeline in POST:
  1. Full constraints pass (`full`)
  2. Style-only relax (`style_drop`)
  3. Shape simplification while preserving CTA (`shape_drop`)
- If degraded, lore-safe note can be appended:
  - EN: `Signal note: reduced noise.`
  - TR: `Not: Gurultuyu azalttim.`

## 3) Generic fallback kill-switch
- If final POST output matches generic explainer patterns, output is invalidated.
- System repairs with a deterministic concrete base and re-applies plan binding.
- Telemetry event: `post_generic_fallback_repair`.

## 4) Telemetry events
- File: `backend/chain_telemetry.log` (or `CHAIN_TELEMETRY_LOG` env).
- Enable/disable: `CHAIN_TELEMETRY_ENABLED=1|0`.
- Key events:
  - `post_mode_lock` (`mode_locked`, `post_style_seed`, `topic_hash`)
  - `mode_lock_violation`
  - `post_degrade` (`attempt_trace`, `length_mode`, `output_chars`)
  - `post_generic_fallback_repair`

## 5) Golden regression
- Script: `backend/scripts/golden_regression_chain.py`
- Run local:
```bash
python backend/scripts/golden_regression_chain.py
```
- Current checks:
  - `C_meta_leak`
  - `D_degrade_conflict`
  - `E_topic_shift_anchor`
  - `F_multiturn_cta_continuity`
  - `NP_strict_format`
  - `P_policy_conflict_post`

## 6) Score gates and alerts
- Score env:
  - `GOLDEN_MIN_SCORE` (global)
  - `GOLDEN_MIN_SCORE_<TEST_KEY>` (per test override)
- Alert thresholds:
  - `GOLDEN_ALERT_DEGRADE_RATE_MAX` (default `5`)
  - `GOLDEN_ALERT_MODE_LOCK_VIOLATIONS_MAX` (default `0`)
  - `GOLDEN_ALERT_INVALID_RETRY_MAX` (default `0`)
- Optional notifications:
  - Slack: `GOLDEN_SLACK_WEBHOOK_URL`
  - SMTP: `GOLDEN_SMTP_*`

## 7) CI
- Workflow: `.github/workflows/golden-regression.yml`
- Blocks merge on regression failure.

