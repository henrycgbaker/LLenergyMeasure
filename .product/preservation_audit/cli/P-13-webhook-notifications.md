# P-13: Webhook Notifications

**Module**: `src/llenergymeasure/notifications/webhook.py`
**Risk Level**: MEDIUM
**Decision**: Pending — keep or cut? The feature is a complete implementation but has zero planning coverage. No peer tool in the research has this pattern.
**Planning Gap**: Not mentioned in any planning document. The `decisions/cli-ux.md` "What Was Cut" table does not include webhooks, meaning it was not explicitly cut — but it was also not confirmed as a keep.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/notifications/webhook.py`
**Key classes/functions**:
- `send_webhook_notification()` (line 19) — module-level function; signature: `(event_type: Literal["complete", "failure"], experiment_id: str, campaign_id: str | None = None, payload: dict[str, Any] | None = None) -> bool`

The function:
1. Loads user config via `load_user_config()` to check `config.notifications.webhook_url`
2. Returns `False` immediately if `webhook_url` is not set
3. Checks per-event-type toggles: `notifications.on_complete` and `notifications.on_failure`
4. Builds payload: `{"event_type": event_type, "experiment_id": experiment_id, "timestamp": UTC_ISO, ...}`; optionally includes `campaign_id` and `data`
5. POSTs via `httpx` with `timeout=10.0`, `follow_redirects=True`
6. On `TimeoutException`, `HTTPStatusError`, or `RequestError`: logs a warning and returns `False` — never raises

The function is entirely non-blocking and silent-failure: errors are logged at WARNING level only. Callers receive a `bool` indicating success but are not expected to act on it.

Config source: `~/.config/llenergymeasure/config.yaml` via `load_user_config()`. The notifications config block (`notifications.webhook_url`, `notifications.on_complete`, `notifications.on_failure`) is part of the Layer 1 user config.

## Why It Matters

For overnight study runs (which may take 4–12 hours), researchers need a mechanism to know when their job completes or fails without monitoring a terminal. Webhooks are the standard mechanism for this: they integrate trivially with Slack (via incoming webhook), Discord, ntfy.sh, custom CI systems, and most monitoring platforms. The implementation is complete, correct, and uses `httpx` which is already a declared dependency. The risk of removing this: researchers lose the only async notification mechanism and must poll or monitor externally.

The counter-argument for cutting: no peer tool in the CLI research (lm-eval-harness, Optimum-Benchmark, Zeus, MLflow CLI) provides webhook notifications. This is an "extra" feature that adds a dependency surface.

## Planning Gap Details

- `decisions/cli-ux.md` — not listed in "What Was Cut and Why"
- `decisions/architecture.md` — not mentioned
- `designs/architecture.md` — `notifications/` module not in the v2.0 module layout; would need a `notifications/` subpackage or module added if kept
- MEMORY.md / session decisions — not mentioned

The feature was not explicitly cut in Phase 4.5 planning. Its fate is genuinely undecided.

## Recommendation for Phase 5

Make an explicit decision. The two options:

**Option A — Keep**: Add `notifications/webhook.py` to the v2.0 module layout in `designs/architecture.md`. Ensure `UserConfig.notifications` schema (`webhook_url`, `on_complete`, `on_failure`) is included in `designs/user-config.md`. The implementation requires no changes. Dependency: `httpx` (already in project dependencies — verify in `pyproject.toml`).

**Option B — Cut**: Remove `notifications/` entirely. Add to `decisions/cli-ux.md` "What Was Cut" with rationale: "No peer tool has this pattern; researchers should use cron + external monitoring." If cut, warn in any migration guide that `notifications.*` in user config will be silently ignored in v2.0.

If kept, a usage note for `llem config` output would be appropriate: show "Notifications: webhook configured" or "Notifications: not configured" to surface discoverability.
