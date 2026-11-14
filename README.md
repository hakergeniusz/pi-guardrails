# pi-guardrails

Decision-model guardrails for [pi](https://pi.dev)'s `bash`, `write`, and `edit` tools.

Every guarded tool call is judged by a decision model before it runs. Allow is
silent; a block refuses the call and feeds the reason back to the agent so it
can pick a different approach.

## Design

Built after a previous guard extension was removed for over-sensitivity, with
four rules that keep it out of the way:

1. **High bar** — block only concrete, likely-irreversible harm: data
   destruction outside the project (or of unrecoverable uncommitted work),
   sending secrets to an external host, exfiltrating private data, malware.
   Never judges task alignment, style, or conventions.
2. **Fail-open** — when no decision model answers within the timeout, the
   action runs and the failure is logged. A dead guard never bricks the agent.
3. **neverBlock regexes** — user-defined safety valve; matching actions always
   pass, even when the model says block.
4. **Audit log** — every decision (verdicts, fast-path allows, fail-opens)
   appends to `~/.pi/agent/guardrails-log.jsonl`, so sensitivity is tuned
   against real data.

A read-only **fast path** (ls, cat, rg, git status, version checks, …) skips
the model entirely: less latency, and the model concentrates on calls that
could actually be dangerous. Compound commands with `cd x && …` prefixes are
matched after stripping the prefix.

## Decision models

Configured as pi model refs, tried in order; the first usable answer wins.

- **classifier models** (pi model type `classifier`) go through
  `modelRegistry.classify()` with a choice question. Three backends:
  - **Google AI Studio** (`google-decisions/gemini-decision-1`, default):
    this extension registers Google's Decisions API
    (`POST /v1beta/decisions`, map-shaped questions, `noul` bools,
    `x-goog-api-key` auth) as a classifier provider. **Multi-key ring**: the
    pi-resolved key first, then every google account in
    `multi-account.json`, then a `GEMINI_API_KEY_N` env scan. The key that
    last answered is sticky; on failure the next key is tried — a per-day
    quota 429 marks the key dead until midnight Pacific, a per-minute hit
    cools it by Google's retry hint (≤120s), an ambiguous quota 429 marks it
    "possibly RPD" for 30 minutes, and an invalid key (401/403) is out for
    the day. Marks are in-memory; success clears them. Generous free quota
    across 8 keys, ~200ms per verdict, calibrated confidences.
  - **llama.cpp** (`llama.cpp/d3-lite`): local decision models through the
    `llama-cpp-classify` fallback (label-token logprobs; d3-lite served by
    the PrismML fork on a dedicated llama-server, started on demand — see
    below).
  - **SystemOne** (`openrouter/~liquid/d1-*` at `/api/v1/systemone`):
    answers natively, needs credit.
- **chat models** (e.g. openrouter `:free`) go through
  `modelRegistry.complete()` with a strict JSON verdict protocol.

Default: `google-decisions/gemini-decision-1`, then
`openrouter/nvidia/nemotron-3.5-lightning:free`, then the lazy local
`llama.cpp/d3-lite`, then `openrouter/nvidia/nemotron-3.5-super-vl:free`.

## The lazy local fallback

`llama.cpp/d3-lite` runs as a systemd *user* unit (`d3-lite.service`) that is
**disabled**: no model sits in RAM while the cloud models are healthy. Only
when the chain reaches the local ref — every cloud model ahead of it has
failed — does the extension start the unit, wait for llama-server's
`/health` (up to 90 s), and call it with a 30 s timeout (a cold start needs
more headroom than the normal 8 s). It goes back down on its own: after
~10 minutes without a verdict, or when the pi session shuts down — but only
if that session started it; a unit running on purpose (user or another
session) is never touched. A failed start, or one that never turns healthy,
is dead-marked for 15 minutes while the chain continues fail-open.

## Configuration

`~/.pi/agent/guardrails.json` (all keys optional, defaults below):

```json
{
  "enabled": true,
  "model": ["google-decisions/gemini-decision-1", "openrouter/nvidia/nemotron-3.5-lightning:free", "llama.cpp/d3-lite", "openrouter/nvidia/nemotron-3.5-super-vl:free"],
  "scope": ["bash", "write", "edit"],
  "failMode": "open",
  "timeoutMs": 8000,
  "log": true,
  "maxChars": 6000,
  "cacheTtlMs": 300000,
  "minBlockConfidence": 0.6,
  "alwaysAllow": ["^\\s*(ls|pwd|cat|rg|grep|fd|jq)\\b", "^\\s*git (status|log|diff|show|branch)\\b"],
  "neverBlock": []
}
```

| key | meaning |
| --- | --- |
| `enabled` | guard on/off (also `/guardrails on|off`) |
| `model` | one ref or ordered list of refs |
| `scope` | which tools to guard (`bash`, `write`, `edit`, `powershell`) |
| `failMode` | `open` (run when models unavailable) or `closed` (block) |
| `timeoutMs` | per-model timeout |
| `log` | append decisions to the audit log |
| `maxChars` | truncate the payload sent to the model |
| `cacheTtlMs` | identical actions reuse their verdict for this long |
| `minBlockConfidence` | block verdicts below this confidence run anyway (allow) and are logged; 0.6 default, 0 disables |
| `alwaysAllow` | bash regexes that skip the model |
| `neverBlock` | regexes (matched against the bare action) that can never be blocked |

## Commands

- `/guardrails` — status + last 6 decisions
- `/guardrails on|off` — toggle (persists)
- `/guardrails model <provider/model-id>` — set the decision model (persists)
- `/guardrails test` — run a safe and a risky canned payload through the
  pipeline (nothing is executed)
- `/guardrails log` — last 15 decisions

## Install

```
pi install git:github.com/hakergeniusz/pi-guardrails
```

## License

MIT
