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
  `modelRegistry.classify()` with a choice question. Local llama.cpp decision
  models work through the `llama-cpp-classify` fallback (label-token
  logprobs, e.g. `llama.cpp/d3-lite` served by llama-server) and SystemOne
  services (`typesafe-system-one`, e.g. openrouter's `~liquid/d1-*` at
  `/api/v1/systemone`) answer natively.
- **chat models** (e.g. openrouter `:free`) go through
  `modelRegistry.complete()` with a strict JSON verdict protocol.

Default: `llama.cpp/d3-lite` (local Decision 3.0 Lite, ~100ms per verdict),
then `openrouter/nvidia/nemotron-3.5-lightning:free`, then
`openrouter/nvidia/nemotron-3.5-super-vl:free`.

## Configuration

`~/.pi/agent/guardrails.json` (all keys optional, defaults below):

```json
{
  "enabled": true,
  "model": ["llama.cpp/d3-lite", "openrouter/nvidia/nemotron-3.5-lightning:free", "openrouter/nvidia/nemotron-3.5-super-vl:free"],
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
