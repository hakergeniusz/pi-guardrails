// Key-rotation test for googleDecisionsClassifier with a stubbed fetch and a
// redirected HOME (no real config or network touched).
//
// Run: node --experimental-strip-types test-keyring.ts
process.env.HOME = "/tmp/pi-guardrails-keyring-home";

let fetchCalls: string[] = [];
let script: (key: string) => { status: number; body: unknown } = () => ({ status: 500, body: {} });
// --- must run before importing index.ts (AGENT_DIR is computed at load) ---
const { googleDecisionsClassifier } = await import("./index.ts");
// --------------------------------------------------------------------------

globalThis.fetch = (async (_url: unknown, init?: { headers?: Record<string, string> }) => {
	const key = init?.headers?.["x-goog-api-key"] ?? "";
	fetchCalls.push(key);
	const r = script(key);
	return new Response(JSON.stringify(r.body), { status: r.status, headers: { "content-type": "application/json" } });
}) as typeof fetch;

const answersBody = {
	answers: {
		verdict: { type: "choice", choice: "allow", probabilities: { allow: 0.97, block: 0.03 } },
		category: { type: "choice", choice: "none" },
	},
};

const context = {
	state: { tool: "bash", action: "ls" },
	questions: {
		verdict: { type: "choice", instructions: "Guard verdict", criteria: { allow: "safe", block: "harmful" } },
		category: { type: "choice", instructions: "Category", criteria: { none: "none", destructive: "destructive" } },
	},
} as never;

const classify = googleDecisionsClassifier();
const model = { id: "gemini-decision-1" };
const K1 = "testkey-1-aaaa";
const K2 = "testkey-2-bbbb";
const K3 = "testkey-3-cccc";
const K4 = "testkey-4-dddd";

function assertEq(actual: unknown, expected: unknown, what: string) {
	if (actual !== expected) throw new Error(`${what}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}
function assert(cond: unknown, what: string) {
	if (!cond) throw new Error(what + ": failed");
}

// 0. No keys anywhere → explicit error, no fetch.
{
	const out = await classify(model, context, { apiKey: undefined });
	assertEq(out.stopReason, "error", "no-key case errors");
	assert((out.errorMessage ?? "").includes("no Google API key"), "no-key message");
	assertEq(fetchCalls.length, 0, "no fetch without keys");
}

process.env.GEMINI_API_KEY_1 = K1;
process.env.GEMINI_API_KEY_2 = K2;
process.env.GEMINI_API_KEY_3 = K3;
process.env.GEMINI_API_KEY_4 = K4;

// 1. First key hits a per-day quota; the next key answers. Rotation + sticky.
script = (key) =>
	key === K1
		? { status: 429, body: { error: { message: 'Quota exceeded for metric, quotaId "GenerateRequestsPerDayPerProjectPerModel", requests per day limit exceeded' } } }
		: { status: 200, body: answersBody };
let out = await classify(model, context, { apiKey: K1 });
assertEq(out.stopReason, "stop", "rotates past per-day 429");
assertEq(out.answers.verdict && (out.answers.verdict as { choice?: string }).choice, "allow", "verdict from second key");
assertEq(fetchCalls.join(","), `${K1},${K2}`, "tried key1 then key2");

// 2. Sticky: the key that answered goes first next time; no re-probe of key1.
out = await classify(model, context, {});
assertEq(fetchCalls.join(","), `${K1},${K2},${K2}`, "sticky key2 first, key1 not re-probed");

// 3. Per-minute hit with Google's retry hint → short cooldown, rotate on.
script = (key) =>
	key === K2
		? { status: 429, body: { error: { message: "Generate content input tokens per minute limit exceeded. Please retry in 12s." } } }
		: { status: 200, body: answersBody };
out = await classify(model, context, {});
assertEq(fetchCalls.join(","), `${K1},${K2},${K2},${K2},${K3}`, "per-minute hit rotates to key3");
assertEq(out.stopReason, "stop", "per-minute rotation still answers");

// 4. Invalid key (401) → out for the day; next key answers.
script = (key) =>
	key === K3
		? { status: 401, body: { error: { message: "API key not valid. Please pass a valid API key." } } }
		: { status: 200, body: answersBody };
out = await classify(model, context, {});
assertEq(fetchCalls.join(","), `${K1},${K2},${K2},${K2},${K3},${K3},${K4}`, "401 rotates to key4");
assertEq(out.stopReason, "stop", "401 rotation still answers");

// 5. Network error is NOT key-specific: nothing is marked, so the sticky key
// is attempted again on the next call (the others are still legitimately
// marked from tests 1/3/4, so the result is an error either way).
script = (key) => {
	if (key === K4) throw new TypeError("fetch failed");
	return { status: 200, body: answersBody };
};
out = await classify(model, context, {});
assertEq(out.stopReason, "error", "network failure with all other keys marked errors");
const callsAfterNet = fetchCalls.length;
out = await classify(model, context, {});
assertEq(fetchCalls.length, callsAfterNet + 1, "network failure left no mark (sticky key retried)");
assertEq(out.stopReason, "error", "still error while other keys are marked");

// 6. Flood: the remaining unmarked key gets a generic quota 429 (suspected,
// 30 min) → every key is marked; a further call makes zero fetch attempts.
script = () => ({ status: 429, body: { error: { message: "You exceeded your current quota, please check your plan and billing details." } } });
out = await classify(model, context, {});
assertEq(out.stopReason, "error", "flood errors out");
assert((out.errorMessage ?? "").includes("key(s)"), "flood message mentions keys");
const callsBefore = fetchCalls.length;
out = await classify(model, context, {});
assertEq(out.stopReason, "error", "all-marked errors");
assertEq(fetchCalls.length, callsBefore, "all-marked makes no fetch attempts");
assert((out.errorMessage ?? "").includes("quota-marked"), "all-marked message");

// 7. Abort before the call → aborted, no fetch.
script = () => ({ status: 200, body: answersBody });
const controller = new AbortController();
controller.abort();
out = await classify(model, context, { signal: controller.signal });
assertEq(out.stopReason, "aborted", "pre-aborted signal");

console.log("OK — key rotation: 429/401/cooldown marks, sticky key, network-failure wrap, flood, abort all pass");
