// Lifecycle tests for d3LiteRuntime with fully injected dependencies — no
// systemd, no network, no real time. Run:
//
//   node --experimental-strip-types test-d3.ts

process.env.HOME = "/tmp/pi-guardrails-d3-home";

const { d3LiteRuntime } = await import("./index.ts");

function assertEq(actual: unknown, expected: unknown, what: string) {
	if (actual !== expected) throw new Error(`${what}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

type State = {
	healthy: boolean;
	unitActive: boolean;
	startCalls: number;
	stopCalls: number;
};

/** Fake clock: every sleep() advances it by one minute. */
function makeRuntime(state: State, opts?: { startFails?: boolean; neverHealthy?: boolean }) {
	let t = 1_000_000;
	return {
		rt: d3LiteRuntime({
			probe: async () => (opts?.neverHealthy ? false : state.healthy),
			isActive: async () => state.unitActive,
			start: async () => {
				state.startCalls++;
				if (opts?.startFails) throw new Error("systemctl: start failed");
				state.unitActive = true;
				state.healthy = true;
			},
			stop: async () => {
				state.stopCalls++;
				state.unitActive = false;
				state.healthy = false;
			},
			sleep: async () => {
				t += 60_000;
			},
			now: () => t,
		}),
		advance: (ms: number) => {
			t += ms;
		},
	};
}

// 1. Unit already up and healthy: use it, never start or stop someone
//    else's server — stopIfOurs is a no-op without ownership.
{
	const state = { healthy: true, unitActive: true, startCalls: 0, stopCalls: 0 };
	const { rt } = makeRuntime(state);
	assertEq(await rt.ensureUp(), true, "already-up ensureUp");
	assertEq(await rt.ensureUp(), true, "already-up ensureUp again");
	assertEq(state.startCalls, 0, "no start when healthy");
	await rt.stopIfOurs();
	assertEq(state.stopCalls, 0, "foreign unit untouched by stopIfOurs");
}

// 2. Unit down: start it, wait for health, own it — stopIfOurs stops it.
{
	const state = { healthy: false, unitActive: false, startCalls: 0, stopCalls: 0 };
	const { rt } = makeRuntime(state);
	assertEq(await rt.ensureUp(), true, "cold start ensureUp");
	assertEq(state.startCalls, 1, "one start");
	await rt.stopIfOurs();
	assertEq(state.stopCalls, 1, "owned unit stopped");
}

// 3. Start fails: marked dead — the next call does not retry systemctl.
{
	const state = { healthy: false, unitActive: false, startCalls: 0, stopCalls: 0 };
	const { rt } = makeRuntime(state, { startFails: true });
	assertEq(await rt.ensureUp(), false, "failing start");
	assertEq(await rt.ensureUp(), false, "dead-marked retry");
	assertEq(state.startCalls, 1, "only one start attempt");
}

// 4. Started but never healthy: cleaned up (no dead unit looping behind
//    Restart=always) and dead-marked until the TTL expires.
{
	const state = { healthy: false, unitActive: false, startCalls: 0, stopCalls: 0 };
	const { rt, advance } = makeRuntime(state, { neverHealthy: true });
	assertEq(await rt.ensureUp(), false, "never healthy");
	assertEq(state.stopCalls, 1, "dead unit stopped again");
	assertEq(await rt.ensureUp(), false, "still dead-marked");
	advance(16 * 60_000);
	assertEq(await rt.ensureUp(), false, "retries after TTL");
	assertEq(state.startCalls, 2, "start retried after TTL");
}

// 5. Unit active but unhealthy (loading weights, or someone else started
//    it): no start, no ownership, no stop — we just wait for health.
{
	const state = { healthy: false, unitActive: true, startCalls: 0, stopCalls: 0 };
	let polls = 0;
	const rt = d3LiteRuntime({
		probe: async () => (polls++ >= 2 ? (state.healthy = true) : false),
		isActive: async () => state.unitActive,
		start: async () => {
			state.startCalls++;
		},
		stop: async () => {
			state.stopCalls++;
		},
		sleep: async () => {},
		now: Date.now,
	});
	assertEq(await rt.ensureUp(), true, "foreign unit healthy");
	assertEq(state.startCalls, 0, "no start for an active unit");
	await rt.stopIfOurs();
	assertEq(state.stopCalls, 0, "foreign unit untouched");
}

// 6. Idle stop: the unit we started goes down after ~10 unused minutes, but
//    a verdict inside the window (noteUsed) keeps it up.
{
	const state = { healthy: false, unitActive: false, startCalls: 0, stopCalls: 0 };
	const { rt, advance } = makeRuntime(state);
	assertEq(await rt.ensureUp(), true, "ensureUp for idle case");
	advance(5 * 60_000);
	await rt.maybeIdleStop();
	assertEq(state.stopCalls, 0, "within idle window");
	rt.noteUsed();
	advance(9 * 60_000);
	await rt.maybeIdleStop();
	assertEq(state.stopCalls, 0, "noteUsed reset the window");
	advance(11 * 60_000);
	await rt.maybeIdleStop();
	assertEq(state.stopCalls, 1, "stopped after idle window");
}

// 7. noteUsed on a foreign unit is a no-op (ownership stays off).
{
	const state = { healthy: true, unitActive: true, startCalls: 0, stopCalls: 0 };
	const rt = d3LiteRuntime({
		probe: async () => state.healthy,
		isActive: async () => state.unitActive,
		start: async () => {
			state.startCalls++;
			state.healthy = true;
		},
		stop: async () => {
			state.stopCalls++;
		},
		sleep: async () => {},
		now: Date.now,
	});
	assertEq(await rt.ensureUp(), true, "foreign healthy");
	rt.noteUsed();
	await rt.stopIfOurs();
	assertEq(state.stopCalls, 0, "noteUsed cannot create ownership");
}

console.log("test-d3: all 7 cases pass");
