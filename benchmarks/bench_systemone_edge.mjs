// Run from any directory: node benchmarks/bench_systemone_edge.mjs
// Local Node microbenchmark; no network, Vercel runtime, or accuracy measurement.
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createHash, webcrypto } from 'node:crypto';
import { cpus, platform, arch, release } from 'node:os';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';

const repo = fileURLToPath(new URL('../', import.meta.url));
const script = fileURLToPath(import.meta.url);
const handlerPath = 'docs/public/api/systemone.js';
const baselineRef = '7989f31^';
const warmups = 200;
const samples = 2000;
const runs = 5;
const workload = {
  state: 'Urgent frozen seafood shipment requires cold storage.',
  questions: {
    storage: { type: 'choice', options: ['cold storage', 'ambient storage'] },
    priority: { type: 'score', criteria: ['routine', 'urgent', 'critical'] },
    frozen: { type: 'noul', instructions: 'frozen seafood' },
  },
};
const hash = (source) => createHash('sha256').update(source).digest('hex');
const baselineRevision = execFileSync('git', ['rev-parse', baselineRef], { cwd: repo, encoding: 'utf8' }).trim();
const originalSource = execFileSync('git', ['show', `${baselineRevision}:${handlerPath}`], { cwd: repo, encoding: 'utf8' });
const patchedSource = readFileSync(new URL(`../${handlerPath}`, import.meta.url), 'utf8');

if (process.argv[2] === '--worker') {
  globalThis.crypto ??= webcrypto;
  const run = Number(process.argv[3]);
  const load = async (source) => (await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).default;
  const handlers = { original: await load(originalSource), patched: await load(patchedSource) };
  const body = JSON.stringify(workload);
  async function invoke(name) {
    const response = await handlers[name](new Request('http://localhost/api/systemone', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
    }));
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.questions.storage.choice, 'cold storage');
    assert.equal(result.questions.priority.level, 'urgent');
    assert.ok(result.questions.frozen.noul > 0.9);
  }
  const order = (iteration) => (iteration + run) % 2 ? ['patched', 'original'] : ['original', 'patched'];
  for (let i = 0; i < warmups; i++) for (const name of order(i)) await invoke(name);
  const timings = { original: [], patched: [] };
  for (let i = 0; i < samples; i++) {
    for (const name of order(i)) {
      const start = performance.now();
      await invoke(name);
      timings[name].push(performance.now() - start);
    }
  }
  const percentile = (sorted, fraction) => sorted[Math.ceil(sorted.length * fraction) - 1];
  const summary = Object.fromEntries(Object.entries(timings).map(([name, values]) => {
    values.sort((a, b) => a - b);
    return [name, { median_ms: percentile(values, 0.5), p95_ms: percentile(values, 0.95) }];
  }));
  console.log(JSON.stringify({ run: run + 1, first_handler: order(0)[0], ...summary,
    patched_source_sha256: hash(patchedSource), original_source_sha256: hash(originalSource) }));
} else {
  const results = [];
  for (let run = 0; run < runs; run++) {
    results.push(JSON.parse(execFileSync(process.execPath, [script, '--worker', String(run)], {
      cwd: repo, encoding: 'utf8', maxBuffer: 1024 * 1024,
    })));
  }
  assert.ok(results.every((result) => result.patched_source_sha256 === hash(patchedSource)),
    'Working-tree handler changed during benchmark; rerun on stable source.');
  console.log(JSON.stringify({
    benchmark: 'systemone-local-handler', measured_at: new Date().toISOString(),
    node: process.version, platform: platform(), arch: arch(), os_release: release(),
    cpu: cpus()[0]?.model ?? 'unknown', logical_cpus: cpus().length,
    baseline_ref: baselineRef, baseline_revision: baselineRevision,
    original_source_sha256: hash(originalSource), patched_source_sha256: hash(patchedSource),
    warmups_per_handler_per_run: warmups, samples_per_handler_per_run: samples,
    independent_process_runs: runs, order: 'Alternate each sample; reverse initial order each run',
    percentile_method: 'Nearest rank', units: 'milliseconds', workload,
    timed_scope: 'Request construction, handler execution, response decoding, and identical result assertions; body serialization excluded',
    limitations: 'Single workload in local Node. Includes runtime noise and GC. No network, cold start, deployed latency, general accuracy, calibration, or statistical significance claim.',
    results,
  }, null, 2));
}
