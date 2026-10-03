import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';

globalThis.crypto ??= webcrypto;
const source = await readFile(new URL('../docs/public/api/systemone.js', import.meta.url), 'utf8');
const { default: localHandler } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
// Set SYSTEMONE_TEST_URL to run these same cases against a deployed endpoint.
const handler = process.env.SYSTEMONE_TEST_URL
  ? async (request) => fetch(process.env.SYSTEMONE_TEST_URL, {
      method: request.method, headers: request.headers, body: await request.text(),
      signal: AbortSignal.timeout(25000),
    })
  : localHandler;
async function ask(state, question) {
  const response = await handler(new Request('http://localhost/api/systemone', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, questions: { result: question } }),
  }));
  assert.equal(response.status, 200);
  return (await response.json()).questions.result;
}

for (const state of ['Frozen seafood shipment.', 'Frozen seafood without paperwork.', 'No paperwork, but frozen seafood.', 'Not only frozen seafood but also ice.']) {
  test(`positive proposition: ${state}`, async () => {
    assert.ok((await ask(state, { type: 'noul', instructions: 'frozen seafood' })).noul > 0.9);
  });
}
for (const state of ['The shipment is not frozen seafood.', 'No frozen seafood.', "The shipment isn't frozen seafood.", 'Frozen seafood is absent.']) {
  test(`negated proposition: ${state}`, async () => {
    assert.ok((await ask(state, { type: 'noul', instructions: 'frozen seafood' })).noul < 0.1);
  });
}
test('contradictory clauses are uncertain', async () => {
  const result = await ask('Frozen seafood arrived, but no frozen seafood remains.', { type: 'noul', instructions: 'frozen seafood' });
  assert.equal(result.noul, 0.5);
  assert.equal(result.confidence, 0);
});
for (const [state, options, reason] of [
  ['Office stationery.', ['cold storage', 'ambient storage'], 'no_matching_evidence'],
  ['Cold and ambient storage.', ['cold storage', 'ambient storage'], 'tied_options'],
  ['Office stationery.', ['cold storage'], 'no_matching_evidence'],
  ['Cold storage.', [], 'no_options'],
]) {
  test(`choice abstention: ${reason}, ${options.length} options`, async () => {
    const result = await ask(state, { type: 'choice', options });
    assert.equal(result.choice, null);
    assert.equal(result.abstained, true);
    assert.equal(result.reason, reason);
    assert.equal(result.confidence, 0);
  });
}
test('supported unique choice preserves probabilities', async () => {
  const result = await ask('Cold storage shipment.', { type: 'choice', options: ['cold storage', 'ambient storage'] });
  assert.equal(result.choice, 'cold storage');
  assert.equal(result.abstained, false);
  assert.ok(Math.abs(Object.values(result.probabilities).reduce((a, b) => a + b, 0) - 1) < 0.001);
});
test('score response remains compatible', async () => {
  const result = await ask('Urgent shipment.', { type: 'score', criteria: ['routine', 'urgent', 'critical'] });
  assert.equal(result.level, 'urgent');
  assert.equal(result.score, 1);
});

for (const [state, instructions, status] of [
  ['Office stationery.', 'frozen seafood', 'unmatched'],
  ['Not chilled or frozen seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['The shipment does not contain frozen seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['There is no evidence of frozen seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood is not absent.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood is not missing.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood is not necessarily absent.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood may be absent.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen not seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['Not without frozen seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood not present.', 'frozen seafood', 'ambiguous_scope'],
  ['No ice and frozen seafood.', 'frozen seafood', 'ambiguous_scope'],
  ['Frozen seafood.', 'not frozen seafood', 'unsupported_question'],
  [{ frozen_seafood: false }, 'frozen seafood', 'unmatched'],
  [{ first: 'frozen', second: 'seafood' }, 'frozen seafood', 'unmatched'],
  [{ description: 'Frozen seafood.', paperwork: 'No paperwork.' }, 'frozen seafood', 'supported'],
  [{ description: 'Not frozen seafood.' }, 'frozen seafood', 'contradicted'],
  [{ first: 'Frozen seafood.', second: 'No frozen seafood.' }, 'frozen seafood', 'conflicting'],
]) {
  test(`noul evidence status: ${status}, ${JSON.stringify(state)}`, async () => {
    const result = await ask(state, { type: 'noul', instructions });
    assert.equal(result.evidence_status, status);
    const abstained = !['supported', 'contradicted'].includes(status);
    assert.equal(result.abstained, abstained);
    if (abstained) {
      assert.equal(result.noul, 0.5);
      assert.equal(result.confidence, 0);
    } else assert.ok(status === 'supported' ? result.noul > 0.9 : result.noul < 0.1);
  });
}

test('choice does not use JSON keys or boolean values as evidence', async () => {
  const result = await ask({ cold_storage: false }, { type: 'choice', options: ['cold storage', 'ambient storage'] });
  assert.equal(result.choice, null);
  assert.equal(result.reason, 'no_matching_evidence');
});
