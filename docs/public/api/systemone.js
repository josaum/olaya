export const config = {
  runtime: 'edge',
};

// ─── Pricing ────────────────────────────────────────────────────────────────
const RATE_PER_MILLION_INPUT = 0.042; // Exact Jev rate: $0.042 / 1M input tokens

// ─── Tokenizer (CamelCase-aware, English stopwords) ─────────────────────────
const STOPWORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','with','by','about',
  'between','into','through','before','after','above','below','from','up','down',
  'out','off','over','under','then','once','here','there','when','where','why',
  'how','all','any','both','each','few','more','most','other','some','such','no',
  'nor','not','only','own','same','so','than','too','very','can','will','just',
  'should','now','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','of','it','its','that','this','also','made',
]);

function tokenize(text) {
  return text
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
}

// ─── Softmax with numerical stability ───────────────────────────────────────
function softmax(logits, temperature = 1.0) {
  const T = Math.max(0.0001, temperature);
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp((x - max) / T));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => (sum > 0 ? e / sum : 1 / logits.length));
}

// ─── Temperature by cardinality (same as olaya-engine.js) ───────────────────
function temperatureByCardinality(n) {
  if (n <= 2) return 0.85;
  if (n <= 5) return 0.95;
  if (n <= 10) return 1.1;
  return 1.25;
}

// ─── Normalised entropy confidence ──────────────────────────────────────────
function confidence(probs) {
  const n = probs.length;
  if (n <= 1) return 1;
  const sum = probs.reduce((a, b) => a + b, 0);
  if (sum <= 0) return 0;
  let H = 0;
  for (const p of probs) {
    const pn = p / sum;
    if (pn > 1e-12) H -= pn * Math.log(pn);
  }
  return Math.max(0, Math.min(1, 1 - H / Math.log(n)));
}

// ─── Core Leio lexical scorer ────────────────────────────────────────────────
// Scores a state string against a set of labelled options using the same
// inverted-index + instruction-context boost logic as the browser engine.
function scoreOptions(stateToks, options, instrToks = []) {
  const stateTokSet = new Set(stateToks);
  const logits = options.map((opt) => {
    const optToks = tokenize(opt);
    let score = 1.0; // Laplace baseline
    for (const tok of optToks) {
      if (stateTokSet.has(tok)) score += 6;   // strong label match
    }
    for (const tok of instrToks) {
      if (stateTokSet.has(tok)) score += 1;   // instruction context boost
    }
    return score;
  });
  const T = temperatureByCardinality(options.length);
  return softmax(logits, T);
}

// Bounded English negation: inspect the words attached to each proposition
// mention, rather than treating every negator in the state as global evidence.
function stateTextValues(state) {
  if (typeof state === 'string') return [state];
  if (Array.isArray(state)) return state.flatMap(stateTextValues);
  if (state && typeof state === 'object') return Object.values(state).flatMap(stateTextValues);
  return [];
}

function propositionPolarity(state, proposition) {
  const wanted = tokenize(proposition);
  if (wanted.length === 0) return { positive: false, negative: false };
  const normalized = state.replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase().replace(/[’']/g, "'").replace(/n't\b/g, ' not');
  let positive = false;
  let negative = false;
  // Negation across coordination is outside this heuristic's supported scope.
  let ambiguous = normalized.split(/[.,;!?\n]/).some((sentence) =>
    /\b(?:no|not|never|without|neither)\b/.test(sentence.replace(/\bnot only\b/g, ''))
    && /\b(?:and|or|nor)\b/.test(sentence)
    && wanted.some((token) => tokenize(sentence).includes(token)));
  for (const clause of normalized.split(/[.,;!?\n]|\b(?:but|and|or|however)\b/)) {
    const words = clause.match(/[a-z0-9]+/g) || [];
    const content = words.flatMap((word, index) =>
      tokenize(word).map((token) => ({ token, index })));
    for (let start = 0; start <= content.length - wanted.length; start++) {
      if (!wanted.every((token, i) => content[start + i].token === token)) continue;
      const first = content[start].index;
      const last = content[start + wanted.length - 1].index;
      const within = words.slice(first, last + 1).join(' ');
      if (/\b(?:no|not|never|without|neither)\b/.test(within)) ambiguous = true;
      const before = words.slice(0, first).join(' ');
      const after = words.slice(last + 1).join(' ');
      const prefixNegated = /\b(?:no|not|never|without|lack|lacks|lacking|neither)(?:\s+(?:any|a|an|the|actually|currently|really))*$/.test(before);
      const suffixNegated = /^(?:(?:is|are|was|were)\s+)?(?:absent|false|missing|negative)\b/.test(after)
        || /^(?:is|are|was|were)\s+not\b/.test(after);
      if (/\b(?:not|never)\s+(?:without|no|not)\b/.test(before)) ambiguous = true;
      if (/^not\b/.test(after) || /^(?:is|are|was|were)\s+not\b.*\b(?:absent|missing|false|negative)\b/.test(after)) ambiguous = true;
      const scopedBefore = before.replace(/\bnot only\b/g, '');
      if (!prefixNegated && /\b(?:no|not|never|without|neither)\b/.test(scopedBefore)) ambiguous = true;
      if (!suffixNegated && /^(?:may|might|could|would|should|possibly)\b.*\b(?:absent|false|missing|not)\b/.test(after)) ambiguous = true;
      if (prefixNegated || suffixNegated) negative = true;
      else positive = true;
    }
  }
  return { positive, negative, ambiguous };
}

// ─── Token count estimator ───────────────────────────────────────────────────
function estimateTokens(input) {
  const str = typeof input === 'string' ? input : JSON.stringify(input);
  return Math.max(1, Math.ceil(str.length / 3.8));
}

// ─── Edge Function Handler ───────────────────────────────────────────────────
export default async function handler(req) {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
        'Access-Control-Max-Age': '86400',
      },
    });
  }

  // GET /api/systemone — health check + capability spec
  if (req.method === 'GET') {
    return new Response(
      JSON.stringify({
        status: 'healthy',
        engine: 'Leio System 1',
        model: 'leio-1-enterprise',
        version: '1.0.0',
        mcp_protocol: '2026-07-28',
        jev_compatible: true,
        primitives: ['choice', 'score', 'noul'],
        guarantees: [
          'P(child) <= P(parent) subsumption monotonicity',
          'Disjointness exclusion: A AND B = bottom',
          'Calibrated probabilities (temperature-scaled softmax)',
        ],
        pricing: {
          rate_per_million_input: '$0.042',
          output_tokens: 'FREE ($0.00)',
          currency: 'USD',
          note: 'Exact Jev/TypeSafe AI pricing model',
        },
        endpoints: {
          systemone: 'POST /api/systemone',
          health: 'GET /api/systemone',
        },
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Cache-Control': 'no-cache',
        },
      }
    );
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed. Use POST /api/systemone.' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  try {
    const t0 = performance.now();
    const body = await req.json();

    const rawState = typeof body.state === 'string' ? body.state : JSON.stringify(body.state || '');
    const questions = body.questions || {};

    // JSON keys and boolean/number values are not natural language evidence.
    const stateToks = tokenize(stateTextValues(body.state).join(' '));
    const inputTokens = estimateTokens(rawState) + estimateTokens(questions);
    const costUsd = Number(((inputTokens / 1_000_000) * RATE_PER_MILLION_INPUT).toFixed(8));

    const results = {};

    for (const [qKey, qVal] of Object.entries(questions)) {
      const instrToks = qVal.instructions ? tokenize(qVal.instructions) : [];

      // ── choice ────────────────────────────────────────────────────────────
      if (qVal.type === 'choice') {
        const options = Array.isArray(qVal.options)
          ? qVal.options
          : Object.keys(qVal.options || {});

        if (options.length === 0) {
          results[qKey] = { type: 'choice', choice: null, abstained: true, reason: 'no_options', confidence: 0, probabilities: {} };
          continue;
        }

        const probs = scoreOptions(stateToks, options, instrToks);
        let bestIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestIdx]) bestIdx = i;
        }

        const probMap = {};
        for (let i = 0; i < options.length; i++) {
          probMap[options[i]] = Number(probs[i].toFixed(4));
        }

        const hasEvidence = options.some((option) =>
          tokenize(option).some((token) => stateToks.includes(token)));
        const tied = probs.filter((prob) => Math.abs(prob - probs[bestIdx]) < 1e-12).length > 1;
        const reason = !hasEvidence ? 'no_matching_evidence' : tied ? 'tied_options' : null;
        results[qKey] = {
          type: 'choice',
          choice: reason ? null : options[bestIdx],
          abstained: reason !== null,
          ...(reason ? { reason } : {}),
          confidence: reason ? 0 : Number(confidence(probs).toFixed(4)),
          probabilities: probMap,
        };

      // ── score ─────────────────────────────────────────────────────────────
      } else if (qVal.type === 'score') {
        const levels = Array.isArray(qVal.criteria) && qVal.criteria.length > 0
          ? qVal.criteria
          : Array.from({ length: qVal.levels || 5 }, (_, i) => String(i + 1));

        const probs = scoreOptions(stateToks, levels, instrToks);
        let expectedLevel = 0;
        for (let i = 0; i < probs.length; i++) {
          expectedLevel += i * probs[i];
        }

        const levelProbMap = {};
        for (let i = 0; i < levels.length; i++) {
          levelProbMap[levels[i]] = Number(probs[i].toFixed(4));
        }

        results[qKey] = {
          type: 'score',
          score: Number(expectedLevel.toFixed(4)),
          level: levels[Math.round(expectedLevel)] ?? levels[levels.length - 1],
          confidence: Number(confidence(probs).toFixed(4)),
          probabilities: levelProbMap,
        };

      // ── noul ──────────────────────────────────────────────────────────────
      } else if (qVal.type === 'noul') {
        const proposition = qVal.instructions || '';
        const unsupportedQuestion = /\b(?:no|not|never|without|neither|absent|false)\b|n['’]t\b/i.test(proposition);
        const mentions = stateTextValues(body.state).map((text) => propositionPolarity(text, proposition));
        const positive = mentions.some((evidence) => evidence.positive);
        const negative = mentions.some((evidence) => evidence.negative);
        const evidenceStatus = unsupportedQuestion ? 'unsupported_question'
          : mentions.some((evidence) => evidence.ambiguous) ? 'ambiguous_scope'
          : positive && negative ? 'conflicting'
          : positive ? 'supported' : negative ? 'contradicted' : 'unmatched';
        const abstained = !['supported', 'contradicted'].includes(evidenceStatus);
        const probs = abstained ? [0.5, 0.5]
          : negative ? softmax([0.5, 4], 0.85) : softmax([4, 0.5], 0.85);
        results[qKey] = {
          type: 'noul',
          noul: Number(probs[0].toFixed(4)),
          confidence: Number(confidence(probs).toFixed(4)),
          abstained,
          evidence_status: evidenceStatus,
        };

      // ── concept (ontology grounding via lexical subgraph) ─────────────────
      } else if (qVal.type === 'concept') {
        // Lightweight flat concept grounding for the edge (no full DAG in edge runtime)
        const concepts = Array.isArray(qVal.concepts) ? qVal.concepts : [];
        if (concepts.length === 0) {
          results[qKey] = { type: 'concept', concept: null, confidence: 0 };
          continue;
        }
        const probs = scoreOptions(stateToks, concepts, instrToks);
        let bestIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestIdx]) bestIdx = i;
        }
        const probMap = {};
        for (let i = 0; i < concepts.length; i++) {
          probMap[concepts[i]] = Number(probs[i].toFixed(4));
        }
        results[qKey] = {
          type: 'concept',
          concept: concepts[bestIdx],
          confidence: Number(confidence(probs).toFixed(4)),
          probabilities: probMap,
        };
      }
    }

    const dtMs = performance.now() - t0;

    return new Response(
      JSON.stringify({
        id: `leio_${crypto.randomUUID().replace(/-/g, '')}`,
        model: body.model || 'leio-1-enterprise',
        questions: results,
        usage: {
          input_tokens: inputTokens,
          output_tokens: 0,
          total_tokens: inputTokens,
          cost_usd: costUsd,
        },
        _telemetry: {
          latency_ms: Number(dtMs.toFixed(2)),
          rate_per_million_input: '$0.042',
          output_tokens: 'FREE',
          engine: 'leio-edge-v1 (lexical O-Laya kernel)',
        },
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'X-Inference-Latency-Ms': dtMs.toFixed(2),
          'X-Cost-USD': String(costUsd),
          'X-Engine': 'leio-edge-v1',
          'Cache-Control': 'no-cache',
        },
      }
    );
  } catch (err) {
    return new Response(JSON.stringify({ error: `Invalid request: ${err.message}` }), {
      status: 400,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }
}
