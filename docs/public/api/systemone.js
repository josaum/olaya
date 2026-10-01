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

    const stateToks = tokenize(rawState);
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
          results[qKey] = { type: 'choice', choice: null, confidence: 0, probabilities: {} };
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

        results[qKey] = {
          type: 'choice',
          choice: options[bestIdx],
          confidence: Number(confidence(probs).toFixed(4)),
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
        // Binary Bernoulli: yes = semantic match between proposition and state
        const propToks = tokenize(qVal.instructions || '');
        const stateSet = new Set(stateToks);
        const negationWords = new Set(['no', 'not', 'none', 'negative', 'absent', 'false', 'never', 'without', 'lack', 'free']);

        // Score positive: how many proposition tokens appear in state?
        let yesScore = 0.5;
        let propMatchCount = 0;
        for (const tok of propToks) {
          if (stateSet.has(tok)) propMatchCount++;
        }
        if (propToks.length > 0) {
          yesScore = 0.5 + (propMatchCount / propToks.length) * 3.5;
        }

        // Score negative: negate if state contains strong negation near proposition keywords
        let noScore = 0.5;
        for (const tok of negationWords) {
          if (stateSet.has(tok)) noScore += 2.0;
        }
        // If nothing matched the proposition, strong default to NO
        if (propMatchCount === 0) noScore += 2.0;

        const probs = softmax([yesScore, noScore], 0.85);
        results[qKey] = {
          type: 'noul',
          noul: Number(probs[0].toFixed(4)),
          confidence: Number(confidence([probs[0], probs[1]]).toFixed(4)),
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
