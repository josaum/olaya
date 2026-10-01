export const config = {
  runtime: 'edge',
};

// Rate: $0.042 per 1,000,000 input tokens. Free outputs.
const RATE_PER_MILLION_INPUT = 0.042;

function estimateTokens(input) {
  const str = typeof input === 'string' ? input : JSON.stringify(input);
  return Math.max(1, Math.ceil(str.length / 3.8));
}

function tokenize(text) {
  const splitCamel = text.replace(/([a-z])([A-Z])/g, '$1 $2');
  const clean = splitCamel.toLowerCase().replace(/[^a-z0-9]+/g, ' ');
  return clean.split(/\s+/).filter((t) => t.length > 2);
}

export default async function handler(req) {
  // CORS
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  }

  const url = new URL(req.url);

  // GET /api/systemone (Health & Info)
  if (req.method === 'GET') {
    return new Response(
      JSON.stringify({
        status: 'healthy',
        engine: 'Leio System 1',
        model: 'leio-1-enterprise',
        mcp_protocol: '2026-07-28',
        pricing: {
          rate_per_million_input: '$0.042',
          output_tokens: 'FREE ($0.00)',
          currency: 'USD',
        },
        endpoints: {
          systemone: 'POST /api/systemone',
          usage: 'GET /api/systemone?usage=true',
        },
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
      }
    );
  }

  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed. Use POST.' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
    });
  }

  try {
    const t0 = performance.now();
    const body = await req.json();
    const rawState = typeof body.state === 'string' ? body.state : JSON.stringify(body.state || '');
    const questions = body.questions || {};
    const stateTokens = new Set(tokenize(rawState));

    const inputTokens = estimateTokens(rawState) + estimateTokens(questions);
    const costUsd = Number(((inputTokens / 1_000_000) * RATE_PER_MILLION_INPUT).toFixed(8));

    const results = {};

    for (const [qKey, qVal] of Object.entries(questions)) {
      if (qVal.type === 'choice') {
        const options = qVal.options || [];
        const instrTokens = qVal.instructions ? tokenize(qVal.instructions) : [];
        let bestOption = options[0] || 'NONE';
        let bestScore = -1;
        const probs = {};

        for (const opt of options) {
          const optTokens = tokenize(opt);
          let matchCount = 0;
          for (const tok of optTokens) {
            if (stateTokens.has(tok)) matchCount += 3;
            if (instrTokens.includes(tok)) matchCount += 1;
          }
          const raw = 1.0 + matchCount;
          probs[opt] = raw;
          if (raw > bestScore) {
            bestScore = raw;
            bestOption = opt;
          }
        }

        // Softmax
        const expSum = Object.values(probs).reduce((sum, v) => sum + Math.exp(v), 0);
        for (const k of Object.keys(probs)) {
          probs[k] = Number((Math.exp(probs[k]) / expSum).toFixed(4));
        }

        results[qKey] = {
          type: 'choice',
          choice: bestOption,
          confidence: Number((probs[bestOption] || 0.9).toFixed(4)),
          probabilities: probs,
        };
      } else if (qVal.type === 'score') {
        const levels = qVal.levels || 5;
        const crit = qVal.instructions || '';
        const critToks = tokenize(crit);
        let match = 0;
        for (const t of critToks) {
          if (stateTokens.has(t)) match++;
        }
        const ratio = critToks.length > 0 ? match / critToks.length : 0.5;
        const level = Math.min(levels, Math.max(1, Math.ceil(ratio * levels)));

        const levelProbs = {};
        for (let l = 1; l <= levels; l++) {
          const dist = Math.abs(l - level);
          levelProbs[String(l)] = Number(Math.exp(-dist * 1.5).toFixed(4));
        }
        const sum = Object.values(levelProbs).reduce((a, b) => a + b, 0);
        for (const k of Object.keys(levelProbs)) {
          levelProbs[k] = Number((levelProbs[k] / sum).toFixed(4));
        }

        results[qKey] = {
          type: 'score',
          score: Number(ratio.toFixed(4)),
          level,
          confidence: Number((levelProbs[String(level)] || 0.85).toFixed(4)),
          probabilities: levelProbs,
        };
      } else if (qVal.type === 'noul') {
        const instr = qVal.instructions || '';
        const instrToks = tokenize(instr);
        let match = 0;
        for (const t of instrToks) {
          if (stateTokens.has(t)) match++;
        }
        const prob = instrToks.length > 0 ? match / instrToks.length : 0.05;
        results[qKey] = {
          type: 'noul',
          noul: Number(prob.toFixed(4)),
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
        },
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'X-Inference-Latency-Ms': dtMs.toFixed(2),
          'X-Cost-USD': String(costUsd),
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
