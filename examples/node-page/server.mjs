// Demo page backed by the leio-decision Node package.
//
// Set OLAYA_EXTRA_ONTOLOGY=/path/to/ontology.json to add a private ontology as
// a domain (it becomes the default). The file is the usual DAG JSON; an optional
// top-level `presets` array ({text} objects) supplies the example observations.
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { OLaya, ONTOLOGY_CATALOG, OntologyDAG } from "leio-decision";

const PORT = Number(process.env.PORT ?? 8787);
const ABSTAIN_BELOW = 0.25;

const catalog = { ...ONTOLOGY_CATALOG };
let defaultDomain = "incident";
if (process.env.OLAYA_EXTRA_ONTOLOGY) {
  const json = JSON.parse(readFileSync(process.env.OLAYA_EXTRA_ONTOLOGY, "utf8"));
  const key = "extra";
  catalog[key] = {
    title: json.name ?? basename(process.env.OLAYA_EXTRA_ONTOLOGY, ".json"),
    summary: json.description ?? "",
    load: () => OntologyDAG.fromJSON(json),
    presets: (json.presets ?? []).map((p) => (typeof p === "string" ? p : p.text)).filter(Boolean),
  };
  defaultDomain = key;
}

const engines = new Map();
const engineFor = (domain) => {
  if (!engines.has(domain)) engines.set(domain, new OLaya(catalog[domain].load()));
  return engines.get(domain);
};

const PRESETS = {
  incident: ["Payment webhook timeout on credit card charge", "Transactions blocked waiting on a row lock in Postgres", "asdf qwerty lorem"],
  gs1: ["Organic whole milk 1 litre carton", "Pallet shipment delayed at the freight warehouse"],
  schemaorg: ["Reserve a table for four at an Italian restaurant on Friday evening", "A mobile app for editing videos on Android", "Quarterly earnings report of a public company"],
  doid: ["Diffuse large B-cell lymphoma in a 62 year old patient"],
  bfo: ["The process of cell division over time"],
  skos: ["A preferred label for a concept in a thesaurus", "asdf qwerty lorem"],
};

const DOMAINS = Object.entries(catalog)
  .map(([key, v]) => ({ key, title: v.title, size: v.load().size(), presets: PRESETS[key] ?? (v.presets ?? []).slice(0, 20) }))
  .sort((a, b) => (a.key === defaultDomain ? -1 : b.key === defaultDomain ? 1 : 0));

function decide(domain, text) {
  const engine = engineFor(domain);
  const dag = engine.ontology;
  const t0 = process.hrtime.bigint();
  return engine.systemOne(text).then((res) => {
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    const sel = res.selectedConcept;
    const ancestorIds = [...dag.getAncestors(sel.id)].filter((id) => id !== sel.id);
    const chain = [];
    let cur = sel;
    while (cur && cur.parents.length) {
      cur = dag.getById(cur.parents[0]);
      if (!cur) break;
      chain.unshift({ curie: cur.curie, label: cur.label });
    }
    const ancestorSet = new Set(ancestorIds);
    const candidates = res.probabilities
      .filter((p) => {
        const c = dag.getByCurie(p.curie);
        return c && c.id !== sel.id && !ancestorSet.has(c.id);
      })
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5)
      .map((p) => {
        const c = dag.getByCurie(p.curie);
        return { curie: p.curie, label: p.label, prob: p.prob, disjoint: dag.isDisjoint(sel.id, c.id) };
      });
    const isRoot = sel.parents.length === 0;
    const abstain = res.confidence < ABSTAIN_BELOW || (isRoot && candidates.every((c) => c.prob === 0));
    return {
      selected: { curie: sel.curie, label: sel.label, description: sel.description ?? "" },
      confidence: res.confidence,
      axiomsVerified: res.axiomsVerified,
      chain,
      candidates,
      abstain,
      ms,
    };
  });
}

const page = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>leio-decision on Node</title>
<style>
:root{
  --bg:#000;--panel:#0a0a0c;--line:rgba(255,255,255,.08);--line-strong:rgba(255,255,255,.18);
  --text:#f5f5f7;--muted:#86868b;--faint:#6e6e73;--blue:#2997ff;--red:#ff453a;--green:#30d158;
  --sans:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 var(--sans)}
a{color:var(--blue);text-decoration:none}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 80px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:36px}
header h1{font-size:20px;font-weight:600;margin:0;letter-spacing:-.01em}
header h1 span{color:var(--muted);font-weight:400}
header p{margin:0;color:var(--muted);font-size:14px}
.grid{display:grid;grid-template-columns:380px 1fr;gap:48px}
@media (max-width:860px){.grid{grid-template-columns:1fr;gap:32px}}
fieldset{border:0;padding:0;margin:0 0 24px}
legend{font-size:13px;color:var(--muted);margin-bottom:10px;padding:0}
.onts{display:flex;flex-direction:column;gap:2px}
.onts label{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-radius:8px;cursor:pointer;color:var(--muted)}
.onts label:hover{background:rgba(255,255,255,.04)}
.onts input{position:absolute;opacity:0;pointer-events:none}
.onts label:has(input:checked){background:rgba(41,151,255,.12);color:var(--text)}
.onts label:has(input:focus-visible){outline:2px solid var(--blue);outline-offset:-2px}
.onts small{font:12px var(--mono);color:var(--faint);white-space:nowrap;margin-left:12px}
textarea{width:100%;min-height:104px;resize:vertical;padding:12px 14px;border-radius:10px;border:1px solid var(--line-strong);background:var(--panel);color:var(--text);font:inherit;line-height:1.45}
textarea:focus-visible{outline:2px solid var(--blue);outline-offset:1px;border-color:transparent}
.presets{margin:10px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:4px;max-height:340px;overflow:auto}
.presets button{all:unset;cursor:pointer;color:var(--muted);font-size:14px;padding:3px 0;border-radius:4px;line-height:1.35}
.presets button:hover{color:var(--text)}
.presets button:focus-visible{outline:2px solid var(--blue)}
.go{margin-top:20px;display:flex;align-items:center;gap:14px}
.go button{font:600 15px var(--sans);color:#fff;background:#0071e3;border:0;border-radius:10px;padding:11px 20px;cursor:pointer}
.go button:hover{background:#0077ed}
.go button:focus-visible{outline:2px solid #fff;outline-offset:2px}
.go span{color:var(--faint);font-size:13px}
.result{min-height:360px}
.empty{color:var(--faint);padding-top:8px}
.lineage{list-style:none;margin:0;padding:0;border-left:1px solid var(--line-strong);margin-left:7px}
.lineage li{position:relative;padding:4px 0 4px 24px;color:var(--muted)}
.lineage li::before{content:"";position:absolute;left:-4px;top:15px;width:7px;height:7px;border-radius:50%;background:var(--line-strong)}
.lineage li.sel{padding-top:10px;padding-bottom:0;color:var(--text)}
.lineage li.sel::before{background:var(--blue);width:9px;height:9px;left:-5px;top:22px}
.lineage li.sel strong{display:block;font-size:34px;line-height:1.15;font-weight:600;letter-spacing:-.02em}
.lineage code,.cands code{font:12px var(--mono);color:var(--faint)}
.lineage li.sel code{font-size:13px}
.desc{color:var(--muted);margin:10px 0 0;max-width:60ch}
.result.abstain .lineage li.sel strong{color:var(--muted);text-decoration:line-through;text-decoration-color:var(--red);text-decoration-thickness:2px}
.facts{display:flex;gap:28px;flex-wrap:wrap;margin:28px 0 0;padding:20px 0 0;border-top:1px solid var(--line)}
.facts div{min-width:120px}
.facts dt{font-size:13px;color:var(--muted)}
.facts dd{margin:2px 0 0;font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.facts dd.ok{color:var(--green)}.facts dd.bad{color:var(--red)}
.note{margin:14px 0 0;padding:12px 14px;border-radius:10px;background:rgba(255,69,58,.1);color:var(--text);font-size:14px;max-width:60ch}
.cands{margin:28px 0 0;padding:20px 0 0;border-top:1px solid var(--line)}
.cands h2{font-size:13px;font-weight:500;color:var(--muted);margin:0 0 10px}
.cands table{width:100%;border-collapse:collapse;max-width:560px}
.cands td{padding:6px 0;border-bottom:1px solid var(--line);vertical-align:baseline}
.cands td:last-child{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted);width:70px}
.cands tr.dis td:first-child{color:var(--faint);text-decoration:line-through;text-decoration-color:var(--red)}
.cands tr.dis td:first-child::after{content:" disjoint";font-size:12px;color:var(--red);text-decoration:none;display:inline-block;margin-left:6px}
.cands .bar{display:inline-block;height:4px;background:var(--blue);border-radius:2px;vertical-align:middle;margin-right:8px}
footer{margin-top:56px;color:var(--faint);font-size:13px;max-width:70ch}
footer code{font:12px var(--mono);color:var(--muted)}
@media (prefers-reduced-motion:no-preference){.result{transition:opacity .15s}.result.busy{opacity:.5}}
</style></head><body><div class="wrap">
<header>
  <h1>leio-decision <span>running in Node</span></h1>
  <p>Lexical routing over a bundled ontology. No neural weights.</p>
</header>
<div class="grid">
  <form id="f">
    <fieldset>
      <legend>Ontology</legend>
      <div class="onts" id="onts"></div>
    </fieldset>
    <fieldset>
      <legend>Observation</legend>
      <textarea id="text" name="text" spellcheck="false"></textarea>
      <ul class="presets" id="presets"></ul>
    </fieldset>
    <div class="go"><button type="submit">Decide</button><span id="rt"></span></div>
  </form>
  <section class="result" id="result" aria-live="polite"><p class="empty">Pick an observation and decide.</p></section>
</div>
<footer>Each decision calls <code>OLaya.systemOne</code> from the npm package. The lineage is read from the ontology DAG; a strikethrough marks a candidate the disjointness axiom rules out. A decision is shown as abstained when confidence is below ${ABSTAIN_BELOW} or nothing matched below the root. A private ontology can be added with <code>OLAYA_EXTRA_ONTOLOGY</code>.</footer>
</div>
<script>
const DOMAINS=${JSON.stringify(DOMAINS)};
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let domain=DOMAINS[0].key;
function renderOnts(){
  $("#onts").innerHTML=DOMAINS.map(d=>\`<label><input type="radio" name="domain" value="\${d.key}" \${d.key===domain?"checked":""}><span>\${esc(d.title)}</span><small>\${d.size} concepts</small></label>\`).join("");
  $("#presets").innerHTML=DOMAINS.find(d=>d.key===domain).presets.map(p=>\`<li><button type="button" data-p="\${esc(p)}">\${esc(p)}</button></li>\`).join("");
}
$("#onts").addEventListener("change",e=>{domain=e.target.value;renderOnts();const p=DOMAINS.find(d=>d.key===domain).presets[0];if(p){$("#text").value=p;run();}});
$("#presets").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;$("#text").value=b.dataset.p;run();});
$("#f").addEventListener("submit",e=>{e.preventDefault();run();});
async function run(){
  const text=$("#text").value.trim(); if(!text)return;
  const out=$("#result"); out.classList.add("busy");
  const t0=performance.now();
  const r=await fetch("/api/decide",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({text,domain})});
  const d=await r.json();
  $("#rt").textContent=\`\${d.ms.toFixed(1)} ms in the engine, \${Math.round(performance.now()-t0)} ms round trip\`;
  out.classList.toggle("abstain",d.abstain); out.classList.remove("busy");
  out.innerHTML=\`
    <ol class="lineage">
      \${d.chain.map(c=>\`<li>\${esc(c.label)} <code>\${esc(c.curie)}</code></li>\`).join("")}
      <li class="sel"><strong>\${esc(d.selected.label)}</strong><code>\${esc(d.selected.curie)}</code></li>
    </ol>
    \${d.selected.description?\`<p class="desc">\${esc(d.selected.description).slice(0,220)}</p>\`:""}
    \${d.abstain?\`<p class="note">No decision. \${d.confidence<${ABSTAIN_BELOW}?"Confidence is below the abstention threshold.":"Nothing in the text matched a class below the root."} Treat this as “don't know”, not as “\${esc(d.selected.label)}”.</p>\`:""}
    <dl class="facts">
      <div><dt>Confidence</dt><dd>\${d.abstain?"—":d.confidence.toFixed(2)}</dd></div>
      <div><dt>Axioms verified</dt><dd class="\${d.axiomsVerified?"ok":"bad"}">\${d.axiomsVerified?"yes":"no"}</dd></div>
      <div><dt>Depth</dt><dd>\${d.chain.length}</dd></div>
    </dl>
    \${d.candidates.length?\`<div class="cands"><h2>Other candidates the router considered</h2><table>\${d.candidates.map(c=>\`<tr class="\${c.disjoint?"dis":""}"><td>\${esc(c.label)} <code>\${esc(c.curie)}</code></td><td><span class="bar" style="width:\${Math.round(c.prob*40)}px"></span>\${c.prob.toFixed(2)}</td></tr>\`).join("")}</table></div>\`:""}\`;
}
renderOnts(); $("#text").value=DOMAINS[0].presets[0] ?? ""; if ($("#text").value) run();
</script></body></html>`;

createServer(async (req, res) => {
  if (req.method === "POST" && req.url === "/api/decide") {
    let body = "";
    for await (const chunk of req) body += chunk;
    let text = "", domain = defaultDomain;
    try { ({ text = "", domain = defaultDomain } = JSON.parse(body || "{}")); } catch { /* fall through */ }
    if (!Object.hasOwn(catalog, domain)) domain = defaultDomain;
    text = String(text).slice(0, 2000);
    if (!text.trim()) { res.writeHead(400, { "content-type": "application/json" }); return res.end('{"error":"empty text"}'); }
    try {
      const out = await decide(domain, text);
      res.writeHead(200, { "content-type": "application/json" });
      return res.end(JSON.stringify(out));
    } catch (err) {
      console.error(err);
      res.writeHead(500, { "content-type": "application/json" });
      return res.end('{"error":"decision failed"}');
    }
  }
  res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
  res.end(page);
}).listen(PORT, () => console.log(`leio-decision demo on http://localhost:${PORT}`));
