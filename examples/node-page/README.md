# leio-decision on Node: demo page

A single-file Node server that serves a page and answers each decision with the
`leio-decision` package: `OLaya.systemOne` over one of the bundled ontologies,
the ancestor chain read from the DAG, the candidates the router considered, and
an explicit abstention when nothing matched below the root.

Set `OLAYA_EXTRA_ONTOLOGY=/path/to/ontology.json` to add a private ontology as
a domain; it becomes the default, and an optional top-level `presets` array in
the file (`{ "text": ... }` objects) supplies the one-click examples.

```bash
cd packages/olaya-node && bun install && bun run build   # once, builds dist/
cd ../../examples/node-page && npm install && npm start
```

Then open <http://localhost:8787>. Lexical routing, no neural weights.
