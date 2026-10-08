# Concept-path inference latency per router / scorer

`OLaya.system_one`, one `concept` question, 60 timed decisions per cell after warm-up, fresh state text
each call (the router's query cache is bypassed). Embedding configurations use the routed candidate policy.
Machine: Linux-6.18.44-fc-v80-x86_64-with-glibc2.39, 4 CPUs, torch 2.14.1+cu130 on `cpu` with 4 threads.
Class texts are embedded once at load (`index`); nothing here is the Rust runtime, see `bench_arrow` for that.

| domain | concepts | configuration | weights | index (s) | median (ms) | p95 (ms) | decisions / s |
|---|---|---|---|---|---|---|---|
| gs1 | 63 | lexical | none (lexical router, untrained head) | - | 9.3 | 12.8 | 107 |
| gs1 | 63 | bge-small | fine-tuned | 0.7 | 32.0 | 100.1 | 31 |
| gs1 | 63 | embeddinggemma-2 | zero-shot | 26.4 | 182.1 | 580.1 | 5 |
| incident | 13 | lexical | none (lexical router, untrained head) | - | 7.7 | 10.5 | 130 |
| incident | 13 | bge-small | fine-tuned | 0.1 | 28.2 | 39.3 | 35 |
| incident | 13 | embeddinggemma-2 | zero-shot | 1.6 | 112.0 | 296.0 | 9 |
| schemaorg | 660 | lexical | none (lexical router, untrained head) | - | 15.8 | 20.2 | 63 |
| schemaorg | 660 | bge-small | fine-tuned | 10.3 | 35.9 | 144.0 | 28 |
| schemaorg | 660 | embeddinggemma-2 | zero-shot | 22.5 | 102.2 | 225.0 | 10 |

The embedding path adds one query encode per decision on top of routing and projection; the per-decision
cost does not grow with the number of classes beyond the matrix product. Latency is machine-dependent and is
not a published claim; run the script on the target hardware.
