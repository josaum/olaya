"""Benchmark warm inference latency and throughput for O-Laya."""

import time
import statistics
from olaya.engine import OLaya

def run_benchmark():
    print("Initializing O-Laya engine...")
    olaya = OLaya()

    state = {
        "service": "billing-worker",
        "log_level": "FATAL",
        "error": "Deadlock detected while updating customer subscription balances in PostgreSQL cluster.",
    }

    questions = {
        "triage": {
            "type": "concept",
            "instructions": "Classify the operational incident into the appropriate ontology category.",
            "root": "INCIDENT:ROOT",
        },
        "is_data_issue": {
            "type": "subsumes",
            "concept": "INCIDENT:DATA",
            "instructions": "Is this a data or storage issue?",
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate the operational disruption level.",
            "criteria": ["low", "medium", "urgent", "critical"],
        },
        "escalate_oncall": {
            "type": "noul",
            "instructions": "Should on-call engineering be paged immediately?",
        },
    }

    # Warmup (3 passes)
    print("Warming up JIT and GPU/MPS kernels...")
    for _ in range(3):
        olaya.system_one(state, questions)

    # Benchmark runs
    n_runs = 25
    print(f"Executing {n_runs} benchmark runs...")
    latencies = []
    
    for i in range(n_runs):
        t0 = time.perf_counter()
        res = olaya.system_one(state, questions)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(0.95 * n_runs)]
    min_lat = min(latencies)

    print("\n" + "=" * 60)
    print("O-Laya Latency Benchmark Results")
    print("=" * 60)
    print(f"Device:           {olaya.device}")
    print(f"Questions/Batch:  {len(questions)} (concept, subsumes, score, noul)")
    print(f"Total Input Toks: {res.usage['input_tokens']}")
    print(f"Output Tokens:    {res.usage['output_tokens']} (Zero generation tax)")
    print("-" * 60)
    print(f"Min Latency:      {min_lat:.2f} ms")
    print(f"Median Latency:   {median_lat:.2f} ms")
    print(f"Mean Latency:     {mean_lat:.2f} ms")
    print(f"P95 Latency:      {p95_lat:.2f} ms")
    print("=" * 60)

    # Verify answers
    ans = res.answers["triage"]
    print(f"\nSample Decision Output:")
    print(f"Grounded Concept: {ans.curie} ({ans.label})")
    print(f"Lineage to Root:  {' -> '.join(ans.lineage)}")
    print(f"Subsumes P(true): {res.answers['is_data_issue'].subsumes}")
    print(f"Urgency Score:    {res.answers['urgency'].score}")
    print(f"Escalate (Noul):  {res.answers['escalate_oncall'].noul}")

if __name__ == "__main__":
    run_benchmark()
