use std::time::Instant;
use olaya_core::{OntologyBatchBuilder, OlayaEngine};

fn main() {
    println!("============================================================");
    println!("O-Laya Rust + Arrow Native Benchmark");
    println!("============================================================");

    let mut builder = OntologyBatchBuilder::new();

    // Generate 100 concepts across hierarchy
    builder.append_concept(0, "INCIDENT:ROOT", "Operational Incident", "Root incident class", &[], &[], &["incident"]);
    builder.append_concept(1, "INCIDENT:DATA", "Data Incident", "Storage failures", &[0], &[2], &["database", "db"]);
    builder.append_concept(2, "INCIDENT:BILLING", "Billing Incident", "Payment failures", &[0], &[1], &["billing", "payment"]);
    builder.append_concept(3, "INCIDENT:DB_DEADLOCK", "Database Deadlock", "Lock wait timeouts", &[1], &[], &["deadlock"]);
    builder.append_concept(4, "INCIDENT:AUTH_BREACH", "Security Breach", "Compromised tokens", &[0], &[], &["breach", "security"]);

    for i in 5..100 {
        let parent = (i % 4) as u32;
        let curie = format!("INCIDENT:GEN_{}", i);
        let label = format!("Generated Incident Type {}", i);
        builder.append_concept(i as u32, &curie, &label, "Auto generated class", &[parent], &[], &[]);
    }

    let batch = builder.finish().expect("Failed to build ontology");
    println!("Ontology Concepts in Arrow Batch: {}", batch.num_rows());

    let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine");

    let sample_states = vec![
        "Fatal deadlock detected on PostgreSQL customer table while locking rows",
        "Payment gateway webhook failed with HTTP 500 when billing subscription",
        "Unauthorized access attempt detected with leaked API key token",
        "High latency on Redis cache cluster causing timeouts",
    ];

    // Warmup
    for state in &sample_states {
        let _ = engine.system_one(state, 8).unwrap();
    }

    // Benchmark 10,000 decisions
    let n_iterations = 2500;
    let total_decisions = n_iterations * sample_states.len();

    println!("Executing {} total decisions across {} iterations...", total_decisions, n_iterations);
    let start = Instant::now();

    for _ in 0..n_iterations {
        for state in &sample_states {
            let _ = engine.system_one(state, 8).unwrap();
        }
    }

    let elapsed = start.elapsed();
    let total_secs = elapsed.as_secs_f64();
    let throughput = total_decisions as f64 / total_secs;
    let avg_latency_us = (total_secs * 1_000_000.0) / total_decisions as f64;

    println!("\n{}", "=".repeat(60));
    println!("Rust + Arrow Native Performance");
    println!("{}", "=".repeat(60));
    println!("Total Decisions:  {}", total_decisions);
    println!("Total Time:       {:.3} seconds", total_secs);
    println!("Avg Latency:      {:.2} microseconds ({:.4} ms)", avg_latency_us, avg_latency_us / 1000.0);
    println!("Throughput:       {:.0} decisions / second", throughput);
    println!("{}", "=".repeat(60));
}
