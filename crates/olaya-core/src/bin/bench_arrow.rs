use arrow::record_batch::RecordBatch;
use olaya_core::{OlayaEngine, OntologyBatchBuilder};
use std::time::Instant;

/// Incident ontology seeded with a few named classes, padded with generated classes in a 10-ary
/// tree to `n` concepts. Generated labels share vocabulary ("generated", "incident", "type") so the
/// router's candidate lists stay realistic as the ontology grows.
fn build_ontology(n: usize) -> RecordBatch {
    let mut builder = OntologyBatchBuilder::new();
    builder.append_concept(
        0,
        "INCIDENT:ROOT",
        "Operational Incident",
        "Root incident class",
        &[],
        &[],
        &["incident"],
    );
    builder.append_concept(
        1,
        "INCIDENT:DATA",
        "Data Incident",
        "Storage failures",
        &[0],
        &[2],
        &["database", "db"],
    );
    builder.append_concept(
        2,
        "INCIDENT:BILLING",
        "Billing Incident",
        "Payment failures",
        &[0],
        &[1],
        &["billing", "payment"],
    );
    builder.append_concept(
        3,
        "INCIDENT:DB_DEADLOCK",
        "Database Deadlock",
        "Lock wait timeouts",
        &[1],
        &[],
        &["deadlock"],
    );
    builder.append_concept(
        4,
        "INCIDENT:AUTH_BREACH",
        "Security Breach",
        "Compromised tokens",
        &[0],
        &[],
        &["breach", "security"],
    );

    for i in 5..n {
        let parent = ((i - 5) / 10) as u32;
        let curie = format!("INCIDENT:GEN_{i}");
        let label = format!("Generated Incident Type {i}");
        builder.append_concept(
            i as u32,
            &curie,
            &label,
            "Auto generated class",
            &[parent],
            &[],
            &[],
        );
    }
    builder.finish().expect("Failed to build ontology")
}

fn main() {
    let sample_states = [
        "Fatal deadlock detected on PostgreSQL customer table while locking rows",
        "Payment gateway webhook failed with HTTP 500 when billing subscription",
        "Unauthorized access attempt detected with leaked API key token",
        "High latency on Redis cache cluster causing timeouts",
    ];
    let total_decisions = 10_000;
    let n_iterations = total_decisions / sample_states.len();

    println!("{}", "=".repeat(72));
    println!("O-Laya Rust + Arrow Native Benchmark ({total_decisions} decisions per size)");
    println!("{}", "=".repeat(72));
    println!(
        "{:>10} {:>12} {:>16} {:>18}",
        "concepts", "build (ms)", "avg latency (us)", "decisions / sec"
    );

    for n in [100usize, 10_000, 100_000] {
        let batch = build_ontology(n);
        let t_build = Instant::now();
        let engine = OlayaEngine::new(batch).expect("Failed to build OlayaEngine");
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;

        for state in &sample_states {
            engine.system_one(state, 8).unwrap();
        }

        let start = Instant::now();
        for _ in 0..n_iterations {
            for state in &sample_states {
                std::hint::black_box(engine.system_one(state, 8).unwrap());
            }
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:>10} {:>12.1} {:>16.2} {:>18.0}",
            n,
            build_ms,
            secs * 1e6 / total_decisions as f64,
            total_decisions as f64 / secs
        );
    }
    println!("{}", "=".repeat(72));

    let real_ontos = [
        ("Master Universe 62k", "models/master_universe.arrow", "Credit default swap hedging acute disease supply chain incident"),
        ("MONDO Disease", "models/mondo.arrow", "Severe neurodegenerative lysosomal storage disorder with sphingomyelinase deficiency"),
        ("FIBO Financial", "models/fibo.arrow", "Credit default swap derivative contract hedging corporate bond default risk"),
        ("NASA SWEET Earth", "models/sweet_earth.arrow", "Atmospheric pressure climate zone vortex fluid dynamics"),
        ("WCO HS Codes Trade", "models/hs_codes.arrow", "Pure bred breeding horses live bovine animals"),
        ("MITRE D3FEND Cyber", "models/d3fend.arrow", "Inbound traffic filtering packet inspection credential dump countermeasure"),
        ("GS1 Supply Chain", "models/gs1_full.arrow", "Refrigerated organic whole milk dairy beverage pasteurized cream"),
        ("Schema.org Web", "models/schemaorg.arrow", "High performance wireless noise cancelling bluetooth headphones with mic"),
        ("LKIF Core Legal", "models/lkif_legal.arrow", "Legal person contractual obligation statutory liability"),
        ("W3C Reference Core", "models/w3c_core.arrow", "Data provenance activity generating derivative dataset catalog record with historical agent attribution"),
    ];

    println!("\nReal Standard Ontologies Zero-Copy Arrow IPC Benchmark (10,000 decisions):");
    println!("{}", "=".repeat(72));
    println!(
        "{:<20} {:>10} {:>12} {:>16} {:>18}",
        "Ontology", "concepts", "file (KB)", "avg latency (us)", "decisions / sec"
    );

    for (name, path_str, test_query) in real_ontos {
        let p = std::path::Path::new(path_str);
        if !p.exists() {
            continue;
        }
        let file_kb = p.metadata().map(|m| m.len() / 1024).unwrap_or(0);
        let batch = olaya_core::read_ontology_ipc(p).expect("Failed to read IPC");
        let n_concepts = batch.num_rows();
        let engine = OlayaEngine::new(batch).expect("Failed to build engine");

        // Warmup
        engine.system_one(test_query, 8).unwrap();

        let start = Instant::now();
        for _ in 0..total_decisions {
            std::hint::black_box(engine.system_one(test_query, 8).unwrap());
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{:<20} {:>10} {:>12} {:>16.2} {:>18.0}",
            name,
            n_concepts,
            file_kb,
            secs * 1e6 / total_decisions as f64,
            total_decisions as f64 / secs
        );
    }
    println!("{}", "=".repeat(72));
}
