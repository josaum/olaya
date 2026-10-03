use anyhow::{Context, Result};
use olaya_core::{write_ontology_ipc, OntologyBatchBuilder};
use owl_fast_core::fca::{ConceptLattice, FormalContext};
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn load_pairs_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>> {
    let p = path.as_ref();
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut pairs = Vec::new();

    if ext == "json" {
        let content = fs::read_to_string(p)?;
        // Try object -> [attributes] map
        if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<String>>>(&content) {
            for (obj, attrs) in map {
                for attr in attrs {
                    pairs.push((obj.clone(), attr));
                }
            }
        } else if let Ok(list) = serde_json::from_str::<Vec<(String, String)>>(&content) {
            pairs = list;
        } else {
            anyhow::bail!(
                "Unrecognized JSON structure. Expected {{object: [attrs]}} or [[obj, attr]]"
            );
        }
    } else {
        // Assume CSV/TSV format: object,attribute
        let file = File::open(p)?;
        let reader = BufReader::new(file);
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // Skip header if looks like "object,attribute"
            if line_num == 0
                && (trimmed.to_lowercase().contains("object")
                    || trimmed.to_lowercase().contains("item"))
            {
                continue;
            }
            let delimiter = if trimmed.contains('\t') { '\t' } else { ',' };
            let parts: Vec<&str> = trimmed.split(delimiter).map(str::trim).collect();
            if parts.len() >= 2 {
                let obj = parts[0].to_string();
                for &attr in &parts[1..] {
                    if !attr.is_empty() {
                        pairs.push((obj.clone(), attr.to_string()));
                    }
                }
            }
        }
    }

    Ok(pairs)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Leio Autonomous Ontology Inducer (Fast FCA + Higher-Order Algebra)");
        eprintln!("Usage: induce_ontology <input.csv|tsv|json> <output.arrow> [--prefix <ns>] [--implications <out.json>]");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let prefix = args
        .windows(2)
        .find(|w| w[0] == "--prefix")
        .map(|w| w[1].as_str())
        .unwrap_or("indiced");
    let implications_path = args
        .windows(2)
        .find(|w| w[0] == "--implications")
        .map(|w| w[1].as_str());

    println!(
        "⚡ Loading incidence context from {}...",
        input_path.display()
    );
    let t0 = Instant::now();
    let raw_pairs = load_pairs_from_file(input_path)
        .with_context(|| format!("Failed to load pairs from {}", input_path.display()))?;

    let string_pairs: Vec<(&str, &str)> = raw_pairs
        .iter()
        .map(|(o, a)| (o.as_str(), a.as_str()))
        .collect();

    let context = FormalContext::from_named_pairs(&string_pairs);
    let t_ctx = t0.elapsed();
    println!(
        "  * Extracted formal context: {} objects, {} attributes ({} incidences) in {:.2}ms",
        context.num_objects(),
        context.num_attributes(),
        string_pairs.len(),
        t_ctx.as_secs_f64() * 1e3
    );

    // 2. Build Concept Lattice via Close-by-One
    println!("⚙️  Inducing concept lattice via Close-by-One (CbO)...");
    let t_lattice = Instant::now();
    let lattice = ConceptLattice::build(context);
    let dt_lattice = t_lattice.elapsed();
    println!(
        "  * Synthesized {} formal concepts with complete covering relation in {:.2}ms",
        lattice.concepts.len(),
        dt_lattice.as_secs_f64() * 1e3
    );

    // 3. Algebraic Properties: Distributivity & Heyting Algebra
    let is_heyting = lattice.is_distributive();
    println!(
        "  * Higher-order algebraic structure: {} (Heyting Algebra = {})",
        if is_heyting {
            "Complete Distributive Lattice"
        } else {
            "Complete Bounded Lattice"
        },
        if is_heyting {
            "✅ YES"
        } else {
            "⚠️ NON-DISTRIBUTIVE"
        }
    );

    // 4. Compute Canonical Implications
    let implications = lattice.compute_canonical_implications();
    println!(
        "  * Extracted {} sound canonical implications (100% confidence)",
        implications.len()
    );

    if let Some(imp_out) = implications_path {
        let mut imp_list = Vec::new();
        for imp in &implications {
            let premise_attrs: Vec<&str> = imp
                .premise
                .iter_ones()
                .map(|i| lattice.context.attributes[i].as_str())
                .collect();
            let conclusion_attrs: Vec<&str> = imp
                .conclusion
                .iter_ones()
                .map(|i| lattice.context.attributes[i].as_str())
                .collect();
            imp_list.push(serde_json::json!({
                "premise": premise_attrs,
                "conclusion": conclusion_attrs,
                "support": imp.support,
                "confidence": imp.confidence
            }));
        }
        let f = File::create(imp_out)?;
        serde_json::to_writer_pretty(f, &imp_list)?;
        println!("  -> Saved canonical implications to {}", imp_out);
    }

    // 5. Convert to Zero-Copy Arrow IPC RecordBatch
    println!("🚀 Compiling induced lattice to zero-copy Arrow IPC...");
    let entities = lattice.to_ontology_entities(prefix);

    let mut uri_to_id: HashMap<String, u32> = HashMap::new();
    for (idx, e) in entities.iter().enumerate() {
        uri_to_id.insert(e.uri.clone(), idx as u32);
    }

    let mut builder = OntologyBatchBuilder::new();
    for (idx, e) in entities.iter().enumerate() {
        let parent_ids: Vec<u32> = e
            .parents
            .iter()
            .filter_map(|p| uri_to_id.get(p).copied())
            .collect();
        let synonyms: Vec<&str> = e.synonyms.iter().map(String::as_str).collect();

        builder.append_concept(
            idx as u32,
            &e.uri,
            &e.label,
            e.description.as_deref().unwrap_or(""),
            &parent_ids,
            &[],
            &synonyms,
        );
    }

    let batch = builder
        .finish()
        .context("Failed to build Arrow RecordBatch")?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    write_ontology_ipc(&batch, output_path)
        .with_context(|| format!("Failed to write Arrow IPC to {}", output_path.display()))?;

    let file_size_kb = output_path
        .metadata()
        .map(|m| m.len() as f64 / 1024.0)
        .unwrap_or(0.0);
    println!(
        "✅ Induced ontology successfully exported to: {} ({:.1} KB)",
        output_path.display(),
        file_size_kb
    );

    Ok(())
}
