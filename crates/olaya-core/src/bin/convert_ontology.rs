use anyhow::{Context, Result};
use olaya_core::{extract_curie, write_ontology_ipc, OntologyBatchBuilder};
use owl_fast_core::{Entity, EntityType, OntologyParser};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
struct JsonConcept {
    curie: String,
    label: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parents: Vec<String>,
    #[serde(default)]
    children: Vec<String>,
    #[serde(default)]
    disjoint_with: Vec<String>,
    #[serde(default)]
    synonyms: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonOntology {
    name: String,
    concepts: Vec<JsonConcept>,
}

fn process_ontology<P: AsRef<Path>>(
    input_path: P,
    name: Option<&str>,
    full_mode: bool,
) -> Result<JsonOntology> {
    let path = input_path.as_ref();
    let mut files = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let p = entry?.path();
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if ext == "ttl" || ext == "owl" || ext == "rdf" {
                    files.push(p);
                }
            }
        }
    } else {
        files.push(path.to_path_buf());
    }

    let mut entities: Vec<Entity> = Vec::new();
    for f in &files {
        println!("Parsing ontology from: {}", f.display());
        let variant = if full_mode {
            owl_fast_core::ConferenceEvalVariant::Full
        } else {
            owl_fast_core::ConferenceEvalVariant::M1
        };
        match OntologyParser::parse_file_with_eval(f, variant) {
            Ok(mut ents) => {
                println!(
                    "  Extracted {} entities from {}",
                    ents.len(),
                    f.file_name().unwrap().to_string_lossy()
                );
                entities.append(&mut ents);
            }
            Err(e) => {
                eprintln!("  Warning: Failed to parse {}: {e}", f.display());
            }
        }
    }

    println!("Total combined entities extracted: {}", entities.len());

    // Filter to classes and (if full_mode) individuals
    let class_entities: Vec<&Entity> = entities
        .iter()
        .filter(|e| {
            if full_mode {
                matches!(e.entity_type, EntityType::Class | EntityType::Individual)
            } else {
                matches!(e.entity_type, EntityType::Class)
            }
        })
        .collect();

    println!("Concepts identified: {}", class_entities.len());

    // Build URI to CURIE mapping
    let mut uri_to_curie = BTreeMap::new();
    for e in &class_entities {
        let curie = extract_curie(&e.uri);
        uri_to_curie.insert(e.uri.clone(), curie);
    }

    // Accumulate concepts
    let mut concept_map: BTreeMap<String, JsonConcept> = BTreeMap::new();

    for e in &class_entities {
        let curie = uri_to_curie
            .get(&e.uri)
            .cloned()
            .unwrap_or_else(|| extract_curie(&e.uri));

        let parents: Vec<String> = e
            .parents
            .iter()
            .filter_map(|p_uri| uri_to_curie.get(p_uri).cloned())
            .filter(|p| p != &curie)
            .collect();

        let disjoint_with: Vec<String> = e
            .disjoint_with
            .iter()
            .filter_map(|d_uri| uri_to_curie.get(d_uri).cloned())
            .filter(|d| d != &curie)
            .collect();

        fn clean_description(raw: &str) -> String {
            let cleaned = raw
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if cleaned.len() > 300 {
                format!("{}...", &cleaned[..297])
            } else {
                cleaned
            }
        }

        let description = e
            .description
            .as_deref()
            .map(clean_description)
            .unwrap_or_else(|| e.kind.clone().unwrap_or_default());

        let entry = concept_map
            .entry(curie.clone())
            .or_insert_with(|| JsonConcept {
                curie: curie.clone(),
                label: if e.label.is_empty() {
                    curie.clone()
                } else {
                    e.label.clone()
                },
                description: description.clone(),
                parents: Vec::new(),
                children: Vec::new(),
                disjoint_with: Vec::new(),
                synonyms: Vec::new(),
            });

        if (entry.description.is_empty() || entry.description == "Class")
            && !description.is_empty()
            && description != "Class"
        {
            entry.description = description;
        }

        // Merge attributes
        for p in parents {
            if !entry.parents.contains(&p) {
                entry.parents.push(p);
            }
        }
        for d in disjoint_with {
            if !entry.disjoint_with.contains(&d) {
                entry.disjoint_with.push(d);
            }
        }
        for s in &e.synonyms {
            if !entry.synonyms.contains(s) {
                entry.synonyms.push(s.clone());
            }
        }
    }

    // Populate children links
    let curie_keys: Vec<String> = concept_map.keys().cloned().collect();
    for curie in curie_keys {
        let parents = concept_map
            .get(&curie)
            .map(|c| c.parents.clone())
            .unwrap_or_default();
        for p in parents {
            if let Some(parent_concept) = concept_map.get_mut(&p) {
                if !parent_concept.children.contains(&curie) {
                    parent_concept.children.push(curie.clone());
                }
            }
        }
    }

    let default_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("converted_ontology");

    let ontology_name = name.unwrap_or(default_name).to_string();

    let concepts: Vec<JsonConcept> = concept_map.into_values().collect();
    println!(
        "Final valid class count with graph hierarchy: {}",
        concepts.len()
    );

    Ok(JsonOntology {
        name: ontology_name,
        concepts,
    })
}

fn export_to_arrow(json_onto: &JsonOntology, output_path: &Path) -> Result<()> {
    let mut builder = OntologyBatchBuilder::new();
    let mut curie_to_id: HashMap<&str, u32> = HashMap::with_capacity(json_onto.concepts.len());

    for (idx, c) in json_onto.concepts.iter().enumerate() {
        curie_to_id.insert(c.curie.as_str(), idx as u32);
    }

    for (idx, c) in json_onto.concepts.iter().enumerate() {
        let parent_ids: Vec<u32> = c
            .parents
            .iter()
            .filter_map(|p| curie_to_id.get(p.as_str()).copied())
            .collect();

        let disjoint_ids: Vec<u32> = c
            .disjoint_with
            .iter()
            .filter_map(|d| curie_to_id.get(d.as_str()).copied())
            .collect();

        let synonyms: Vec<&str> = c.synonyms.iter().map(|s| s.as_str()).collect();

        builder.append_concept(
            idx as u32,
            &c.curie,
            &c.label,
            &c.description,
            &parent_ids,
            &disjoint_ids,
            &synonyms,
        );
    }

    let batch = builder.finish().context("Failed to build Arrow RecordBatch")?;
    write_ontology_ipc(&batch, output_path).context("Failed to write Arrow IPC")?;
    println!(
        "⚡ Successfully exported {} concepts to zero-copy Arrow IPC: {}",
        json_onto.concepts.len(),
        output_path.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: convert_ontology <input.ttl|owl|rdf|dir> <output.json|output.arrow> [ontology_name] [--full]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let full_mode = args.iter().any(|a| a == "--full");
    let name = args.get(3).filter(|&s| s != "--full").map(|s| s.as_str());

    let json_onto = process_ontology(input_path, name, full_mode)?;

    let out_p = Path::new(output_path);
    if let Some(parent) = out_p.parent() {
        fs::create_dir_all(parent)?;
    }

    if output_path.ends_with(".arrow") || output_path.ends_with(".ipc") {
        export_to_arrow(&json_onto, out_p)?;
    } else {
        let mut out_file = File::create(out_p)
            .with_context(|| format!("Failed to create output file: {output_path}"))?;
        let json_bytes = serde_json::to_vec_pretty(&json_onto)?;
        out_file.write_all(&json_bytes)?;

        println!(
            "Successfully exported {} concepts to JSON: {}",
            json_onto.concepts.len(),
            output_path
        );
    }

    Ok(())
}
