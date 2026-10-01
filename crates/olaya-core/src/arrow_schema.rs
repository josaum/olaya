//! Arrow-native schema definitions and zero-copy IPC helpers for O-Laya.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    ArrayRef, ListBuilder, StringBuilder, UInt32Builder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use owl_fast_core::{ConferenceEvalVariant, Entity, OntologyParser, RdfFormat};

/// Returns the standardized Arrow schema for an in-memory or on-disk Ontology.
pub fn ontology_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("concept_id", DataType::UInt32, false),
        Field::new("curie", DataType::Utf8, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("description", DataType::Utf8, true),
        Field::new(
            "parents",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            true,
        ),
        Field::new(
            "disjoint_with",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            true,
        ),
        Field::new(
            "synonyms",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]))
}

/// Returns the standardized Arrow schema for Decision Model outputs.
pub fn decision_output_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("state_id", DataType::UInt32, false),
        Field::new("question_id", DataType::Utf8, false),
        Field::new("selected_concept_id", DataType::UInt32, false),
        Field::new("selected_curie", DataType::Utf8, false),
        Field::new("selected_label", DataType::Utf8, false),
        Field::new("confidence", DataType::Float32, false),
        Field::new("axioms_verified", DataType::Boolean, false),
    ]))
}

/// Helper to serialize an ontology RecordBatch to an Arrow IPC file.
pub fn write_ontology_ipc(batch: &RecordBatch, path: &Path) -> Result<()> {
    let file = File::create(path).context("Failed to create Arrow IPC file")?;
    let mut writer = FileWriter::try_new(BufWriter::new(file), &batch.schema())
        .context("Failed to create Arrow IPC writer")?;
    writer.write(batch).context("Failed to write RecordBatch")?;
    writer.finish().context("Failed to finish IPC writer")?;
    Ok(())
}

/// Helper to read an ontology RecordBatch from an Arrow IPC file.
pub fn read_ontology_ipc(path: &Path) -> Result<RecordBatch> {
    let file = File::open(path).context("Failed to open Arrow IPC file")?;
    let mut reader = FileReader::try_new(BufReader::new(file), None)
        .context("Failed to create Arrow IPC reader")?;
    let batch = reader
        .next()
        .context("Arrow IPC file contains no record batches")??;
    Ok(batch)
}

/// Builder for constructing an Arrow ontology RecordBatch in memory.
pub struct OntologyBatchBuilder {
    pub concept_id: UInt32Builder,
    pub curie: StringBuilder,
    pub label: StringBuilder,
    pub description: StringBuilder,
    pub parents: ListBuilder<UInt32Builder>,
    pub disjoint_with: ListBuilder<UInt32Builder>,
    pub synonyms: ListBuilder<StringBuilder>,
}

impl Default for OntologyBatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OntologyBatchBuilder {
    pub fn new() -> Self {
        Self {
            concept_id: UInt32Builder::new(),
            curie: StringBuilder::new(),
            label: StringBuilder::new(),
            description: StringBuilder::new(),
            parents: ListBuilder::new(UInt32Builder::new()),
            disjoint_with: ListBuilder::new(UInt32Builder::new()),
            synonyms: ListBuilder::new(StringBuilder::new()),
        }
    }

    pub fn append_concept(
        &mut self,
        id: u32,
        curie: &str,
        label: &str,
        description: &str,
        parents: &[u32],
        disjoint_with: &[u32],
        synonyms: &[&str],
    ) {
        self.concept_id.append_value(id);
        self.curie.append_value(curie);
        self.label.append_value(label);
        self.description.append_value(description);

        // Parents list
        for &p in parents {
            self.parents.values().append_value(p);
        }
        self.parents.append(true);

        // Disjoint list
        for &d in disjoint_with {
            self.disjoint_with.values().append_value(d);
        }
        self.disjoint_with.append(true);

        // Synonyms list
        for &s in synonyms {
            self.synonyms.values().append_value(s);
        }
        self.synonyms.append(true);
    }

    pub fn finish(mut self) -> Result<RecordBatch> {
        let schema = ontology_schema();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.concept_id.finish()),
            Arc::new(self.curie.finish()),
            Arc::new(self.label.finish()),
            Arc::new(self.description.finish()),
            Arc::new(self.parents.finish()),
            Arc::new(self.disjoint_with.finish()),
            Arc::new(self.synonyms.finish()),
        ];
        RecordBatch::try_new(schema, columns).context("Failed to construct Arrow RecordBatch")
    }

    /// Construct an Arrow ontology RecordBatch directly from parsed owl-fast-core entities.
    pub fn from_entities(entities: &[Entity]) -> Result<RecordBatch> {
        let mut builder = Self::new();

        let mut uri_to_id: HashMap<&str, u32> = HashMap::with_capacity(entities.len());
        for (idx, entity) in entities.iter().enumerate() {
            uri_to_id.insert(&entity.uri, idx as u32);
        }

        for (idx, entity) in entities.iter().enumerate() {
            let parent_ids: Vec<u32> = entity
                .parents
                .iter()
                .filter_map(|p_uri| uri_to_id.get(p_uri.as_str()).copied())
                .collect();

            let disjoint_ids: Vec<u32> = entity
                .disjoint_with
                .iter()
                .filter_map(|d_uri| uri_to_id.get(d_uri.as_str()).copied())
                .collect();

            let synonyms: Vec<&str> = entity.synonyms.iter().map(|s| s.as_str()).collect();
            let curie = extract_curie(&entity.uri);
            let desc = entity.kind.as_deref().unwrap_or("");

            builder.append_concept(
                idx as u32,
                &curie,
                &entity.label,
                desc,
                &parent_ids,
                &disjoint_ids,
                &synonyms,
            );
        }

        builder.finish()
    }

    /// Parse an ontology file using owl-fast-core and build an Arrow ontology RecordBatch.
    pub fn from_owl_file<P: AsRef<Path>>(path: P) -> Result<RecordBatch> {
        let entities = OntologyParser::parse_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to parse ontology file: {e}"))?;
        Self::from_entities(&entities)
    }

    /// Parse an ontology file with a specific evaluation variant (M1, M2, M3, Full).
    pub fn from_owl_file_with_eval<P: AsRef<Path>>(
        path: P,
        eval: ConferenceEvalVariant,
    ) -> Result<RecordBatch> {
        let entities = OntologyParser::parse_file_with_eval(path, eval)
            .map_err(|e| anyhow::anyhow!("Failed to parse ontology file: {e}"))?;
        Self::from_entities(&entities)
    }

    /// Parse raw ontology bytes (e.g. Turtle or RDF/XML) and build an Arrow ontology RecordBatch.
    pub fn from_owl_bytes(bytes: &[u8], format: RdfFormat) -> Result<RecordBatch> {
        let entities = OntologyParser::parse_bytes(bytes, format)
            .map_err(|e| anyhow::anyhow!("Failed to parse ontology bytes: {e}"))?;
        Self::from_entities(&entities)
    }
}

/// Helper to extract a short CURIE or local name from an ontology URI.
pub fn extract_curie(uri: &str) -> String {
    if !uri.contains("://") {
        return uri.to_string();
    }
    let local = if let Some(pos) = uri.rfind('#') {
        &uri[pos + 1..]
    } else if let Some(pos) = uri.rfind('/') {
        &uri[pos + 1..]
    } else {
        uri
    };
    if let Some((prefix, id)) = local.split_once('_') {
        if !prefix.is_empty() && !id.is_empty() {
            return format!("{prefix}:{id}");
        }
    }
    local.to_string()
}

