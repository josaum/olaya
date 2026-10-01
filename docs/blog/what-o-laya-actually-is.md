---
title: "What O-Laya Actually Is: An Ontology-Grounded System 1 Decision Model"
description: "A parallel and architectural response to Laya: why modern supply chain, warehousing, and agent pipelines require ontology-grounded decision models with mathematical guarantees."
date: "2026-10-01"
author: "Dionisio Chiuratto Agourakis (dionisio.ca@gmail.com)"
tags: ["ai", "decision-model", "ontology", "supply-chain", "gs1", "rust", "mcp"]
---

# What O-Laya Actually Is: An Ontology-Grounded System 1 Decision Model

*A parallel and architectural response to [What Laya Actually Is](https://relay.sparkrouter.com/ai--and--local-models/what-laya-actually-is).*

```mermaid
flowchart TD
    Start(["Incoming Warehouse & Supply Chain Stream<br/>(GTIN barcode, EPCIS sensor event, packing slip, ASN)"]) --> Input["Structured GS1 Decision Query:<br/>concept, subsumes, choice, score, noul"]
    
    subgraph S1["Stage 1: Microsecond Subgraph Routing (Rust Arrow)"]
        Input --> Router["Inverted Token Index<br/>(CamelCase-aware, IDF weighting)"]
        Onto[("GS1 Web Vocabulary Standard<br/>Product, FoodBeverage, PackagingDetails, MeatPoultry, etc.")] --> Router
        Router --> Sub["Active Candidate Subgraph<br/>(top-K leaves + complete ancestor hierarchy)"]
    end
    
    subgraph S2["Stage 2: Non-Autoregressive Scoring"]
        Sub --> Scorer["Cross-Attention Scorer<br/>(Rust Lexical: 4.8µs | Custom BERT: 12.7ms | ModernBERT: 199ms)"]
    end
    
    subgraph S3["Stage 3: Axiom Projection Kernel"]
        Scorer --> Proj["Axiom Projection (spec/AXIOMS.md)<br/>• Subsumption Monotonicity: P(child) ≤ P(parent)<br/>• Disjointness Exclusion: A ⊓ B = ⊥<br/>• Normalised Entropy Confidence"]
    end
    
    Proj --> Out["Grounded Decision Output:<br/>• Verified CURIE + Lineage Path<br/>• Full Descendant/Ancestor Probabilities<br/>• axioms_verified: true"]
    
    Out --> Decision{"Where should the result go?"}
    Decision -->|"High Confidence + Axioms Verified"| AutoAct["Act Locally & Instantly:<br/>Divert conveyer chute, route cold-chain zone, print customs label"]
    Decision -->|"Low Confidence / Unknown SKU"| EscalateLLM["Escalate to LLM (System 2):<br/>Extract unstructured vendor invoice, parse novel packaging PDF"]
    Decision -->|"Critical Inspection / Hazard"| HumanReview["Human QA Inspector & Dock Supervisor"]
    
    classDef olaya fill:#1b5e20,stroke:#66bb6a,color:#ffffff
    classDef kernel fill:#0d47a1,stroke:#42a5f5,color:#ffffff
    classDef action fill:#263238,stroke:#455a64,color:#ffffff
    class Router,Sub,Scorer,Proj olaya
    class Onto,Out kernel
    class AutoAct,EscalateLLM,HumanReview action
```

## The Non-Generative Decision Paradigm in the Physical World

In his recent piece, *What Laya Actually Is*, the author laid out an argument for non-generative classification:
> *"Laya is not a chatbot. It doesn’t write code or summarize text. It is a small local model that reads text and answers structured questions, returning probabilities instead of paragraphs."*

We agree with the fundamental premise. In modern industrial operations—automated distribution centers, automated sorters, drone pickers, and supply chain control towers—calling a cloud-hosted 70B autoregressive LLM to generate 100 tokens of JSON just to decide which conveyor chute to divert a package into is disastrously slow, prohibitively expensive, and prone to hallucinations.

**However, unconstrained classification models suffer from a fatal structural flaw: they view the physical world as flat lists of strings.**

Consider a real logistics example at a high-speed cross-dock facility:
- A barcode scanner reads GTIN `0614141000036` on an incoming pallet: *"Organic whole milk carton 1 gallon, refrigerated perishable dairy, returnable pallet."*
- You want to determine:
  1. Product category for storage zone allocation (`ColdStorage`, `DryGoods`, `Dairy`).
  2. Hazmat / environmental compliance.
  3. Packaging disposal requirements.

When you ask a flat classifier to route between `"Dairy"`, `"FoodBeverageTobaccoProduct"`, and `"Product"`, the model treats those options as disconnected, competing strings in an unconstrained softmax pool. It has no mathematical awareness that **Dairy is a sub-class of Food & Beverage**, which in turn is a **sub-class of Product**.

In real production runs, flat classifiers frequently output:
$$P(\text{MilkButterCreamYogurtCheeseEggsSubstitutes}) = 0.88 \quad \text{while} \quad P(\text{FoodBeverageTobaccoProduct}) = 0.42$$

This is a **catastrophic logical contradiction**. In formal ontology and physical reality, if an item is milk, it is *by definition* a food/beverage product. If your automated warehouse software routes based on parent categories, the item will be routed to the wrong ambient storage aisle because the parent probability mysteriously collapsed.

**O-Laya solves this by grounding decisions in formal ontology lattices like GS1.**

---

## What O-Laya Actually Is

O-Laya is an **ontology-grounded System 1 decision engine**. You provide:
1. **An Input State**: Barcode scan, OCR text from a bill of lading, IoT temperature log, or EPCIS event stream.
2. **A Formal Ontology DAG**: The **GS1 Web Vocabulary Standard** (`gs1:Product`, `gs1:FoodBeverageTobaccoProduct`, `gs1:PackagingDetails`, `gs1:MeatPoultry`, etc.), complete with `subClassOf` and `disjointWith` axioms.
3. **Structured Questions**: Formulated using 5 formal decision primitives.

In a single forward pass, O-Laya projects candidate scores through an axiom kernel ([`spec/AXIOMS.md`](https://github.com/josaum/olaya/blob/main/spec/AXIOMS.md)), guaranteeing that:
- **Subsumption Monotonicity holds**: $P(\text{child}) \le P(\text{parent})$ everywhere across the supply chain hierarchy.
- **Disjointness holds**: Mutually exclusive categories (e.g., `FrozenColdChain` $\sqcap$ `AmbientDry` $= \bot$) sum to at most 1.0.
- **Lineage is returned**: Decisions provide full explainability from the root down to the exact GS1 CURIE.
- **`axioms_verified` is computed**: Verified on the output tensor at runtime, never hard-coded.

---

## The 5 Question Primitives: A Warehousing & GS1 Walkthrough

Let's look at an automated warehouse receiving scanner running O-Laya against the GS1 Web Vocabulary:

```python
from olaya import OLaya

# Ingest standard GS1 Web Vocabulary DAG
olaya = OLaya()

state = {
    "gtin": "0614141000036",
    "scan_text": "Grade A pasteurized whole milk, 1 gallon jug with tamper-evident induction seal. Keep refrigerated at 34-38 F.",
    "sensor_temp_c": 3.2,
    "carrier": "ColdExpress Logistics",
}

questions = {
    "taxonomy_grounding": {
        "type": "concept",
        "root": "Product",
        "instructions": "Ground trade item into the specific GS1 product category."
    },
    "requires_cold_chain": {
        "type": "subsumes",
        "concept": "FoodBeverageTobaccoProduct",
        "instructions": "Does this SKU fall under perishable consumable food handling rules?"
    },
    "packaging_triage": {
        "type": "choice",
        "instructions": "Determine returnable container handling.",
        "criteria": {
            "pallet_return": "reusable wooden or composite pallet requiring return credit",
            "corrugated_bale": "cardboard master carton for bulk recycling",
            "hazardous_drum": "chemical container requiring Hazmat manifest"
        }
    },
    "dock_urgency": {
        "type": "score",
        "instructions": "Unload and cross-dock priority.",
        "criteria": ["ambient_low", "standard_dock", "expedited_perishable_cold_chain"]
    },
    "seal_breach_alert": {
        "type": "noul",
        "instructions": "Is there any indication of seal damage or temperature excursion?"
    }
}

result = olaya.system_one(state, questions)
```

### 1. `concept` (Leaf Taxonomy Grounding with Guaranteed Monotonicity)
O-Laya identifies the exact leaf concept in the GS1 hierarchy and projects ancestral probabilities:

```json
{
  "curie": "gs1:MilkButterCreamYogurtCheeseEggsSubstitutes",
  "lineage": [
    "gs1:Product",
    "gs1:FoodBeverageTobaccoProduct",
    "gs1:MilkButterCreamYogurtCheeseEggsSubstitutes"
  ],
  "candidate_probabilities": {
    "gs1:MilkButterCreamYogurtCheeseEggsSubstitutes": 0.894,
    "gs1:Beverage": 0.082,
    "gs1:MeatPoultry": 0.024
  },
  "probabilities": {
    "gs1:Product": 1.0,
    "gs1:FoodBeverageTobaccoProduct": 1.0,
    "gs1:MilkButterCreamYogurtCheeseEggsSubstitutes": 0.894
  },
  "confidence": 0.841,
  "axioms_verified": true
}
```
*Notice: Because O-Laya derives parent probabilities from descendant mass, $P(\text{Product}) \ge P(\text{FoodBeverage}) \ge P(\text{Milk})$ holds with mathematical certainty. No downstream sorting system will ever get confused by broken hierarchy math.*

### 2. `subsumes` (Principal Down-Set Membership $\downarrow C$)
Tests whether the item is in the principal down-set of `FoodBeverageTobaccoProduct`.
- Returns $P = 0.992$.
- The warehouse programmable logic controller (PLC) instantly diverts the pallet to **Aisle 4 (Refrigerated Receiving)** without querying an external LLM.

### 3. `choice` (Disjoint Packaging Rules)
Evaluates container types with formal exclusion. Pallet return is selected with $P = 0.941$, triggering an automated credit event in the warehouse management system (WMS).

### 4. `score` (Ordered Unload Priority)
Returns `2.0` on the scale (`expedited_perishable_cold_chain`) with 91.2% probability mass on level 2.

### 5. `noul` (Out-of-Domain / Anomaly Gating)
Binary hypothesis check on seal breach / temp anomaly. With `sensor_temp_c: 3.2` (within 34–38°F range), returns $P = 0.003$ ($0.3\%$). Clean bill of health; no quarantine needed.

---

## Solving the Scalability Bottleneck: From 20 Options to 100,000 SKUs

A major limitation highlighted in the original Laya post is candidate scalability:
> *"Keep option lists short. Accuracy drops past roughly 20 options."*

In modern enterprise warehousing, a standard distribution center manages **50,000 to 200,000 distinct SKUs** across hundreds of GS1 GPC (Global Product Classification) brick values. Feeding 5,000 product classes into an unconstrained prompt blows past context limits and collapses model attention.

O-Laya solves this using a **Two-Tier Subgraph Routing Architecture**:

1. **Inverted Token Indexing with IDF Weighting (Sub-Microsecond)**:
   - All GS1 labels, synonyms, GPC bricks, and semantic descriptions (`rdfs:comment`, `skos:definition`, `schema:description`) are indexed into an Arrow-native memory buffer.
   - Incoming barcodes, OCR text, or shipping manifests are tokenized with CamelCase awareness and matched in **4.8 microseconds**.
2. **Active Subgraph Extraction**:
   - The top-$K$ candidate leaves and their complete ancestor chains are extracted into an active sub-lattice.
3. **Neural Scorer**:
   - The neural decision model only reasons over the active subgraph, allowing O-Laya to scale to **hundreds of thousands of products and parts** without latency degradation.

---

## Measured Performance & Hardware Benchmarks

Measured on consumer hardware (Apple Silicon MPS / single CPU core) on operational GS1 supply chain workloads:

| Runtime / Architecture | Parameters | Dimension & Depth | Median Latency | P95 Latency | Throughput | Role in Warehouse / Supply Chain |
|---|---|---|---|---|---|---|
| **Rust Arrow Lexical** (`olaya-core`) | 0 (pure index) | Arrow IPC Buffer | **4.8 µs** | 5.2 µs | **~210,000 qps** | Embedded PLC / Conveyor Chute Divert (<1ms) |
| **Custom BERT Decision Model** | 15.37M | 4 layers, 256d | **12.71 ms** | **14.52 ms** | **231.7 qps** | Edge Scanner / Handheld Forklift Terminal |
| **ModernBERT-base Backbone** | 149.91M | 22 layers, 768d | **199.10 ms** | **217.52 ms** | **14.9 qps** | Complex ASN & Customs Document Ingestion |
| **WebMCP Agent Bridge** (JSON-RPC) | N/A (RPC) | Custom BERT | **11.57 ms** | **12.83 ms** | **86.3 qps** | Autonomous Logistics Agent (MCP 2024-11-05) |

---

## The Zero-Install Agent Wedge: `olaya-mcp`

If you are building supply chain agents with Claude Desktop, Cursor, LangChain, or CrewAI, you can drop O-Laya into your stack with a single command:

### Run Instantly via uvx or npx

```bash
# Python runtime (PyPI / uvx)
uvx olaya-mcp

# Or zero-dependency Node/TypeScript runtime (npm / npx)
npx @mycelia/olaya-mcp
```

### Auto-Configure Claude Desktop or Cursor in 1 Second

```bash
# Automatically detects and configures Claude Desktop:
uvx olaya-mcp install --client claude

# Or Cursor IDE:
uvx olaya-mcp install --client cursor
```

### Instant CLI Grounding

```bash
# Fast sub-millisecond lexical grounding
olaya-mcp ground "Organic whole milk carton 1 gallon barcode 0614141000036" --domain gs1 --mode fast

# Neural cross-encoder grounding
olaya-mcp ground "Returnable wooden euro pallet 1200x800mm with RFID tag" --domain gs1 --mode neural
```

---

## Summary: When to Use What

| Operational Need | Best Tool | Why |
|---|---|---|
| **Dispute resolution, vendor negotiation emails, custom exception letters** | **LLM (Claude, GPT, Gemini)** | Expressive autoregressive prose generation |
| **Simple 3-way email inbox triage (<10 flat strings)** | **Laya** | Fast local classifier |
| **Barcode/GTIN routing, GS1 compliance, cold-chain gating, conveyor PLC control, warehouse sorters** | **O-Laya** | **Axiom-verified, $P(\text{child}) \le P(\text{parent})$, scales to 100k+ SKUs, 4.8µs to 12ms latency, MCP-native** |

---

*Open Source Code & GS1 Datasets: [github.com/josaum/olaya](https://github.com/josaum/olaya)*
