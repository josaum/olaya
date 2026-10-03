//! Fast Formal Concept Analysis (FCA) engine.
//!
//! Provides high-performance formal context definition, Galois derivation operators,
//! Close-by-One (CbO) concept lattice generation, and conversion to OWL/DAG taxonomies.
//!
//! Based on mathematical Formal Concept Analysis (Ganter & Wille 1999, Kuznetsov 1993, Andrews 2011).

use crate::Entity;
use std::collections::HashMap;

/// Dense bitset backed by 64-bit words for ultra-fast set operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// Creates a new BitSet of the given capacity in bits, initialized to 0.
    pub fn new(len: usize) -> Self {
        let num_words = len.div_ceil(64);
        Self {
            words: vec![0; num_words],
            len,
        }
    }

    /// Creates a full BitSet where the first `len` bits are set to 1.
    pub fn full(len: usize) -> Self {
        let num_words = len.div_ceil(64);
        let mut words = vec![!0u64; num_words];
        if len > 0 && !len.is_multiple_of(64) {
            let last_valid = len % 64;
            let mask = (1u64 << last_valid) - 1;
            if let Some(last) = words.last_mut() {
                *last &= mask;
            }
        }
        Self { words, len }
    }

    /// Sets the bit at index `idx` to `val`.
    pub fn set(&mut self, idx: usize, val: bool) {
        if idx >= self.len {
            return;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        if val {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Returns the value of the bit at index `idx`.
    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        (self.words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Returns true if this bitset is a subset of `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        let min_words = self.words.len().min(other.words.len());
        for i in 0..min_words {
            if (self.words[i] & !other.words[i]) != 0 {
                return false;
            }
        }
        for word in self.words.iter().skip(min_words) {
            if *word != 0 {
                return false;
            }
        }
        true
    }

    /// Bitwise in-place intersection (self = self AND other).
    pub fn intersect_with(&mut self, other: &Self) {
        let min_words = self.words.len().min(other.words.len());
        for i in 0..min_words {
            self.words[i] &= other.words[i];
        }
        for word in self.words.iter_mut().skip(min_words) {
            *word = 0;
        }
    }

    /// Bitwise in-place union (self = self OR other).
    pub fn union_with(&mut self, other: &Self) {
        let min_words = self.words.len().min(other.words.len());
        for i in 0..min_words {
            self.words[i] |= other.words[i];
        }
    }

    /// Counts the total number of set bits (population count / cardinality).
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Returns an iterator over the indices of all set bits in ascending order.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(move |(w_idx, &word)| {
                let mut w = word;
                std::iter::from_fn(move || {
                    if w == 0 {
                        None
                    } else {
                        let trailing = w.trailing_zeros();
                        w &= w - 1; // Clear lowest set bit
                        let bit_idx = w_idx * 64 + trailing as usize;
                        if bit_idx < self.len {
                            Some(bit_idx)
                        } else {
                            None
                        }
                    }
                })
            })
    }

    /// Returns the length in bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if there are zero bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0 || self.count_ones() == 0
    }
}

/// A formal context $\mathbb{K} = (G, M, I)$ where:
/// - $G$ is a set of objects
/// - $M$ is a set of attributes
/// - $I \subseteq G \times M$ is a binary incidence relation
#[derive(Debug, Clone)]
pub struct FormalContext {
    /// Object names (G).
    pub objects: Vec<String>,
    /// Attribute names (M).
    pub attributes: Vec<String>,
    object_map: HashMap<String, usize>,
    attribute_map: HashMap<String, usize>,
    /// Incidence relation: for each object g, bitset of its attributes g^↑.
    pub object_incidence: Vec<BitSet>,
    /// Inverse incidence relation: for each attribute m, bitset of its objects m^↓.
    pub attribute_incidence: Vec<BitSet>,
}

impl FormalContext {
    /// Creates a new empty formal context with the specified objects and attributes.
    pub fn new(objects: Vec<String>, attributes: Vec<String>) -> Self {
        let num_obj = objects.len();
        let num_attr = attributes.len();

        let object_map: HashMap<String, usize> = objects
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let attribute_map: HashMap<String, usize> = attributes
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let object_incidence = vec![BitSet::new(num_attr); num_obj];
        let attribute_incidence = vec![BitSet::new(num_obj); num_attr];

        Self {
            objects,
            attributes,
            object_map,
            attribute_map,
            object_incidence,
            attribute_incidence,
        }
    }

    /// Builds a formal context from a list of (object, attribute) pair names.
    pub fn from_named_pairs(pairs: &[(&str, &str)]) -> Self {
        let mut obj_names: Vec<String> = Vec::new();
        let mut attr_names: Vec<String> = Vec::new();
        let mut obj_seen: HashMap<String, usize> = HashMap::new();
        let mut attr_seen: HashMap<String, usize> = HashMap::new();

        for &(o, a) in pairs {
            if !obj_seen.contains_key(o) {
                let idx = obj_names.len();
                obj_seen.insert(o.to_string(), idx);
                obj_names.push(o.to_string());
            }
            if !attr_seen.contains_key(a) {
                let idx = attr_names.len();
                attr_seen.insert(a.to_string(), idx);
                attr_names.push(a.to_string());
            }
        }

        let mut ctx = Self::new(obj_names, attr_names);
        for &(o, a) in pairs {
            let o_idx = ctx.object_map[o];
            let a_idx = ctx.attribute_map[a];
            ctx.add_incidence(o_idx, a_idx);
        }
        ctx
    }

    /// Builds a formal context from ontology entities and their associated parent/synonym/type relations.
    pub fn from_entities(entities: &[Entity]) -> Self {
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        for e in entities {
            for p in &e.parents {
                pairs.push((e.uri.as_str(), p.as_str()));
            }
            for s in &e.synonyms {
                pairs.push((e.uri.as_str(), s.as_str()));
            }
        }
        Self::from_named_pairs(&pairs)
    }

    /// Adds an incidence relation (object_idx, attribute_idx) ∈ I.
    pub fn add_incidence(&mut self, object_idx: usize, attribute_idx: usize) {
        if object_idx < self.objects.len() && attribute_idx < self.attributes.len() {
            self.object_incidence[object_idx].set(attribute_idx, true);
            self.attribute_incidence[attribute_idx].set(object_idx, true);
        }
    }

    /// Derivation operator $A^\uparrow$: attributes shared by all objects in extent $A$.
    pub fn derive_intent(&self, extent: &BitSet) -> BitSet {
        if extent.count_ones() == 0 {
            // Empty extent derives all attributes: ∅^↑ = M
            return BitSet::full(self.attributes.len());
        }

        let mut intent = BitSet::full(self.attributes.len());
        for obj_idx in extent.iter_ones() {
            intent.intersect_with(&self.object_incidence[obj_idx]);
        }
        intent
    }

    /// Derivation operator $B^\downarrow$: objects possessing all attributes in intent $B$.
    pub fn derive_extent(&self, intent: &BitSet) -> BitSet {
        if intent.count_ones() == 0 {
            // Empty intent derives all objects: ∅^↓ = G
            return BitSet::full(self.objects.len());
        }

        let mut extent = BitSet::full(self.objects.len());
        for attr_idx in intent.iter_ones() {
            extent.intersect_with(&self.attribute_incidence[attr_idx]);
        }
        extent
    }

    /// Closure operator on extents: $A^{\uparrow\downarrow}$.
    pub fn close_extent(&self, extent: &BitSet) -> BitSet {
        let intent = self.derive_intent(extent);
        self.derive_extent(&intent)
    }

    /// Closure operator on intents: $B^{\downarrow\uparrow}$.
    pub fn close_intent(&self, intent: &BitSet) -> BitSet {
        let extent = self.derive_extent(intent);
        self.derive_intent(&extent)
    }

    /// Number of objects in context.
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Number of attributes in context.
    pub fn num_attributes(&self) -> usize {
        self.attributes.len()
    }
}

/// A formal concept $(A, B)$ satisfying $A^\uparrow = B$ and $B^\downarrow = A$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormalConcept {
    /// Zero-based identifier of the concept in the lattice.
    pub id: usize,
    /// Extent $A \subseteq G$.
    pub extent: BitSet,
    /// Intent $B \subseteq M$.
    pub intent: BitSet,
}

impl FormalConcept {
    /// Returns the number of objects in the concept's extent.
    pub fn extent_size(&self) -> usize {
        self.extent.count_ones()
    }

    /// Returns the number of attributes in the concept's intent.
    pub fn intent_size(&self) -> usize {
        self.intent.count_ones()
    }

    /// Resolves object names in the concept's extent.
    pub fn object_names<'a>(&self, ctx: &'a FormalContext) -> Vec<&'a str> {
        self.extent
            .iter_ones()
            .map(|i| ctx.objects[i].as_str())
            .collect()
    }

    /// Resolves attribute names in the concept's intent.
    pub fn attribute_names<'a>(&self, ctx: &'a FormalContext) -> Vec<&'a str> {
        self.intent
            .iter_ones()
            .map(|i| ctx.attributes[i].as_str())
            .collect()
    }
}

/// An attribute implication $X \implies Y$ holding across the context.
#[derive(Debug, Clone, PartialEq)]
pub struct Implication {
    /// Premise attribute set $X$.
    pub premise: BitSet,
    /// Conclusion attribute set $Y$.
    pub conclusion: BitSet,
    /// Number of objects supporting the premise ($|X^\downarrow|$).
    pub support: usize,
    /// Confidence fraction $\frac{|(X \cup Y)^\downarrow|}{|X^\downarrow|}$.
    pub confidence: f64,
}

/// Complete concept lattice $\mathcal{B}(G, M, I)$ with subconcept-superconcept covering relations.
#[derive(Debug, Clone)]
pub struct ConceptLattice {
    /// The underlying formal context.
    pub context: FormalContext,
    /// All formal concepts in the lattice.
    pub concepts: Vec<FormalConcept>,
    /// Direct superconcepts (covering upper neighbors) for each concept.
    pub parents: Vec<Vec<usize>>,
    /// Direct subconcepts (covering lower neighbors) for each concept.
    pub children: Vec<Vec<usize>>,
    /// Index of the top concept ($\top = (G, G^\uparrow)$).
    pub top: usize,
    /// Index of the bottom concept ($\bot = (M^\downarrow, M)$).
    pub bottom: usize,
}

impl ConceptLattice {
    /// Builds the concept lattice from a formal context using the Close-by-One (CbO) algorithm.
    pub fn build(context: FormalContext) -> Self {
        let num_obj = context.num_objects();
        let mut concepts: Vec<FormalConcept> = Vec::new();

        // 1. Recursive Close-by-One (CbO) enumeration
        fn cbo_recurse(
            curr_extent: &BitSet,
            start_obj: usize,
            context: &FormalContext,
            concepts: &mut Vec<FormalConcept>,
        ) {
            let curr_intent = context.derive_intent(curr_extent);
            let id = concepts.len();
            concepts.push(FormalConcept {
                id,
                extent: curr_extent.clone(),
                intent: curr_intent,
            });

            for y in start_obj..context.num_objects() {
                if !curr_extent.get(y) {
                    let mut next_ext = curr_extent.clone();
                    next_ext.set(y, true);
                    let closed_ext = context.close_extent(&next_ext);

                    // Canonical test: closed_ext is canonical if no j < y was added
                    let mut is_canonical = true;
                    for j in 0..y {
                        if closed_ext.get(j) && !curr_extent.get(j) {
                            is_canonical = false;
                            break;
                        }
                    }

                    if is_canonical {
                        cbo_recurse(&closed_ext, y + 1, context, concepts);
                    }
                }
            }
        }

        let empty_extent = BitSet::new(num_obj);
        let bottom_extent = context.close_extent(&empty_extent);
        cbo_recurse(&bottom_extent, 0, &context, &mut concepts);

        // 2. Compute transitive reduction (covering relation / Hasse diagram)
        let num_concepts = concepts.len();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); num_concepts];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_concepts];

        // Sort indices by ascending extent cardinality
        let mut sorted_indices: Vec<usize> = (0..num_concepts).collect();
        sorted_indices.sort_by_key(|&idx| concepts[idx].extent_size());

        for i in 0..num_concepts {
            let u = sorted_indices[i];
            let u_ext = &concepts[u].extent;
            let u_size = u_ext.count_ones();

            let mut upper_candidates: Vec<usize> = Vec::new();
            for &v in sorted_indices.iter().skip(i + 1) {
                let v_ext = &concepts[v].extent;
                if v_ext.count_ones() > u_size && u_ext.is_subset_of(v_ext) {
                    upper_candidates.push(v);
                }
            }

            // Keep only minimal upper concepts (no intermediate concept in between)
            for &c in &upper_candidates {
                let c_ext = &concepts[c].extent;
                let mut is_direct = true;
                for &other in &parents[u] {
                    if concepts[other].extent.is_subset_of(c_ext) {
                        is_direct = false;
                        break;
                    }
                }
                if is_direct {
                    parents[u].push(c);
                    children[c].push(u);
                }
            }
        }

        // Identify Top and Bottom concepts
        let mut top = 0;
        let mut bottom = 0;
        let mut max_extent = 0;
        let mut min_extent = usize::MAX;

        for (i, c) in concepts.iter().enumerate() {
            let sz = c.extent_size();
            if sz > max_extent {
                max_extent = sz;
                top = i;
            }
            if sz < min_extent {
                min_extent = sz;
                bottom = i;
            }
        }

        Self {
            context,
            concepts,
            parents,
            children,
            top,
            bottom,
        }
    }

    /// Computes the infimum / meet ($\wedge$) of two concepts in the lattice:
    /// $(A_1, B_1) \wedge (A_2, B_2) = (A_1 \cap A_2, (B_1 \cup B_2)^{\downarrow\uparrow})$.
    pub fn meet(&self, c1: usize, c2: usize) -> usize {
        let mut meet_extent = self.concepts[c1].extent.clone();
        meet_extent.intersect_with(&self.concepts[c2].extent);
        let closed_extent = self.context.close_extent(&meet_extent);

        for (i, c) in self.concepts.iter().enumerate() {
            if c.extent == closed_extent {
                return i;
            }
        }
        self.bottom
    }

    /// Computes the supremum / join ($\vee$) of two concepts in the lattice:
    /// $(A_1, B_1) \vee (A_2, B_2) = ((A_1 \cup A_2)^{\uparrow\downarrow}, B_1 \cap B_2)$.
    pub fn join(&self, c1: usize, c2: usize) -> usize {
        let mut meet_intent = self.concepts[c1].intent.clone();
        meet_intent.intersect_with(&self.concepts[c2].intent);
        let closed_intent = self.context.close_intent(&meet_intent);

        for (i, c) in self.concepts.iter().enumerate() {
            if c.intent == closed_intent {
                return i;
            }
        }
        self.top
    }

    /// Tests if superconcept subsumes subconcept: $(A_1, B_1) \le (A_2, B_2) \iff A_1 \subseteq A_2$.
    pub fn subsumes(&self, superconcept: usize, subconcept: usize) -> bool {
        self.concepts[subconcept]
            .extent
            .is_subset_of(&self.concepts[superconcept].extent)
    }

    /// Tests whether the concept lattice satisfies the distributive law:
    /// $x \wedge (y \vee z) = (x \wedge y) \vee (x \wedge z)$ for all concepts.
    ///
    /// By Birkhoff's Theorem, a finite lattice is a Heyting algebra if and only if it is distributive.
    pub fn is_distributive(&self) -> bool {
        let n = self.concepts.len();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    let y_join_z = self.join(y, z);
                    let lhs = self.meet(x, y_join_z);

                    let x_meet_y = self.meet(x, y);
                    let x_meet_z = self.meet(x, z);
                    let rhs = self.join(x_meet_y, x_meet_z);

                    if lhs != rhs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Computes the Heyting relative pseudocomplement $(a \to b) = \bigvee \{c \in L \mid c \wedge a \le b\}$.
    ///
    /// In a distributive concept lattice (Heyting algebra), this represents the strongest implication concept
    /// that, when conjoined with `a`, guarantees `b`.
    pub fn implies(&self, a: usize, b: usize) -> usize {
        let mut result = self.bottom;
        for (i, _) in self.concepts.iter().enumerate() {
            let meet_val = self.meet(i, a);
            if self.subsumes(b, meet_val) {
                result = self.join(result, i);
            }
        }
        result
    }

    /// Computes the Heyting pseudocomplement / intuitionistic negation: $\neg a = (a \to \bot)$.
    pub fn negate(&self, a: usize) -> usize {
        self.implies(a, self.bottom)
    }

    /// Returns true if concept is the top element $\top = (G, G^\uparrow)$ (tautology).
    pub fn is_tautology(&self, concept_idx: usize) -> bool {
        concept_idx == self.top
    }

    /// Extracts sound attribute implications ($X \implies Y$) from the lattice.
    pub fn compute_canonical_implications(&self) -> Vec<Implication> {
        let mut implications = Vec::new();
        for c in &self.concepts {
            if c.extent_size() == 0 {
                continue;
            }
            let intent = &c.intent;
            for other in &self.concepts {
                if other.id != c.id && c.extent.is_subset_of(&other.extent) {
                    let conclusion = other.intent.clone();
                    let support = c.extent_size();
                    implications.push(Implication {
                        premise: intent.clone(),
                        conclusion,
                        support,
                        confidence: 1.0,
                    });
                }
            }
        }
        implications
    }

    /// Converts the concept lattice into OWL-compatible [`Entity`] records.
    ///
    /// Each concept is represented as an ontological Class, with:
    /// - `uri`: prefix-namespaced concept CURIE (e.g. `fca:concept_42`)
    /// - `label`: summary of salient extent objects and intent attributes
    /// - `description`: full extent and intent listing
    /// - `parents`: URIs of direct superconcepts in the lattice covering relation
    /// - `children`: URIs of direct subconcepts in the lattice covering relation
    pub fn to_ontology_entities(&self, uri_prefix: &str) -> Vec<Entity> {
        self.concepts
            .iter()
            .map(|c| {
                let curie = format!("{}:concept_{}", uri_prefix, c.id);
                let obj_labels = c.object_names(&self.context);
                let attr_labels = c.attribute_names(&self.context);

                let label = if !attr_labels.is_empty() {
                    attr_labels.join(" + ")
                } else if !obj_labels.is_empty() {
                    obj_labels.join(", ")
                } else {
                    format!("Concept {}", c.id)
                };

                let description = format!(
                    "FCA Concept {} | Extent ({}/{}): [{}] | Intent ({}/{}): [{}]",
                    c.id,
                    c.extent_size(),
                    self.context.num_objects(),
                    obj_labels.join(", "),
                    c.intent_size(),
                    self.context.num_attributes(),
                    attr_labels.join(", ")
                );

                let parents: Vec<String> = self.parents[c.id]
                    .iter()
                    .map(|&p| format!("{}:concept_{}", uri_prefix, p))
                    .collect();

                let children: Vec<String> = self.children[c.id]
                    .iter()
                    .map(|&ch| format!("{}:concept_{}", uri_prefix, ch))
                    .collect();

                Entity {
                    uri: curie,
                    entity_type: crate::EntityType::Class,
                    kind: None,
                    label,
                    synonyms: obj_labels.iter().map(|s| s.to_string()).collect(),
                    parents,
                    children,
                    depth: self.parents[c.id].len(),
                    disjoint_with: Vec::new(),
                    same_as: Vec::new(),
                    equivalent_properties: Vec::new(),
                    siblings: Vec::new(),
                    description: Some(description),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_operations() {
        let mut b1 = BitSet::new(130);
        b1.set(10, true);
        b1.set(65, true);
        b1.set(129, true);

        assert_eq!(b1.count_ones(), 3);
        assert!(b1.get(10));
        assert!(b1.get(65));
        assert!(b1.get(129));
        assert!(!b1.get(11));

        let ones: Vec<usize> = b1.iter_ones().collect();
        assert_eq!(ones, vec![10, 65, 129]);

        let mut b2 = BitSet::new(130);
        b2.set(10, true);
        b2.set(65, true);

        assert!(b2.is_subset_of(&b1));
        assert!(!b1.is_subset_of(&b2));

        let mut b3 = b1.clone();
        b3.intersect_with(&b2);
        assert_eq!(b3.count_ones(), 2);
        assert!(!b3.get(129));
    }

    #[test]
    fn test_formal_context_galois_connection() {
        // Living beings formal context (classical Ganter & Wille example)
        let pairs = [
            ("leech", "needs_water"),
            ("bream", "needs_water"),
            ("bream", "can_fly"), // flying fish proxy
            ("frog", "needs_water"),
            ("frog", "has_limbs"),
            ("dog", "needs_water"),
            ("dog", "has_limbs"),
            ("dog", "mammal"),
            ("bat", "needs_water"),
            ("bat", "has_limbs"),
            ("bat", "can_fly"),
            ("bat", "mammal"),
        ];

        let ctx = FormalContext::from_named_pairs(&pairs);
        assert_eq!(ctx.num_objects(), 5);
        assert_eq!(ctx.num_attributes(), 4);

        // Galois connection property: A ⊆ B^↓ <=> B ⊆ A^↑
        let mut extent = BitSet::new(ctx.num_objects());
        extent.set(ctx.object_map["dog"], true);
        extent.set(ctx.object_map["bat"], true);

        let intent = ctx.derive_intent(&extent);
        // Both dog and bat share: needs_water, has_limbs, mammal
        assert!(intent.get(ctx.attribute_map["needs_water"]));
        assert!(intent.get(ctx.attribute_map["has_limbs"]));
        assert!(intent.get(ctx.attribute_map["mammal"]));
        assert!(!intent.get(ctx.attribute_map["can_fly"])); // dog cannot fly

        // Closure is idempotent: (A^↑↓)^↑↓ = A^↑↓
        let closed_once = ctx.close_extent(&extent);
        let closed_twice = ctx.close_extent(&closed_once);
        assert_eq!(closed_once, closed_twice);
    }

    #[test]
    fn test_concept_lattice_construction_and_lattice_laws() {
        let pairs = [
            ("Apple", "Fruit"),
            ("Apple", "Sweet"),
            ("Lemon", "Fruit"),
            ("Lemon", "Sour"),
            ("Sugar", "Sweet"),
            ("Vinegar", "Sour"),
        ];

        let ctx = FormalContext::from_named_pairs(&pairs);
        let lattice = ConceptLattice::build(ctx);

        assert!(!lattice.concepts.is_empty());
        // Verify Top concept contains all objects
        let top = &lattice.concepts[lattice.top];
        assert_eq!(top.extent_size(), lattice.context.num_objects());

        // Verify meet and join operations
        for i in 0..lattice.concepts.len() {
            for j in 0..lattice.concepts.len() {
                let m = lattice.meet(i, j);
                let join_idx = lattice.join(i, j);

                // Meet extent must be subset of both extents
                assert!(lattice.concepts[m]
                    .extent
                    .is_subset_of(&lattice.concepts[i].extent));
                assert!(lattice.concepts[m]
                    .extent
                    .is_subset_of(&lattice.concepts[j].extent));

                // Join intent must be subset of both intents
                assert!(lattice.concepts[join_idx]
                    .intent
                    .is_subset_of(&lattice.concepts[i].intent));
                assert!(lattice.concepts[join_idx]
                    .intent
                    .is_subset_of(&lattice.concepts[j].intent));
            }
        }

        // Test conversion to ontology entities
        let entities = lattice.to_ontology_entities("fca");
        assert_eq!(entities.len(), lattice.concepts.len());
        for e in &entities {
            assert!(e.uri.starts_with("fca:concept_"));
            assert!(e.description.is_some());
        }
    }

    #[test]
    fn test_heyting_algebra_pseudocomplement_and_intuitionistic_negation() {
        // Boolean powerset / distributive context
        let pairs = [
            ("Obj1", "AttrA"),
            ("Obj2", "AttrB"),
            ("Obj3", "AttrC"),
            ("Obj12", "AttrA"),
            ("Obj12", "AttrB"),
            ("Obj13", "AttrA"),
            ("Obj13", "AttrC"),
            ("Obj23", "AttrB"),
            ("Obj23", "AttrC"),
            ("Obj123", "AttrA"),
            ("Obj123", "AttrB"),
            ("Obj123", "AttrC"),
        ];

        let ctx = FormalContext::from_named_pairs(&pairs);
        let lattice = ConceptLattice::build(ctx);

        assert!(
            lattice.is_distributive(),
            "Powerset lattice must be distributive"
        );

        // 1. Identity of Implication: a -> a = Top
        for a in 0..lattice.concepts.len() {
            let imp = lattice.implies(a, a);
            assert!(
                lattice.is_tautology(imp),
                "a -> a must equal Top for all concepts"
            );
        }

        // 2. Implication to Top: a -> Top = Top
        for a in 0..lattice.concepts.len() {
            let imp = lattice.implies(a, lattice.top);
            assert!(lattice.is_tautology(imp));
        }

        // 3. Negation of Bottom: ~Bottom = Top
        assert_eq!(lattice.negate(lattice.bottom), lattice.top);

        // 4. Negation of Top: ~Top = Bottom
        assert_eq!(lattice.negate(lattice.top), lattice.bottom);

        // 5. Heyting Adjunction: c <= (a -> b) <=> c ^ a <= b
        for a in 0..lattice.concepts.len() {
            for b in 0..lattice.concepts.len() {
                let a_imp_b = lattice.implies(a, b);
                for c in 0..lattice.concepts.len() {
                    let left_side = lattice.subsumes(a_imp_b, c);
                    let c_meet_a = lattice.meet(c, a);
                    let right_side = lattice.subsumes(b, c_meet_a);
                    assert_eq!(
                        left_side, right_side,
                        "Heyting adjunction must hold for all (a, b, c) in distributive lattice"
                    );
                }
            }
        }
    }
}
