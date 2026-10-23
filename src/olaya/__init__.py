"""O-Laya: Ontology-Grounded System 1 Decision Model."""

# Pure standard-library surface first: importing `olaya.pii` must work on a
# base install (`olaya[pii]`). Anything that needs numpy or the ML stack is
# guarded below and simply absent when its extra is not installed.
from olaya.candidate_router import CandidateRouter
from olaya.domain_pack import DomainPack, load_domain_pack
from olaya.ontology import Concept, Individual, ObjectProperty, Ontology, create_incident_ontology
from olaya.ontology_io import load_ontology, ontology_fingerprint
from olaya.owl2_rl import OWL2RLReasoner, abox_consistency_report
from olaya.pii import ObfuscationMode, ObfuscationResult, PIIEntity, PIIObfuscator
from olaya.sequence import SequenceBundle, build_ontology_sequence
from olaya.workflow import CompiledWorkflow, compile_workflow, run_workflow
from olaya.workflow_replay import replay_workflow

try:
    from olaya.axioms import calibrate_and_project_distribution, enforce_subsumption_monotonicity
    from olaya.engine import (
        ChoiceAnswer,
        ConceptAnswer,
        NoulAnswer,
        OLaya,
        ScoreAnswer,
        SubsumesAnswer,
        SystemOneResult,
    )
except ImportError:  # olaya[reasoning] not installed
    pass

try:
    from olaya.calibration import CalibrationMetrics, compute_ece, fit_temperature
    from olaya.model import OntologyDecisionHead, OntologyDecisionModel
    from olaya.train import train_olaya
except ImportError:
    pass

__all__ = [
    "ObfuscationMode",
    "ObfuscationResult",
    "PIIEntity",
    "PIIObfuscator",
    "CompiledWorkflow",
    "compile_workflow",
    "run_workflow",
    "replay_workflow",
    "Ontology",
    "Concept",
    "Individual",
    "ObjectProperty",
    "OWL2RLReasoner",
    "abox_consistency_report",
    "create_incident_ontology",
    "CandidateRouter",
    "OntologyDecisionModel",
    "OntologyDecisionHead",
    "OLaya",
    "SystemOneResult",
    "ConceptAnswer",
    "SubsumesAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "calibrate_and_project_distribution",
    "enforce_subsumption_monotonicity",
    "build_ontology_sequence",
    "SequenceBundle",
    "train_olaya",
    "compute_ece",
    "fit_temperature",
    "CalibrationMetrics",
    "load_ontology",
    "ontology_fingerprint",
    "DomainPack",
    "load_domain_pack",
]
