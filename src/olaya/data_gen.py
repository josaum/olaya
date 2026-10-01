"""Synthetic incident data generator for training O-Laya decision heads.

Generates realistic operational incidents across the incident ontology classes,
providing ground truth training pairs for all 5 question types:
- concept: Multi-class triage among leaf concepts
- subsumes: Calibrated yes/no membership in the down-set of any target concept
- choice: Discrete remediation actions
- score: Severity/disruption rubric level (0 to 3)
- noul: Operational escalation decision (P(true))
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from olaya.ontology import Concept, Ontology, create_incident_ontology


@dataclass
class IncidentInstance:
    """A generated incident with its state and typed task targets."""

    state: dict[str, Any]
    target_curie: str
    target_action: str
    severity: int  # 0: low, 1: medium, 2: urgent, 3: critical
    escalate: bool  # whether on-call page is required


# Domain-specific incident templates
INCIDENT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "INCIDENT:DB_OUTAGE": [
        {
            "service": "postgres-primary",
            "log_level": "FATAL",
            "error": "terminating connection due to administrator command: database backend unavailable",
            "action": "failover_to_replica",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "mysql-cluster",
            "log_level": "CRITICAL",
            "error": "Connection refused to primary database host db-01.internal on port 3306",
            "action": "failover_to_replica",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "aurora-db-reader",
            "log_level": "ERROR",
            "error": "Database cluster unreachable: all read replicas failed health check within 30s",
            "action": "restart_db_instance",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "pgbouncer-pooler",
            "log_level": "FATAL",
            "error": "server login failed: server closed the connection unexpectedly (server crashed)",
            "action": "restart_connection_pooler",
            "severity": 3,
            "escalate": True,
        },
    ],
    "INCIDENT:DB_DEADLOCK": [
        {
            "service": "billing-worker",
            "log_level": "ERROR",
            "error": "Deadlock detected while updating customer subscription balances in PostgreSQL cluster.",
            "action": "kill_blocking_queries",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "order-processing-service",
            "log_level": "ERROR",
            "error": "Process 18492 waits for ExclusiveLock on row of table orders; blocked by process 18401",
            "action": "kill_blocking_queries",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "inventory-lock-manager",
            "log_level": "WARN",
            "error": "Lock wait timeout exceeded; try restarting transaction in catalog reservation",
            "action": "retry_with_exponential_backoff",
            "severity": 1,
            "escalate": False,
        },
        {
            "service": "ledger-writer",
            "log_level": "ERROR",
            "error": "Transaction deadlock detected on concurrent journal entry updates for account ID 98214",
            "action": "kill_blocking_queries",
            "severity": 2,
            "escalate": True,
        },
    ],
    "INCIDENT:DATA_PIPELINE": [
        {
            "service": "kafka-event-consumer",
            "log_level": "ERROR",
            "error": "Kafka consumer lag exceeding 450,000 messages on topic telemetry-events-v1",
            "action": "scale_consumer_replicas",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "airflow-scheduler",
            "log_level": "ERROR",
            "error": "DAG run data_warehouse_sync failed: upstream task extract_postgres_cdc timed out",
            "action": "reprocess_failed_offset",
            "severity": 1,
            "escalate": False,
        },
        {
            "service": "flink-realtime-agg",
            "log_level": "CRITICAL",
            "error": "Flink job failed with OutOfMemoryError in checkpoint barrier alignment buffer",
            "action": "restart_stream_processor",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "celery-worker-etl",
            "log_level": "ERROR",
            "error": "Data pipeline stall: message offset 84921 stuck in unacknowledged state for >20m",
            "action": "reprocess_failed_offset",
            "severity": 1,
            "escalate": False,
        },
    ],
    "INCIDENT:AUTH_BREACH": [
        {
            "service": "api-gateway-auth",
            "log_level": "CRITICAL",
            "error": "Unauthorized token issuance detected from unverified JWT signer certificate",
            "action": "rotate_signing_keys",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "iam-service",
            "log_level": "CRITICAL",
            "error": "Leaked admin API secret key used from anonymous Tor exit node IP 185.220.101.5",
            "action": "revoke_compromised_token",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "identity-proxy",
            "log_level": "FATAL",
            "error": "Anomalous volume of credential stuffing attempts (>50,000/min) targeting customer logins",
            "action": "block_source_ip_range",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "oauth-token-issuer",
            "log_level": "CRITICAL",
            "error": "Security breach alert: revoked refresh token used to request active access grants",
            "action": "revoke_compromised_token",
            "severity": 3,
            "escalate": True,
        },
    ],
    "INCIDENT:DDOS": [
        {
            "service": "edge-cloudflare-ingress",
            "log_level": "CRITICAL",
            "error": "High-volume distributed SYN flood traffic overwhelming edge ingress bandwidth (120 Gbps)",
            "action": "enable_ddos_under_attack_mode",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "envoy-gateway-public",
            "log_level": "FATAL",
            "error": "Incoming HTTP request rate spiked 65x normal baseline with randomized bot user-agents",
            "action": "rate_limit_edge_waf",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "api-frontdoor",
            "log_level": "CRITICAL",
            "error": "Edge ingress capacity exhausted: TCP connection backlog full on all public load balancers",
            "action": "enable_ddos_under_attack_mode",
            "severity": 3,
            "escalate": True,
        },
    ],
    "INCIDENT:PAYMENT_GATEWAY": [
        {
            "service": "stripe-webhook-handler",
            "log_level": "ERROR",
            "error": "Stripe API returning HTTP 500 Internal Server Error during customer subscription charge capture",
            "action": "route_to_backup_gateway",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "checkout-service",
            "log_level": "ERROR",
            "error": "Payment decline storm: 78% of card authorizations failed with gateway timeout from processor",
            "action": "route_to_backup_gateway",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "adyen-connector",
            "log_level": "CRITICAL",
            "error": "Merchant banking connector offline: TLS handshake timeout to payment processing gateway",
            "action": "retry_pending_webhooks",
            "severity": 3,
            "escalate": True,
        },
    ],
    "INCIDENT:INVOICE_MISCALC": [
        {
            "service": "tax-calculation-engine",
            "log_level": "ERROR",
            "error": "Negative VAT tax amount calculated for EU enterprise billing invoice #INV-2026-9812",
            "action": "halt_invoice_dispatch",
            "severity": 1,
            "escalate": False,
        },
        {
            "service": "invoice-generator",
            "log_level": "WARN",
            "error": "Currency conversion applied inverted exchange rate resulting in customer 100x overcharge",
            "action": "recalculate_ledger_balances",
            "severity": 1,
            "escalate": False,
        },
        {
            "service": "proration-calculator",
            "log_level": "WARN",
            "error": "Tier upgrade proration produced negative line item total on invoice renewal",
            "action": "halt_invoice_dispatch",
            "severity": 0,
            "escalate": False,
        },
    ],
    "INCIDENT:INFRA": [
        {
            "service": "k8s-node-worker-pool",
            "log_level": "CRITICAL",
            "error": "Kernel panic on hypervisor node-04; NVMe driver I/O timeout causing filesystem read-only remount",
            "action": "cordon_and_drain_node",
            "severity": 3,
            "escalate": True,
        },
        {
            "service": "baremetal-switch",
            "log_level": "ERROR",
            "error": "Packet loss on physical 100GbE uplink interface eth0 reached 18% due to optical transceiver degradation",
            "action": "reboot_host",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "cgroup-memory-controller",
            "log_level": "ERROR",
            "error": "Memory cgroup OOM killer invoked repeatedly across infrastructure host pods",
            "action": "cordon_and_drain_node",
            "severity": 2,
            "escalate": True,
        },
    ],
    "INCIDENT:APPLICATION": [
        {
            "service": "graphql-api-router",
            "log_level": "ERROR",
            "error": "Uncaught TypeError: Cannot read properties of undefined (reading 'customer_id') in user resolver",
            "action": "rollback_canary_deployment",
            "severity": 1,
            "escalate": False,
        },
        {
            "service": "frontend-ssr-service",
            "log_level": "ERROR",
            "error": "HTTP 500 Internal Server Error rate spiked to 14% following release v2.14.0 deployment",
            "action": "rollback_canary_deployment",
            "severity": 2,
            "escalate": True,
        },
        {
            "service": "notification-dispatcher",
            "log_level": "WARN",
            "error": "Unhandled promise rejection in push notification webhook delivery loop",
            "action": "restart_application_pods",
            "severity": 0,
            "escalate": False,
        },
    ],
}

ALL_REMEDIATION_ACTIONS: list[str] = [
    "failover_to_replica",
    "restart_db_instance",
    "restart_connection_pooler",
    "kill_blocking_queries",
    "retry_with_exponential_backoff",
    "scale_consumer_replicas",
    "reprocess_failed_offset",
    "restart_stream_processor",
    "rotate_signing_keys",
    "revoke_compromised_token",
    "block_source_ip_range",
    "enable_ddos_under_attack_mode",
    "rate_limit_edge_waf",
    "route_to_backup_gateway",
    "retry_pending_webhooks",
    "halt_invoice_dispatch",
    "recalculate_ledger_balances",
    "cordon_and_drain_node",
    "reboot_host",
    "rollback_canary_deployment",
    "restart_application_pods",
]

DEFAULT_SCORE_CRITERIA: list[str] = ["low", "medium", "urgent", "critical"]


def generate_synthetic_incidents(
    num_samples: int = 1200,
    seed: int = 42,
) -> list[IncidentInstance]:
    """Generate diverse synthetic operational incidents with ground truth targets."""
    rng = random.Random(seed)
    instances: list[IncidentInstance] = []

    curies = list(INCIDENT_TEMPLATES.keys())
    per_curie = max(1, num_samples // len(curies))

    for curie in curies:
        templates = INCIDENT_TEMPLATES[curie]
        for _ in range(per_curie):
            tmpl = rng.choice(templates)
            # Add synthetic noise/jitter to create variety
            incident_id = rng.randint(1000, 99999)
            latency_ms = rng.randint(250, 45000)
            affected_users = rng.randint(1, 100000)

            state = {
                "service": tmpl["service"],
                "incident_id": f"INC-{incident_id}",
                "log_level": tmpl["log_level"],
                "error": tmpl["error"],
                "latency_ms": latency_ms,
                "affected_users": affected_users,
            }

            # Vary the error representation slightly
            if rng.random() < 0.3:
                state["cluster"] = rng.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
            if rng.random() < 0.2:
                state["environment"] = rng.choice(["prod-primary", "prod-us", "prod-eu"])

            instances.append(
                IncidentInstance(
                    state=state,
                    target_curie=curie,
                    target_action=tmpl["action"],
                    severity=tmpl["severity"],
                    escalate=tmpl["escalate"],
                )
            )

    rng.shuffle(instances)
    return instances[:num_samples]
