from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

TASKS: Dict[str, Dict[str, Any]] = {
    "authsdk_mobile_contract_break": {
        "task_id": "authsdk_mobile_contract_break",
        "title": "Auth SDK upgrade breaks mobile token contract",
        "service": "auth-service",
        "problem": "auth-service CI fails after authsdk v2 upgrade. Local fix can turn CI green, but token expiry contract may break mobile-gateway.",
        "failed_job": "unit-tests",
        "initial_log": """ImportError while importing test module tests/test_retry.py\nfrom app.retry import policy\napp/retry.py:1: in <module>\n    from authsdk.helpers import build_retry_policy\nE   ImportError: cannot import name 'build_retry_policy' from 'authsdk.helpers'\n""",
        "recent_changes": [
            {"commit": "c41", "summary": "Refactor retry policy tests", "status": "passing"},
            {"commit": "c42", "summary": "Upgrade authsdk from 1.8.0 to 2.0.0", "status": "failing"},
        ],
        "causal_commit": "c42",
        "causal_change": "authsdk upgraded from 1.8.0 to 2.0.0",
        "files": {
            "requirements.txt": "authsdk==2.0.0\npytest==8.0.0\n",
            "app/retry.py": "from authsdk.helpers import build_retry_policy\n\npolicy = build_retry_policy(max_attempts=3)\n",
            "docs/authsdk_migration.md": "authsdk v2 migration: build_retry_policy was renamed to create_retry_policy. Token expiry is now serialized as ISO-8601 by default unless compatibility_mode='seconds' is set.",
            "contracts/mobile-gateway.json": '{"expects": {"token_expiry": "integer_seconds"}}',
            "contracts/checkout-service.json": '{"expects": {"token": "opaque_string"}}',
        },
        "expected_replacements": [
            {
                "file_path": "app/retry.py",
                "search": "build_retry_policy",
                "replace": "create_retry_policy",
                "reason": "renamed helper in authsdk v2",
            }
        ],
        "optional_safe_replacement": {
            "file_path": "app/retry.py",
            "search": "create_retry_policy(max_attempts=3)",
            "replace": "create_retry_policy(max_attempts=3, compatibility_mode='seconds')",
            "reason": "preserve mobile token_expiry contract",
        },
        "dependency_graph": {
            "auth-service": ["checkout-service", "mobile-gateway", "fraud-service", "notification-service"],
            "checkout-service": ["payment-service"],
            "mobile-gateway": [],
            "fraud-service": [],
            "notification-service": [],
        },
        "impacted_services": ["checkout-service", "mobile-gateway", "fraud-service"],
        "not_impacted_services": ["notification-service"],
        "contract_results_before_optional_fix": {
            "checkout-service": "passed",
            "fraud-service": "passed",
            "mobile-gateway": "failed",
            "notification-service": "passed",
        },
        "contract_results_after_optional_fix": {
            "checkout-service": "passed",
            "fraud-service": "passed",
            "mobile-gateway": "passed",
            "notification-service": "passed",
        },
        "ownership": {
            "auth-service": "identity-platform",
            "checkout-service": "payments-platform",
            "mobile-gateway": "mobile-platform",
            "fraud-service": "risk-platform",
            "notification-service": "comms-platform",
        },
        "security_advisories": {
            "authsdk==1.8.0": "HIGH: token validation bypass CVE-SIM-2026-001",
            "authsdk==2.0.0": "clean",
        },
        "correct_decision_before_optional_fix": "block",
        "correct_decision_after_optional_fix": "canary",
        "correct_owner": "mobile-platform",
        "release_policy": "Ship only if local CI passes, security scan is clean, and all impacted downstream contract tests pass. If downstream contract fails, block or canary only after compatibility fix and owner review.",
    },
    "payment_schema_checkout_break": {
        "task_id": "payment_schema_checkout_break",
        "title": "Payment schema migration breaks checkout contract",
        "service": "payment-service",
        "problem": "payment-service tests fail after schema rename. Local fix can pass, but checkout-service still expects old field.",
        "failed_job": "integration-tests",
        "initial_log": """AssertionError: expected field 'payment_status', got 'status'\nFAILED tests/test_payment_response.py::test_payment_response_contract\n""",
        "recent_changes": [
            {"commit": "p17", "summary": "Rename payment_status to status", "status": "failing"},
        ],
        "causal_commit": "p17",
        "causal_change": "payment_status renamed to status",
        "files": {
            "app/payment_response.py": "def serialize_payment(p):\n    return {'status': p.status, 'id': p.id}\n",
            "docs/schema_migration.md": "During transition, payment-service must emit both payment_status and status for one release window.",
            "contracts/checkout-service.json": '{"expects": {"payment_status": "string"}}',
        },
        "expected_replacements": [
            {
                "file_path": "app/payment_response.py",
                "search": "return {'status': p.status, 'id': p.id}",
                "replace": "return {'status': p.status, 'payment_status': p.status, 'id': p.id}",
                "reason": "dual-write response field during migration window",
            }
        ],
        "optional_safe_replacement": None,
        "dependency_graph": {
            "payment-service": ["checkout-service", "fraud-service", "analytics-batch"],
            "checkout-service": [],
            "fraud-service": [],
            "analytics-batch": [],
        },
        "impacted_services": ["checkout-service", "fraud-service"],
        "not_impacted_services": ["analytics-batch"],
        "contract_results_before_optional_fix": {
            "checkout-service": "passed",
            "fraud-service": "passed",
            "analytics-batch": "passed",
        },
        "contract_results_after_optional_fix": {
            "checkout-service": "passed",
            "fraud-service": "passed",
            "analytics-batch": "passed",
        },
        "ownership": {
            "payment-service": "payments-platform",
            "checkout-service": "checkout-platform",
            "fraud-service": "risk-platform",
            "analytics-batch": "data-platform",
        },
        "security_advisories": {},
        "correct_decision_before_optional_fix": "ship",
        "correct_decision_after_optional_fix": "ship",
        "correct_owner": "checkout-platform",
        "release_policy": "Schema changes must preserve backward compatibility for one release window and pass impacted contracts.",
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id={task_id}. Available: {list(TASKS)}")
    return deepcopy(TASKS[task_id])


def list_tasks() -> list[Dict[str, str]]:
    out: list[Dict[str, str]] = []
    for task_id, task in TASKS.items():
        out.append(
            {
                "task_id": task_id,
                "title": task.get("title", task_id),
                "service": task.get("service", "unknown"),
                "failed_job": task.get("failed_job", "unknown"),
            }
        )
    return out


def find_task_for_ci_event(service: str | None = None, job_name: str | None = None) -> str | None:
    """Best-effort mapping from CI event metadata to a known arena task_id."""
    service_norm = (service or "").strip().lower()
    job_norm = (job_name or "").strip().lower()
    for task_id, task in TASKS.items():
        task_service = str(task.get("service", "")).strip().lower()
        task_job = str(task.get("failed_job", "")).strip().lower()
        if service_norm and task_service and service_norm == task_service:
            return task_id
        if job_norm and task_job and job_norm == task_job:
            return task_id
    return None
