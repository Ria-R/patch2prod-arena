from __future__ import annotations

import json
import requests

BASE = "http://localhost:8000"


def post(path, payload):
    r = requests.post(BASE + path, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


print(json.dumps(post("/reset", {"task_id": "authsdk_mobile_contract_break"}), indent=2)[:1000])
for action in [
    {"action_type": "view_log", "params": {"job_name": "unit-tests"}},
    {"action_type": "view_migration_guide", "params": {"package": "authsdk"}},
    {"action_type": "replace", "params": {"file_path": "app/retry.py", "search": "build_retry_policy", "replace": "create_retry_policy"}},
    {"action_type": "run_unit_tests", "params": {"service": "auth-service"}},
]:
    print(json.dumps(post("/step", action), indent=2)[:1200])
print(requests.get(BASE + "/state", timeout=10).json())
