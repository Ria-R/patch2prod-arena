#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def gitlab_payload(status: str) -> dict:
    return {
        "object_kind": "build",
        "project": {"path_with_namespace": "demo/auth-service"},
        "commit": {"id": "9a7cb1f4d8a2"},
        "ref": "main",
        "build_name": "unit-tests",
        "build_stage": "test",
        "build_status": status,
        "object_attributes": {
            "id": 1201,
            "status": status,
            "name": "unit-tests",
            "stage": "test",
            "url": "https://gitlab.example.com/demo/auth-service/-/jobs/1201",
        },
    }


def jenkins_payload(status: str) -> dict:
    return {
        "name": "payment-service",
        "displayName": "integration-tests",
        "url": "https://jenkins.example.com/job/payment-service/212/",
        "build": {
            "number": 212,
            "status": status,
            "branch": "main",
            "commit": "d10ac4e22f0a",
            "stage": "integration-tests",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Replay CI break/fix webhook events to Patch2Prod.")
    p.add_argument("--base", default="http://127.0.0.1:8001", help="Patch2Prod API base URL")
    p.add_argument("--token", default="patch2prod-demo-token", help="Webhook auth token")
    p.add_argument("--delay", type=float, default=1.5, help="Seconds between events")
    args = p.parse_args()

    headers_gitlab = {"Content-Type": "application/json", "X-Gitlab-Token": args.token}
    headers_jenkins = {"Content-Type": "application/json", "X-Patch2Prod-Token": args.token}

    sequence = [
        ("gitlab", "failed"),
        ("jenkins", "running"),
        ("gitlab", "passed"),
        ("jenkins", "passed"),
    ]
    for provider, status in sequence:
        if provider == "gitlab":
            url = f"{args.base}/ci/webhook/gitlab"
            payload = gitlab_payload(status)
            out = post_json(url, payload, headers_gitlab)
        else:
            url = f"{args.base}/ci/webhook/jenkins"
            payload = jenkins_payload(status)
            out = post_json(url, payload, headers_jenkins)
        print(f"{provider}:{status} -> {out.get('ok')} task={out.get('event', {}).get('task_id')}")
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
