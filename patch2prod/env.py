from __future__ import annotations

import uuid
from collections import defaultdict
from difflib import unified_diff
from typing import Any, Dict, List, Optional

from .models import Action, EnvState, Observation
from .tasks import get_task


AVAILABLE_ACTIONS = [
    "view_log",
    "view_commit_history",
    "view_diff",
    "cat",
    "view_migration_guide",
    "view_security_advisory",
    "replace",
    "run_unit_tests",
    "view_dependency_graph",
    "mark_impacted_service",
    "run_contract_tests",
    "view_ownership_map",
    "submit_causal_change",
    "submit_blast_radius",
    "submit_release_decision",
    "view_reward",
]


class Patch2ProdEnv:
    """OpenEnv-style environment.

    It implements reset(), step(action), and state(). The server exposes these
    over HTTP. This class intentionally keeps deterministic task simulation so
    RL reward is stable and fast during hackathon demos.
    """

    def __init__(self, default_task_id: str = "authsdk_mobile_contract_break", max_steps: int = 18):
        self.default_task_id = default_task_id
        self.max_steps = max_steps
        self.task: Dict[str, Any] = {}
        self.files: Dict[str, str] = {}
        self.original_files: Dict[str, str] = {}
        self._state: Optional[EnvState] = None
        self._reward_breakdown = defaultdict(float)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        self.task = get_task(task_id or self.default_task_id)
        self.files = dict(self.task["files"])
        self.original_files = dict(self.task["files"])
        self._reward_breakdown = defaultdict(float)
        episode_id = str(uuid.uuid4())[:8]
        self._state = EnvState(
            episode_id=episode_id,
            task_id=self.task["task_id"],
            step=0,
            max_steps=self.max_steps,
            done=False,
            pipeline_status="failed",
            changed_files=[],
            current_diff="",
            discovered={
                "viewed_logs": [],
                "viewed_files": [],
                "viewed_commits": False,
                "viewed_dependency_graph": False,
                "marked_impacted": [],
                "submitted_causal_change": None,
                "submitted_blast_radius": None,
                "security_checked": False,
            },
            validations={},
            release_decision=None,
            reward_total=0.0,
            reward_breakdown={},
            action_trace=[],
        )
        return self._obs(
            message="Episode started. Pipeline is red. Diagnose, repair, compute blast radius, validate, and decide release.",
            last_action_result={
                "service": self.task["service"],
                "problem": self.task["problem"],
                "pipeline_status": "failed",
                "failed_job": self.task["failed_job"],
                "log_excerpt": self.task["initial_log"].splitlines()[-2:],
            },
            reward_delta=0.0,
        )

    @property
    def state(self) -> EnvState:
        assert self._state is not None, "Call reset() first"
        return self._state

    def step(self, action: Action) -> Observation:
        assert self._state is not None, "Call reset() first"
        if self._state.done:
            return self._obs("Episode is already done.", {"ignored_action": action.model_dump()}, 0.0)

        self._state.step += 1
        result: Dict[str, Any] = {"action_type": action.action_type, "ok": True}
        reward_delta = -0.02  # small step cost

        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler is None:
            result = {"ok": False, "error": f"Unknown action_type={action.action_type}"}
            reward_delta -= 0.5
            self._add_reward("invalid_action", -0.5)
        else:
            try:
                specific_result, specific_reward = handler(action.params)
                result.update(specific_result)
                reward_delta += specific_reward
            except Exception as exc:  # keep env robust for bad agent outputs
                result = {"ok": False, "error": str(exc), "action_type": action.action_type}
                reward_delta -= 0.3
                self._add_reward("tool_error", -0.3)

        self._add_reward("step_cost", -0.02)
        self._state.reward_total += reward_delta
        self._state.reward_breakdown = dict(self._reward_breakdown)
        self._state.action_trace.append({"step": self._state.step, "action": action.model_dump(), "result": result, "reward_delta": round(reward_delta, 3)})

        if self._state.step >= self.max_steps and not self._state.done:
            self._state.done = True
            result["terminated"] = "max_steps"
            self._add_reward("timeout", -0.5)
            self._state.reward_total -= 0.5
            self._state.reward_breakdown = dict(self._reward_breakdown)

        return self._obs("Action executed.", result, reward_delta)

    # ---------- Tool handlers ----------

    def _handle_view_log(self, params: Dict[str, Any]):
        job = params.get("job_name", self.task["failed_job"])
        self.state.discovered["viewed_logs"].append(job)
        self._add_reward("investigation", 0.1)
        return {"job_name": job, "log": self.task["initial_log"]}, 0.1

    def _handle_view_commit_history(self, params: Dict[str, Any]):
        self.state.discovered["viewed_commits"] = True
        self._add_reward("causal_diagnosis", 0.15)
        return {"recent_changes": self.task["recent_changes"]}, 0.15

    def _handle_view_diff(self, params: Dict[str, Any]):
        commit_id = params.get("commit_id")
        if commit_id:
            if commit_id == self.task["causal_commit"]:
                self._add_reward("causal_diagnosis", 0.2)
                return {"commit_id": commit_id, "diff_summary": self.task["causal_change"]}, 0.2
            return {"commit_id": commit_id, "diff_summary": "No relevant breaking change found."}, -0.05
        return {"current_diff": self._build_diff()}, 0.05

    def _handle_cat(self, params: Dict[str, Any]):
        file_path = params["file_path"]
        if file_path not in self.files:
            return {"ok": False, "error": f"File not found: {file_path}"}, -0.2
        self.state.discovered["viewed_files"].append(file_path)
        reward = 0.08 if file_path in self._relevant_files() else -0.02
        self._add_reward("investigation", reward)
        return {"file_path": file_path, "content": self.files[file_path]}, reward

    def _handle_view_migration_guide(self, params: Dict[str, Any]):
        guide_files = [p for p in self.files if "migration" in p or "docs" in p]
        content = "\n\n".join(f"# {p}\n{self.files[p]}" for p in guide_files)
        self._add_reward("investigation", 0.2)
        return {"migration_guide": content}, 0.2

    def _handle_view_security_advisory(self, params: Dict[str, Any]):
        self.state.discovered["security_checked"] = True
        package = params.get("package")
        advisories = self.task.get("security_advisories", {})
        result = advisories.get(package, advisories)
        self._add_reward("validation", 0.15)
        return {"package": package, "advisory": result}, 0.15

    def _handle_replace(self, params: Dict[str, Any]):
        file_path = params["file_path"]
        search = params["search"]
        replace = params["replace"]
        if file_path not in self.files:
            return {"ok": False, "error": f"File not found: {file_path}"}, -0.3
        if search not in self.files[file_path]:
            return {"ok": False, "error": "Search string not found"}, -0.2

        self.files[file_path] = self.files[file_path].replace(search, replace)
        if file_path not in self.state.changed_files:
            self.state.changed_files.append(file_path)
        self.state.current_diff = self._build_diff()

        reward = 0.0
        if self._matches_expected_replacement(file_path, search, replace):
            reward += 0.8
            self._add_reward("repair", 0.8)
        elif self._matches_optional_safe_replacement(file_path, search, replace):
            reward += 0.6
            self._add_reward("safety_fix", 0.6)
        else:
            reward -= 0.25
            self._add_reward("unrelated_patch", -0.25)
        return {"file_path": file_path, "diff": self.state.current_diff}, reward

    def _handle_run_unit_tests(self, params: Dict[str, Any]):
        service = params.get("service", self.task["service"])
        passed = self._expected_patch_applied()
        self.state.pipeline_status = "passed" if passed else "failed"
        self.state.validations[f"unit:{service}"] = "passed" if passed else "failed"
        reward = 1.4 if passed else -0.25
        self._add_reward("ci_repair", reward)
        return {"service": service, "status": "passed" if passed else "failed", "summary": "12 passed" if passed else self.task["initial_log"]}, reward

    def _handle_view_dependency_graph(self, params: Dict[str, Any]):
        service = params.get("service", self.task["service"])
        self.state.discovered["viewed_dependency_graph"] = True
        self._add_reward("blast_radius", 0.2)
        return {"service": service, "graph": self.task["dependency_graph"]}, 0.2

    def _handle_mark_impacted_service(self, params: Dict[str, Any]):
        service = params["service"]
        reason = params.get("reason", "")
        if service not in self.state.discovered["marked_impacted"]:
            self.state.discovered["marked_impacted"].append(service)
        if service in self.task["impacted_services"]:
            reward = 0.35
            self._add_reward("blast_radius", reward)
        else:
            reward = -0.15
            self._add_reward("blast_radius_false_positive", reward)
        return {"service": service, "reason": reason, "marked_impacted": self.state.discovered["marked_impacted"]}, reward

    def _handle_run_contract_tests(self, params: Dict[str, Any]):
        service = params["service"]
        after_optional = self._optional_safe_patch_applied()
        table = self.task["contract_results_after_optional_fix"] if after_optional else self.task["contract_results_before_optional_fix"]
        status = table.get(service, "not_applicable")
        self.state.validations[f"contract:{service}"] = status
        reward = 0.0
        if service in self.task["impacted_services"]:
            reward += 0.25
            self._add_reward("targeted_validation", 0.25)
        else:
            reward -= 0.08
            self._add_reward("validation_cost", -0.08)
        if status == "failed":
            reward += 0.3  # finding real downstream risk is good
            self._add_reward("risk_discovery", 0.3)
        elif status == "passed" and service in self.task["impacted_services"]:
            reward += 0.1
            self._add_reward("validation", 0.1)
        return {"service": service, "contract_status": status}, reward

    def _handle_view_ownership_map(self, params: Dict[str, Any]):
        self._add_reward("release_ops", 0.1)
        return {"ownership": self.task["ownership"]}, 0.1

    def _handle_submit_causal_change(self, params: Dict[str, Any]):
        commit = params.get("commit")
        summary = params.get("summary", "")
        self.state.discovered["submitted_causal_change"] = {"commit": commit, "summary": summary}
        reward = 0.8 if commit == self.task["causal_commit"] else -0.4
        self._add_reward("causal_diagnosis", reward)
        return {"correct": commit == self.task["causal_commit"], "expected": self.task["causal_commit"]}, reward

    def _handle_submit_blast_radius(self, params: Dict[str, Any]):
        predicted = set(params.get("impacted_services", []))
        expected = set(self.task["impacted_services"])
        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        reward = f1 * 1.0 - fp * 0.15 - fn * 0.25
        self.state.discovered["submitted_blast_radius"] = sorted(predicted)
        self._add_reward("blast_radius_f1", reward)
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "expected": sorted(expected)}, reward

    def _handle_submit_release_decision(self, params: Dict[str, Any]):
        decision = params.get("decision")
        owner = params.get("owner_to_notify") or params.get("required_owner")
        reason = params.get("reason", "")
        after_optional = self._optional_safe_patch_applied()
        correct_decision = self.task["correct_decision_after_optional_fix"] if after_optional else self.task["correct_decision_before_optional_fix"]

        reward = 0.0
        if decision == correct_decision:
            reward += 1.6
            self._add_reward("release_decision", 1.6)
        else:
            reward -= 1.2
            self._add_reward("bad_release_decision", -1.2)
        if owner == self.task["correct_owner"] or (decision == "ship" and correct_decision == "ship"):
            reward += 0.35
            self._add_reward("owner_escalation", 0.35)
        elif decision in {"block", "canary", "request_owner_approval"}:
            reward -= 0.2
            self._add_reward("wrong_owner", -0.2)

        # Bonus for evidence-backed decision.
        if "contract" in reason.lower() or "downstream" in reason.lower() or "mobile" in reason.lower():
            reward += 0.25
            self._add_reward("evidence", 0.25)

        self.state.release_decision = params
        self.state.done = True
        return {"correct_decision": correct_decision, "submitted": params, "episode_complete": True}, reward

    def _handle_view_reward(self, params: Dict[str, Any]):
        return {"reward_total": self.state.reward_total, "reward_breakdown": dict(self._reward_breakdown)}, 0.0

    # ---------- Helpers ----------

    def _obs(self, message: str, last_action_result: Dict[str, Any], reward_delta: float) -> Observation:
        state = self.state
        visible = {
            "service": self.task.get("service"),
            "pipeline_status": state.pipeline_status,
            "failed_job": self.task.get("failed_job"),
            "changed_files": state.changed_files,
            "current_diff": state.current_diff,
            "discovered": state.discovered,
            "validations": state.validations,
            "release_decision": state.release_decision,
            "steps_remaining": state.max_steps - state.step,
        }
        return Observation(
            episode_id=state.episode_id,
            task_id=state.task_id,
            step=state.step,
            done=state.done,
            message=message,
            visible_state=visible,
            last_action_result=last_action_result,
            reward=round(reward_delta, 3),
            reward_breakdown=dict(self._reward_breakdown),
            available_actions=AVAILABLE_ACTIONS,
        )

    def _add_reward(self, key: str, value: float) -> None:
        self._reward_breakdown[key] += value

    def _build_diff(self) -> str:
        chunks: List[str] = []
        for path, current in self.files.items():
            original = self.original_files.get(path, "")
            if original != current:
                chunks.extend(
                    unified_diff(
                        original.splitlines(),
                        current.splitlines(),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                        lineterm="",
                    )
                )
        return "\n".join(chunks)

    def _relevant_files(self) -> set[str]:
        files = {r["file_path"] for r in self.task.get("expected_replacements", [])}
        optional = self.task.get("optional_safe_replacement")
        if optional:
            files.add(optional["file_path"])
        files.update(p for p in self.files if "docs" in p or "contract" in p or "requirements" in p)
        return files

    def _matches_expected_replacement(self, file_path: str, search: str, replace: str) -> bool:
        return any(r["file_path"] == file_path and r["search"] == search and r["replace"] == replace for r in self.task.get("expected_replacements", []))

    def _matches_optional_safe_replacement(self, file_path: str, search: str, replace: str) -> bool:
        r = self.task.get("optional_safe_replacement")
        return bool(r and r["file_path"] == file_path and r["search"] == search and r["replace"] == replace)

    def _expected_patch_applied(self) -> bool:
        for r in self.task.get("expected_replacements", []):
            if r["search"] in self.files[r["file_path"]]:
                return False
            if r["replace"] not in self.files[r["file_path"]]:
                return False
        return True

    def _optional_safe_patch_applied(self) -> bool:
        r = self.task.get("optional_safe_replacement")
        if not r:
            return True
        return r["replace"] in self.files.get(r["file_path"], "")
