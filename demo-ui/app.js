const $ = (id) => document.getElementById(id);
const API_BASE = "http://127.0.0.1:8001";

const state = {
  selectedScenario: "authsdk_mobile_contract_break",
  selectedRun: "trained",
  runs: { baseline: null, trained: null, reference: null },
};

const SCENARIOS = {
  authsdk_mobile_contract_break: {
    title: "Auth SDK Migration",
    service: "auth-service",
    failedJob: "unit-tests",
    category: "dependency upgrade",
    expectedDecision: "block",
    hiddenRisk: "mobile-gateway contract break",
    file: "app/retry.py",
    commit: "c42",
    impacted: ["checkout-service", "mobile-gateway", "fraud-service"],
    owner: "mobile-platform",
  },
  payment_schema_checkout_break: {
    title: "Payment Schema Change",
    service: "payment-service",
    failedJob: "integration-tests",
    category: "schema migration",
    expectedDecision: "ship",
    hiddenRisk: "checkout-service compatibility break",
    file: "app/payment_response.py",
    commit: "p17",
    impacted: ["checkout-service", "fraud-service"],
    owner: "checkout-platform",
  },
};

function pretty(v) { return JSON.stringify(v, null, 2); }

function showError(msg) {
  console.error(msg);
  alert(msg);
}

function clearError() {
  // no-op in simplified demo UI
}

async function api(path, opts = {}) {
  const url = API_BASE + path;
  let res;
  try {
    res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  } catch (_e) {
    res = await fetch(url.replace("://localhost:", "://127.0.0.1:"), { headers: { "Content-Type": "application/json" }, ...opts });
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${pretty(data)}`);
  return data;
}

function renderScenarioCards() {
  const wrap = $("scenarioCards");
  wrap.innerHTML = "";
  Object.entries(SCENARIOS).forEach(([id, s]) => {
    const card = document.createElement("button");
    card.className = `scenario-card ${state.selectedScenario === id ? "active" : ""}`;
    card.innerHTML = `
      <h3>${s.title}</h3>
      <div class="scenario-meta">
        <div>Service: ${s.service}</div>
        <div>Failed job: ${s.failedJob}</div>
        <div>Category: ${s.category}</div>
        <div>Expected decision: <strong>${s.expectedDecision.toUpperCase()}</strong></div>
        <div>Hidden risk: ${s.hiddenRisk}</div>
      </div>
    `;
    card.onclick = () => {
      state.selectedScenario = id;
      renderScenarioCards();
    };
    wrap.appendChild(card);
  });
}

function normalizeAction(a, s) {
  const out = JSON.parse(JSON.stringify(a));
  out.params = out.params || {};
  if (out.action_type === "view_log" && !out.params.job_name) out.params.job_name = s.failedJob;
  if (out.action_type === "view_diff" && !out.params.commit_id) out.params.commit_id = s.commit;
  if (out.action_type === "cat" && !out.params.file_path) out.params.file_path = s.file;
  if (out.action_type === "run_unit_tests" && !out.params.service) out.params.service = s.service;
  if (out.action_type === "run_contract_tests" && !out.params.service) out.params.service = s.impacted[0];
  if (out.action_type === "submit_release_decision") {
    if (!out.params.owner_to_notify && out.params.owner) out.params.owner_to_notify = out.params.owner;
    if (!out.params.owner_to_notify) out.params.owner_to_notify = s.owner;
    delete out.params.owner;
  }
  return out;
}

function plan(kind, sid) {
  const s = SCENARIOS[sid];
  const patch = sid === "authsdk_mobile_contract_break"
    ? { action_type: "replace", params: { file_path: "app/retry.py", search: "build_retry_policy", replace: "create_retry_policy" } }
    : { action_type: "replace", params: { file_path: "app/payment_response.py", search: "return {'status': p.status, 'id': p.id}", replace: "return {'status': p.status, 'payment_status': p.status, 'id': p.id}" } };

  const baseline = [
    { action_type: "view_log", params: {} }, patch,
    { action_type: "run_unit_tests", params: {} },
    { action_type: "submit_release_decision", params: { decision: "ship", reason: "CI is green", owner: s.owner } },
  ];

  const trained = [
    { action_type: "view_log", params: {} },
    { action_type: "view_commit_history", params: {} },
    { action_type: "view_diff", params: {} },
    { action_type: "cat", params: {} },
    patch,
    { action_type: "run_unit_tests", params: {} },
    { action_type: "view_dependency_graph", params: { service: s.service } },
    { action_type: "submit_blast_radius", params: { impacted_services: s.impacted } },
    { action_type: "run_contract_tests", params: { service: s.impacted[0] } },
    ...(sid === "authsdk_mobile_contract_break" ? [{ action_type: "run_contract_tests", params: { service: "mobile-gateway" } }] : []),
    { action_type: "submit_release_decision", params: { decision: s.expectedDecision, reason: "Decision based on downstream validation", owner: s.owner } },
  ];

  const reference = [...trained, { action_type: "view_ownership_map", params: {} }];
  if (kind === "baseline") return baseline;
  if (kind === "reference") return reference;
  return trained;
}

async function run(kind) {
  clearError();
  try {
    const sid = state.selectedScenario;
    const s = SCENARIOS[sid];
    await api("/reset", { method: "POST", body: JSON.stringify({ task_id: sid }) });
    let st = await api("/state");
    const records = [];
    for (const raw of plan(kind, sid)) {
      if (st.done) break;
      const norm = normalizeAction(raw, s);
      const obs = await api("/step", { method: "POST", body: JSON.stringify(norm) });
      st = await api("/state");
      records.push({ raw, norm, obs: obs.last_action_result || {}, reward: Number(obs.reward || 0) });
    }
    state.runs[kind] = { sid, st, records, rb: st.reward_breakdown || {}, total: Number(st.reward_total || 0) };
    state.selectedRun = kind;
    renderAll();
  } catch (e) {
    showError(e.message || String(e));
  }
}

function renderResult(kind, id) {
  const r = state.runs[kind];
  if (!r) { $(id).textContent = "Not run"; return; }
  const d = r.st.release_decision || {};
  $(id).innerHTML = `<div><strong>Decision:</strong> ${(d.decision || "none").toUpperCase()}</div><div><strong>Reward:</strong> ${r.total.toFixed(2)}</div><div><strong>Pipeline:</strong> ${r.st.pipeline_status}</div>`;
}

function updatePicker() {
  [["baseline","pickBaselineBtn"],["trained","pickTrainedBtn"],["reference","pickReferenceBtn"]].forEach(([k,id]) => {
    $(id).classList.toggle("active", state.selectedRun === k);
  });
}

function renderEvidence() {
  const r = state.runs[state.selectedRun];
  if (!r) {
    $("timeline").innerHTML = "Run an agent to view timeline.";
    $("graph").innerHTML = "Run an agent to view dependency graph.";
    $("rewardBreakdown").innerHTML = "Run an agent to view reward breakdown.";
    $("decisionCard").innerHTML = "Run an agent to view final decision.";
    return;
  }
  $("timeline").innerHTML = r.records.map((x,i) => `<div class="row ${x.reward>0.2?"good":x.reward<-0.2?"bad":"warn"}"><div><strong>Step ${i+1}</strong>: ${x.norm.action_type} (reward ${x.reward.toFixed(2)})</div><div class="muted">Raw: ${escapeHtml(JSON.stringify(x.raw))}</div><div class="muted">Normalized: ${escapeHtml(JSON.stringify(x.norm))}</div></div>`).join("");
  const s = SCENARIOS[r.sid];
  const v = r.st.validations || {};
  $("graph").innerHTML = `<div><strong>${s.service}</strong></div><ul>${s.impacted.map(it=>`<li>${it} ${v[`contract:${it}`]==="failed"?"❌":v[`contract:${it}`]==="passed"?"✅":"•"}</li>`).join("")}</ul>`;
  $("rewardBreakdown").innerHTML = Object.entries(r.rb).map(([k,vv])=>`<div><strong>${k}</strong>: ${Number(vv).toFixed(2)}</div>`).join("") || "No reward breakdown.";
  const d = r.st.release_decision || {};
  $("decisionCard").innerHTML = `<div><strong>Final Decision:</strong> ${(d.decision||"none").toUpperCase()}</div><div><strong>Reason:</strong> ${escapeHtml(d.reason||"")}</div><div><strong>Owner:</strong> ${escapeHtml(d.owner_to_notify||d.owner||"")}</div>`;
}

function renderAll() {
  renderResult("baseline", "baselineResult");
  renderResult("trained", "trainedResult");
  renderResult("reference", "referenceResult");
  updatePicker();
  renderEvidence();
}

function escapeHtml(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function setupTabs() {
  document.querySelectorAll(".subtab").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".subtab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".subpanel").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      $(b.dataset.subtab).classList.add("active");
    };
  });
}

async function init() {
  renderScenarioCards();
  setupTabs();
  renderAll();

  $("runBaselineBtn").onclick = () => run("baseline");
  $("runTrainedBtn").onclick = () => run("trained");
  $("runReferenceBtn").onclick = () => run("reference");
  $("pickBaselineBtn").onclick = () => { state.selectedRun = "baseline"; renderAll(); };
  $("pickTrainedBtn").onclick = () => { state.selectedRun = "trained"; renderAll(); };
  $("pickReferenceBtn").onclick = () => { state.selectedRun = "reference"; renderAll(); };
}

init();
