"""Live end-to-end evaluation of SEVERANCE with the real local models (Ollama).
Single pass over a fixed query set, isolated temp database, every event recorded."""
import json, os, re, sys, tempfile, time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="sev_eval_"))
DB = str(TMP / "eval.db")
os.environ["SEVERANCE_DB_PATH"] = DB  # network monitor + everything else log here, not the app DB
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).parent / "results_live.json"

from contracts import AskRequest, Compartment, Principal
from harness.runner import run_query
from ingest.pdf import load_corpus
from trust.ledger import init_ledger_table, verify as ledger_verify
from trust.reports import init_reports_table
from trust.conversations import init_conversations_table
from trust.network_monitor import init_network_log_table, get_stats, get_recent_events, check_and_record, ExternalConnectionBlocked
from agents.real import RealLlmAgent

for f in (init_ledger_table, init_reports_table, init_conversations_table, init_network_log_table):
    f(DB)
corpus = load_corpus(REPO / "corpus/manifest.json", REPO / "corpus/documents")
principal = Principal(person_id="eval-001", name="Evaluator", job_title="Evaluator", grade="I",
                      compartments=frozenset(Compartment))

E = "emergency_rulebook"
Q = [
    # (id, question, expected_category, expected_sources or check)
    ("C1", "How do the regulations define an off site emergency?", "content", {(E, 3)}),
    ("C2", "What are the responsibilities of the Chief Incident Controller?", "content", {(E, 20), (E, 21), (E, 22)}),
    ("C3", "What are the duties of the Site Incident Controller?", "content", {(E, 20), (E, 21), (E, 23)}),
    ("C4", "Which incidents must be reported as major incidents?", "content", {(E, 34), (E, 49), (E, 50)}),
    ("C5", "What emergency control systems and facilities should an installation have?", "content", {(E, 30)}),
    ("C6", "What information does a transport emergency card contain?", "content", {(E, 56), (E, 58)}),
    ("C7", "What are the overall objectives of an emergency control organization?", "content", {(E, 18)}),
    ("C8", "What should the termination of an emergency concentrate on?", "content", {(E, 35)}),
    ("C9", "What is the responsibility of the Recorder during an emergency?", "content", {(E, 25)}),
    ("C10", "How is individual risk per annum categorized?", "content", {(E, 11)}),
    ("C11", "What are the objectives of preparing an ERDMP?", "content", {(E, 6), (E, 2)}),
    ("C12", "What should the first aid box carried for road transport emergencies contain?", "content", {(E, 53), (E, 54)}),
    ("C13", "Where must personnel assemble during a toxic gas release at CDU-2?", "content", {("mrpl-hse-sop-101", 1)}),
    ("C14", "What minimum water flow rate must deluge systems on crude distillation heaters maintain?", "content", {("mrpl-hse-sop-101", 2)}),
    ("C15", "What did the vigilance inquiry find about the heat exchanger maintenance bids?", "content", {("mrpl-vig-proc-2025", 1)}),
    ("C16", "What does Section 10 of the RTI Act allow regarding severability?", "content", {("mrpl-rti-manual-sec4", 2)}),
    ("O1", "What was MRPL's net profit in financial year 2023-24?", "content", None),
    ("O2", "Who won the cricket world cup in 2011?", "other", None),
    ("K1", "convert 250 bar to psi", "calculation", "3625.9"),
    ("K2", "what is 2500 * 4.5 + 120", "calculation", "11370"),
    ("K3", "how many barrels is 12000 liters", "calculation", "75.4"),
    ("K4", "convert 45 degrees celsius to fahrenheit", "calculation", "113"),
    ("P1", "write python code that prints the first 10 fibonacci numbers", "code", None),
    ("P2", "write python code that checks whether the string 'level' is a palindrome and prints the result", "code", None),
    ("P3", "write python code that computes the factorial of 10 and prints it", "code", None),
    ("P4", "write python code that counts the vowels in the sentence 'emergency response plan' and prints the count", "code", None),
    ("Q1", "what can you do?", "capability", None),
    ("Q2", "which documents can I access?", "capability", None),
]

rows = []
for qid, question, exp_cat, expect in Q:
    agent = RealLlmAgent()  # one agent per request, exactly as api/routes/ask.py::get_agent() does
    events = []
    t0 = time.perf_counter()
    try:
        resp = run_query(AskRequest(question=question, top_k=3), corpus, principal, agent, DB, emit=events.append)
        err = None
    except Exception as exc:
        resp, err = None, repr(exc)
    secs = time.perf_counter() - t0
    cat = next((e.get("category") for e in events if e.get("stage") == "intent" and e.get("status") == "done"), None)
    draft_attempts = sum(1 for e in events if e["stage"] == "drafting" and e["status"] == "start")
    ver = [e for e in events if e["stage"] == "verification" and e["status"] in ("pass", "fail")]
    grounding = [e for e in events if e["stage"] == "grounding" and e["status"] != "start"]
    code_attempts = sum(1 for e in events if e["stage"] == "code" and e["status"] == "start")
    sandbox = [e["status"] for e in events if e["stage"] == "sandbox" and e["status"] != "start"]
    calc = [e for e in events if e["stage"] == "calculation" and e["status"] != "start"]
    row = {"id": qid, "question": question, "expected_category": exp_cat, "routed_category": cat,
           "seconds": round(secs, 1), "error": err}
    if resp is not None:
        cites = [(c.doc_id, c.page) for c in resp.citations]
        row.update({"status": resp.status, "answer": resp.answer, "citations": cites,
                    "effective_tier": resp.effective_label.tier.value})
    row.update({"draft_attempts": draft_attempts, "verification": [v["status"] for v in ver],
                "verification_fail_reasons": [v.get("reason", "")[:160] for v in ver if v["status"] == "fail"],
                "grounding": [{"status": g["status"], "checked": g.get("checked"), "removed": g.get("removed")} for g in grounding],
                "code_attempts": code_attempts, "sandbox": sandbox,
                "calculation": [{"status": c["status"], "tool": c.get("tool"), "reason": c.get("reason")} for c in calc]})
    if isinstance(expect, set) and resp is not None:
        row["expected_sources"] = sorted(expect)
        row["source_hit"] = bool(set(cites) & expect)
    if isinstance(expect, str) and resp is not None:
        row["expected_value"] = expect
        row["value_correct"] = expect.replace(",", "") in resp.answer.replace(",", "")
    rows.append(row)
    print(json.dumps({k: row.get(k) for k in ("id", "routed_category", "status", "seconds", "draft_attempts", "verification", "grounding", "code_attempts", "sandbox", "source_hit", "value_correct", "error")}), flush=True)

# on-demand egress probe through the same guard
try:
    check_and_record("sovereignty-test-probe", "https://example.com/probe")
    probe_blocked = False
except ExternalConnectionBlocked:
    probe_blocked = True

stats = get_stats()
events = get_recent_events(limit=10000)
by_label = {}
for e in events:
    k = (e["process_label"], f'{e["host"]}:{e["port"]}', "allowed" if e["allowed"] else "blocked")
    by_label[str(k)] = by_label.get(str(k), 0) + 1
intact, broken = ledger_verify(DB)
OUT.write_text(json.dumps({"rows": rows, "network_stats": stats, "network_by_label": by_label,
                           "probe_blocked": probe_blocked, "ledger_intact": intact, "ledger_broken_at": broken,
                           "models": {"draft": agent.ollama_model, "fallback": agent.ollama_fallback_model,
                                      "code": agent.ollama_code_model, "vision": agent.ollama_vision_model}},
                          indent=2, default=str))
print("NETWORK", json.dumps(stats), json.dumps(by_label, indent=1))
print("PROBE_BLOCKED", probe_blocked, "LEDGER_INTACT", intact)
