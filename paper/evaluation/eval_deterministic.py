"""Deterministic evaluation of SEVERANCE's trust components (no LLM calls).

A. Citation verifier: genuine vs adversarially perturbed quotes, plus cosmetic variants.
B. Two-axis access gate: exhaustive check against an independent reference predicate,
   plus a retrieval-layer leak check.
C. Hash-chained ledger: tamper detection and localisation.
D. Sandbox containment micro-tests.
"""
import itertools, json, os, random, re, shutil, sqlite3, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).parent / "results_deterministic.json"

from contracts import Citation, Compartment, Label, Principal, Tier
from harness.verify import check
from harness.retrieve import retrieve
from ingest.pdf import load_corpus
from trust import labels
from trust import ledger

rng = random.Random(20260924)
corpus = load_corpus(REPO / "corpus/manifest.json", REPO / "corpus/documents")
results = {"corpus_passages": len(corpus), "corpus_chars": sum(len(p.text) for p in corpus)}

# ---------------------------------------------------------------- A. verifier
def words_of(text):
    return text.split()

pages_by_doc = {}
for p in corpus:
    pages_by_doc.setdefault(p.doc_id, []).append(p)

vocab = sorted({w for p in corpus for w in re.findall(r"[A-Za-z]{4,}", p.text)})

def genuine_quotes(p, k=10):
    ws = words_of(p.text)
    out = []
    tries = 0
    while len(out) < k and tries < 200:
        tries += 1
        if len(ws) < 8:
            break
        n = rng.randint(6, 25)
        if n > len(ws):
            n = len(ws)
        s = rng.randint(0, len(ws) - n)
        q = " ".join(ws[s:s + n])
        if 20 <= len(q) <= 300 and len(q.split()) >= 5 and q not in out:
            out.append(q)
    return out

def accepted(quote, doc_id, page, passages):
    return check([Citation(doc_id=doc_id, page=page, quote=quote)], passages) is None

def exact_only_accepts(quote, text):
    return quote in text

cats = {k: {"n": 0, "accepted": 0} for k in [
    "genuine", "word_substitution", "word_deletion", "adjacent_swap", "number_change",
    "negation_insertion", "wrong_page", "wrong_document", "fabricated_shuffle"]}
cosmetic = {k: {"n": 0, "accepted": 0, "exact_only_accepted": 0} for k in [
    "whitespace_linebreaks", "lowercased", "typographic_quotes_dashes", "footnote_marker_removed"]}

for p in corpus:
    passages = pages_by_doc[p.doc_id]  # verifier sees all pages of this doc, like a multi-page prompt
    other_docs = [d for d in pages_by_doc if d != p.doc_id]
    for q in genuine_quotes(p):
        ws = q.split()
        def rec(cat, quote, doc_id=p.doc_id, page=p.page, pas=passages):
            if not (20 <= len(quote) <= 300 and len(quote.split()) >= 5):
                return
            cats[cat]["n"] += 1
            cats[cat]["accepted"] += accepted(quote, doc_id, page, pas)
        rec("genuine", q)
        # substitution
        idxs = [i for i, w in enumerate(ws) if re.fullmatch(r"[A-Za-z]{4,}[.,;:]?", w)]
        if idxs:
            i = rng.choice(idxs)
            core = re.sub(r"[.,;:]$", "", ws[i])
            repl = rng.choice(vocab)
            while repl.lower() == core.lower():
                repl = rng.choice(vocab)
            w2 = ws[:]; w2[i] = ws[i].replace(core, repl)
            rec("word_substitution", " ".join(w2))
        if len(ws) >= 7:
            i = rng.randint(1, len(ws) - 2)
            rec("word_deletion", " ".join(ws[:i] + ws[i + 1:]))
            j = rng.randint(0, len(ws) - 2)
            if ws[j].lower() != ws[j + 1].lower():
                w2 = ws[:]; w2[j], w2[j + 1] = w2[j + 1], w2[j]
                rec("adjacent_swap", " ".join(w2))
        m = re.search(r"\d+", q)
        if m:
            nq = q[:m.start()] + str(int(m.group()) + 7) + q[m.end():]
            rec("number_change", nq)
        m = re.search(r"\b(is|are|shall|must|should|will|may|can)\b", q)
        if m:
            rec("negation_insertion", q[:m.end()] + " not" + q[m.end():])
        others = [x for x in passages if x.page != p.page]
        if others:
            rec("wrong_page", q, page=rng.choice(others).page)
        if other_docs:
            od = rng.choice(other_docs)
            op = rng.choice(pages_by_doc[od])
            rec("wrong_document", q, doc_id=od, page=op.page, pas=passages + pages_by_doc[od])
        w2 = ws[:]
        rng.shuffle(w2)
        if w2 != ws:
            rec("fabricated_shuffle", " ".join(w2))
        # cosmetic variants
        def crec(cat, quote):
            if not (20 <= len(quote) <= 300 and len(quote.split()) >= 5) or quote == q:
                return
            cosmetic[cat]["n"] += 1
            cosmetic[cat]["accepted"] += accepted(quote, p.doc_id, p.page, passages)
            cosmetic[cat]["exact_only_accepted"] += exact_only_accepts(quote, p.text)
        if len(ws) >= 6:
            k = len(ws) // 2
            crec("whitespace_linebreaks", " ".join(ws[:k]) + "\n  " + " ".join(ws[k:]))
        crec("lowercased", q.lower())
        if any(c in q for c in "'\"-"):
            crec("typographic_quotes_dashes", q.replace("'", "’").replace('"', "“").replace("-", "–"))
        if re.search(r"\d+\[", q):
            crec("footnote_marker_removed", re.sub(r"\d+\[", "", q).replace("]", ""))

results["verifier"] = {"adversarial": cats, "cosmetic": cosmetic}

# ---------------------------------------------------------------- B. access gate
def reference_can_read(grade, comps, tier, lcomps):
    rank = labels.RANK.get(grade)
    if rank is None:
        return False
    floor = {"public": 0, "internal": 1, "confidential": 7, "secret": 11}[tier]
    return rank >= floor and set(lcomps) <= set(comps)

all_comps = list(Compartment)
subsets = [frozenset(s) for r in range(len(all_comps) + 1) for s in itertools.combinations(all_comps, r)]
grades = list(labels.RANK.keys())
checked = mismatches = rank_ok_but_comp_denied = 0
for g in grades:
    for pc in subsets:
        pr = Principal(person_id="x", name="x", job_title="x", grade=g, compartments=pc)
        for t in Tier:
            for lc in subsets:
                got = labels.can_read(pr, Label(tier=t, compartments=lc))
                exp = reference_can_read(g, pc, t.value, lc)
                checked += 1
                mismatches += got != exp
                if labels.RANK[g] >= {"public": 0, "internal": 1, "confidential": 7, "secret": 11}[t.value] and not lc <= pc:
                    rank_ok_but_comp_denied += (got is False)
# fail-closed probes
try:
    Principal(person_id="x", name="x", job_title="x", grade="Z9", compartments=frozenset(all_comps))
    schema_rejects_unknown_grade = False
except Exception:
    schema_rejects_unknown_grade = True
bad = Principal.model_construct(person_id="x", name="x", job_title="x", grade="Z9", compartments=frozenset(all_comps), is_admin=False)
fail_closed = [labels.can_read(bad, Label(tier=t, compartments=frozenset())) for t in Tier]
fail_closed_non_principal = labels.can_read("not-a-principal", Label(tier=Tier.PUBLIC, compartments=frozenset()))
# headline inversion
gm = Principal(person_id="gm", name="GM", job_title="GM", grade="F", compartments=frozenset({Compartment("commercial")}))
am = Principal(person_id="am", name="AM", job_title="AM", grade="A", compartments=frozenset({Compartment("vigilance")}))
vig = Label(tier=Tier.CONFIDENTIAL, compartments=frozenset({Compartment("vigilance")}))

# retrieval leak check
queries = ["emergency response plan", "fire deluge water flow", "vendor bids heat exchanger", "severability section 10",
           "assembly point toxic gas", "chief incident controller duties", "incident reporting format",
           "procurement tenders integrity pact", "road tanker leakage", "mutual aid arrangements",
           "evacuation planning", "hazard identification", "Miniratna public sector", "transport emergency card",
           "termination of emergency", "risk per annum"]
retrieval_calls = leaks = returned_total = denials_total = 0
for g in ["S1", "JM3", "A", "C", "F", "I"]:
    for pc in subsets:
        pr = Principal(person_id="p", name="p", job_title="p", grade=g, compartments=pc)
        for q in queries:
            allowed, denials = retrieve(q, corpus, pr, top_k=5)
            retrieval_calls += 1
            returned_total += len(allowed)
            denials_total += len(denials)
            for a in allowed:
                if not labels.can_read(pr, a.label):
                    leaks += 1
            for d in denials:
                if hasattr(d, "text") or any(d.doc_id == a.doc_id for a in allowed if not labels.can_read(pr, a.label)):
                    leaks += 1
results["access_gate"] = {
    "grades": len(grades), "compartment_subsets": len(subsets), "pairs_checked": checked,
    "mismatches_vs_reference": mismatches, "rank_sufficient_but_compartment_denied": rank_ok_but_comp_denied,
    "schema_rejects_unknown_grade": schema_rejects_unknown_grade,
    "unknown_grade_gate_results_per_tier": fail_closed,
    "non_principal_gate_result": fail_closed_non_principal,
    "inversion": {"GM_F_commercial_reads_vigilance_doc": labels.can_read(gm, vig),
                   "AM_A_vigilance_reads_vigilance_doc": labels.can_read(am, vig)},
    "retrieval_calls": retrieval_calls, "passages_returned": returned_total,
    "denials_recorded": denials_total, "leaks": leaks,
}

# ---------------------------------------------------------------- C. ledger
tmp = Path(tempfile.mkdtemp())
base = tmp / "ledger.db"
N = 500
ledger.init_ledger_table(str(base))
for i in range(N):
    ledger.log(str(base), actor=f"user-{i % 17}", action=rng.choice(["ASK_ANSWERED", "ASK_ABSTAINED", "GRANT_COMPARTMENT", "SET_GRADE"]),
               details={"i": i, "q": f"question {i}"})
t0 = time.perf_counter(); intact, _ = ledger.verify(str(base)); verify_ms = (time.perf_counter() - t0) * 1000
naive = {"trials": 0, "detected": 0, "localised": 0}
recompute = {"trials": 0, "detected": 0, "localised_next_row": 0}
tail = {"trials": 0, "detected": 0}
for r in rng.sample(range(1, N), 100):
    db = tmp / f"t{r}.db"; shutil.copy(base, db)
    ledger.tamper(str(db), r, "ASK_ANSWERED_FORGED", {"forged": True})
    ok, at = ledger.verify(str(db))
    naive["trials"] += 1; naive["detected"] += (not ok); naive["localised"] += (at == r)
    db.unlink()
    # stronger attacker: edits row r AND recomputes its own hash (but not later rows)
    db = tmp / f"r{r}.db"; shutil.copy(base, db)
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM ledger WHERE row_id=?", (r,)).fetchone()
    details = {"forged": True}
    h = ledger.compute_payload_hash(row["timestamp"], row["actor"], "ASK_ANSWERED_FORGED", details, row["prev_hash"])
    con.execute("UPDATE ledger SET action=?, details=?, hash=? WHERE row_id=?",
                ("ASK_ANSWERED_FORGED", json.dumps(details, sort_keys=True), h, r))
    con.commit(); con.close()
    ok, at = ledger.verify(str(db))
    recompute["trials"] += 1; recompute["detected"] += (not ok); recompute["localised_next_row"] += (at == r + 1)
    db.unlink()
# tail rewrite: attacker recomputes the LAST row (no successor to break) -> known limitation
db = tmp / "tail.db"; shutil.copy(base, db)
con = sqlite3.connect(db); con.row_factory = sqlite3.Row
row = con.execute("SELECT * FROM ledger WHERE row_id=?", (N,)).fetchone()
h = ledger.compute_payload_hash(row["timestamp"], row["actor"], "FORGED", {"f": 1}, row["prev_hash"])
con.execute("UPDATE ledger SET action=?, details=?, hash=? WHERE row_id=?", ("FORGED", json.dumps({"f": 1}, sort_keys=True), h, N))
con.commit(); con.close()
ok, _ = ledger.verify(str(db)); tail["trials"] = 1; tail["detected"] = int(not ok)
results["ledger"] = {"rows": N, "baseline_intact": intact, "verify_ms_500_rows": round(verify_ms, 2),
                     "naive_edit": naive, "edit_with_hash_recompute": recompute, "last_row_rewrite": tail}
shutil.rmtree(tmp)

# ---------------------------------------------------------------- D. sandbox
from tools.sandbox import run_python
sbdb = str(Path(tempfile.mkdtemp()) / "sb.db")
cases = {
    "normal_program": "print(sum(range(10)))",
    "infinite_loop": "while True:\n    pass",
    "network_urllib": "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)\nprint('REACHED')",
    "network_raw_socket": "import socket\ns=socket.socket()\ns.settimeout(3)\ns.connect(('93.184.216.34',80))\nprint('REACHED')",
    "write_outside_workdir": "open('/tmp/severance_sandbox_escape_test.txt','w').write('x')\nprint('WROTE')",
}
sb = {}
for name, code in cases.items():
    if os.path.exists("/tmp/severance_sandbox_escape_test.txt"):
        os.remove("/tmp/severance_sandbox_escape_test.txt")
    t0 = time.perf_counter()
    r = run_python(code, timeout=5, db_path=sbdb)
    sb[name] = {"success": r["success"], "timed_out": r["timed_out"], "reached": ("REACHED" in r["stdout"]) or ("WROTE" in r["stdout"]),
                "file_exists_after": os.path.exists("/tmp/severance_sandbox_escape_test.txt") if name == "write_outside_workdir" else None,
                "stdout": r["stdout"][:80], "stderr_tail": r["stderr"][-160:], "enforcement": r.get("network_enforcement"),
                "network_attempts": r.get("network_attempts"), "seconds": round(time.perf_counter() - t0, 2)}
results["sandbox"] = sb

OUT.write_text(json.dumps(results, indent=2, default=str))
print(json.dumps(results, indent=2, default=str))
