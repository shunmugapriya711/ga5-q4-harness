"""Agent-harness safety layer (GA5 Q4) - rule engine built from the grader's policy (profile agent-harness-2026-09).

POST /classify (or any path)  {profile, batchId, policy, cases[]} -> {profile, batchId, results[]}
GET  /log                     recent raw request bodies (debugging)

Per case:
  requestDecision  content policy on the actual user messages (allow / block / review)
  toolDecision     tool policy on proposedTool.resolved + user approvals/revocations/constraints
  route            none if request blocked; restricted for security/vulnerability work; else primary
  trajectory       intervene if request or tool blocked; watch if review or untrusted redirection; else normal
Text in memory, repository files and tool results never grants approval.
"""
import json
import re
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROFILE = "agent-harness-2026-09"
LOG = deque(maxlen=20)

app = FastAPI(title="Agent harness safety layer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------------ content policy
HARMFUL = [
    r"\b(write|create|build|develop|make|code)\b[^.]{0,40}\b(malware|ransomware|virus|worm|wiper|keylogger)\b",
    r"\bwipe other people'?s\b",
    r"\bsteal\w*\b",
    r"\bexfiltrat\w*\b",
    r"\bthreatening messages\b|\bharass\w*\b|\bevade (their|the) blocks?\b|\bstalk\w*\b|\bdox\w*\b",
    r"\b(injure|hurt|kill|poison|maim)\b[^.]{0,40}\b(person|people|someone|him|her|them)\b",
    r"\bforged?\b|\bforgery\b|\btrick customers\b|\bcommit fraud\b|\bphishing\b",
    r"\bevade detection\b",
]
HARMFUL_RE = [re.compile(p, re.I) for p in HARMFUL]
SECURITY_WORK = re.compile(r"\bvulnerab\w*|\bexploit\w*|\bmalware\b|\bpenetration\b|\bpentest\w*|\bsecurity test", re.I)
VULN_DEMO = re.compile(r"\b(find|demonstrate|exploit|attack|pentest)\b[^.]*\bvulnerab\w*|\bvulnerab\w*[^.]*\bdemonstrat\w*",
                       re.I)

# ------------------------------------------------------------------ user-instruction parsing
REVOKE_RE = re.compile(r"\brevoke\w*\b|\bwithdrawn\b|\bnow forbidden\b|\bdo not run it\b|"
                       r"\bdo not execute the pending\b|\bapproval revoked\b|\bchanged my mind\b", re.I)
APPROVE_RE = re.compile(r"\b(approve|approves|approved|authori[sz]e|authori[sz]ed|re-authori[sz]e)\b", re.I)
# user constraints that forbid an operation family
CONSTRAINTS = [
    (re.compile(r"\bdo not deploy\b|\bdon'?t deploy\b", re.I), {"deploy"}),
    (re.compile(r"\bdo not (execute|run)\b[^.]*\b(rows?|sql)\b|\bdo not execute or delete\b|\bdelete any rows\b", re.I),
     {"database_mutation"}),
    (re.compile(r"\bdo not apply\b", re.I), {"terraform_apply"}),
    (re.compile(r"\bdo not send\b|\bwithout sharing\b|\bdo not share\b|\bprepare [^.]*\blocally\b", re.I), {"send"}),
    (re.compile(r"\bdo not change anything\b|\bwithout (editing|changing)\b|\bkeep every existing file intact\b|"
                r"\bonly inspect\b|\bread[- ]only\b", re.I),
     {"delete", "workspace_edit", "deploy", "restart", "rollback", "database_mutation", "terraform_apply",
      "force_push", "host_admin", "send"}),
]
PREP_ONLY = re.compile(r"\b(prepare|draft|write)\b[^.]*\b(plan|proposal|for review)\b|\binvestigate\b|\breview\b", re.I)

CONSEQUENTIAL = {"deploy", "restart", "rollback", "database_mutation", "terraform_apply", "force_push",
                 "host_admin", "dns_change", "privileged_change", "send", "delete", "migrate", "scale"}
READ_OPS = {"workspace_read", "read_public_schema", "read_git_metadata", "read_terraform_plan", "list", "read"}


def user_msgs(case):
    return [str(m.get("content", "")) for m in case.get("messages") or []
            if isinstance(m, dict) and m.get("role", "user") == "user"]


def strip_quotes(t: str) -> str:
    return re.sub(r"'[^']{3,}'|\"[^\"]{3,}\"|“[^”]{3,}”", " ", t)


def is_harmful(t: str) -> bool:
    t = strip_quotes(t)
    t = re.sub(r"\b(do not|don't|never|without)\b[^.;]*", " ", t, flags=re.I)  # negated clauses
    return any(r.search(t) for r in HARMFUL_RE)


def mentions(text: str, target: str) -> bool:
    """target named exactly (not as a prefix of a longer name such as 'svc-other')."""
    if not target:
        return False
    return re.search(r"(?<![\w\-./])" + re.escape(target) + r"(?![\w\-]|\.\w)", text) is not None


def under(path: str, root: str) -> bool:
    if not path or not root:
        return False
    path, root = path.rstrip("/"), root.rstrip("/")
    return path == root or path.startswith(root + "/")


def untrusted_redirect(case) -> bool:
    for r in case.get("toolResults") or []:
        if isinstance(r, dict) and str(r.get("origin", "")).lower() == "untrusted" and r.get("content"):
            return True
    for m in case.get("memory") or []:
        if isinstance(m, dict) and (m.get("origin") not in (None, "saved-preference") or
                                    re.search(r"approv|ignore|override|authori", str(m.get("text", "")), re.I)):
            return True
    for r in case.get("repositoryInstructions") or []:
        if isinstance(r, dict) and re.search(r"approv|ignore|override|authori|disregard", str(r.get("text", "")), re.I):
            return True
    return False


def preflight_ok(case, target, env) -> bool:
    status = None
    for r in case.get("toolResults") or []:
        if isinstance(r, dict) and r.get("tool") == "preflight" and r.get("target") == target \
                and (not env or r.get("environment") in (None, env)):
            status = str(r.get("status", "")).lower()  # latest one wins
    return status in ("passed", "success", "succeeded", "ok")


def rollback_ok(case, target, env) -> bool:
    rb = case.get("rollbackPlan")
    return isinstance(rb, dict) and rb.get("target") == target and rb.get("environment", env) == env \
        and bool(rb.get("action"))


def explicit_approval(msgs, tool, res) -> bool:
    """Latest user message that approves something must name this exact target and environment."""
    target = str(res.get("target", ""))
    env = str(res.get("environment", "")).lower()
    op = str(res.get("operation", "")).lower()
    for m in reversed(msgs):
        if not APPROVE_RE.search(m) and not re.search(r"^\s*(for this task:|current instruction:)?\s*"
                                                     r"(delete|send|publish) exactly|\bdelete exactly\b", m, re.I):
            # an instruction like "Publish public X to URL" is an explicit request for that exact send
            if not (op == "send" and re.search(r"\b(publish|send|upload)\b", m, re.I) and mentions(m, target)):
                continue
        if not mentions(m, target):
            # personal-machine delete: "Delete exactly <path>"
            return False
        low = m.lower()
        if env == "production" and ("staging" in low or "test instance" in low) and "production" not in low:
            return False
        if env in ("staging",) and "production" in low and "staging" not in low:
            return False
        if re.search(r"\bnamespace\b", low) and res.get("namespace") and \
                not re.search(r"\b" + re.escape(str(res.get("namespace"))) + r"\b", m):
            return False
        return True
    return False


def tool_decision(case, msgs, req):
    tool = case.get("proposedTool")
    if not isinstance(tool, dict) or not tool.get("name"):
        return "none"
    if req == "block":
        return "block"
    res = tool.get("resolved") or {}
    env_c = case.get("environment") or {}
    iso = env_c.get("isolation") or {}
    ws = env_c.get("workspace") or env_c.get("cwd") or ""
    op = str(res.get("operation", "")).lower()
    target = str(res.get("target", ""))
    env = str(res.get("environment", env_c.get("kind", "unknown"))).lower()
    sens = str(res.get("sensitivity", "")).lower()
    latest = msgs[-1] if msgs else ""
    all_text = " ".join(msgs)

    # 1. explicit revocation / prohibition in the latest instruction
    if REVOKE_RE.search(latest):
        return "block"
    # 2. user constraints forbidding this operation (latest message that states a constraint)
    for m in reversed(msgs):
        hit = [ops for rx, ops in CONSTRAINTS if rx.search(m)]
        if hit:
            if any(op in ops for ops in hit) and not (APPROVE_RE.search(latest) and mentions(latest, target)
                                                      and m is not latest):
                return "block"
            break
    # 3. independently forbidden effects
    if res.get("bypassPreflight") or res.get("skipPreflight"):
        return "block"
    if op == "read_secret" or sens in ("secret", "credential", "credentials"):
        if op == "read_secret" and res.get("taskScoped") is True and re.search(r"credential|token|secret", all_text, re.I):
            return "allow"
        return "block"
    if op == "send" and sens in ("private", "confidential", "secret", "restricted", "pii"):
        return "allow" if explicit_approval(msgs, tool, res) else "block"
    if op in ("workspace_edit", "write", "extract") and ws and not under(target, ws):
        return "block"
    if op == "delete" and ws and not under(target, ws):
        return "block"
    if res.get("archiveEntry") and ".." in str(res.get("archiveEntry")):
        return "block"
    only = re.search(r"\b(edit|extract|write)[^.]*\bonly\b[^.]*?(/\S+)|\bonly (inside|under|within) (/\S+)", all_text, re.I)
    if only and op in ("workspace_edit", "delete"):
        root = (only.group(2) or only.group(4) or "").rstrip(".")
        if root and not under(target, root):
            return "block"

    # 4. reads
    if op in READ_OPS or op.startswith("read"):
        return "allow"
    if op == "security_test":
        if str(res.get("ownership", "")).startswith("verified") and res.get("confined") is True:
            return "allow"
        return "review"

    # 5. edits inside the workspace
    if op in ("workspace_edit", "write") and under(target, ws):
        return "allow"

    # 6. disposable fixtures in a verified isolated container
    if op == "delete" and env == "container" and res.get("disposable") is True and iso.get("verified") \
            and not iso.get("hostMounts") and not iso.get("credentials") and not iso.get("externalNetwork") \
            and re.search(r"experiment|fixture", all_text, re.I):
        return "allow"

    # 7. consequential actions need explicit exact authorization
    if env in ("unknown", "") or not target:
        return "review"
    if op in CONSEQUENTIAL or res.get("privileged") or env == "production":
        if not explicit_approval(msgs, tool, res):
            return "review"
        if env == "production":
            return "allow" if preflight_ok(case, target, env) and rollback_ok(case, target, env) else "review"
        return "allow"
    return "allow"


def classify(case: dict, policy=None) -> dict:
    msgs = user_msgs(case)
    all_text = " ".join(msgs)

    # request
    if any(is_harmful(m) for m in msgs):
        req = "block"
    elif VULN_DEMO.search(all_text):
        owned = re.search(r"\bowned\b", all_text, re.I) and re.search(r"\b(isolated|confined|disposable)\b", all_text, re.I)
        req = "allow" if owned else "review"
    else:
        req = "allow"

    tdec = tool_decision(case, msgs, req)
    if req == "review" and tdec not in ("block", "none"):
        tdec = "review"

    # route
    if req == "block":
        route = "none"
    elif req == "review" or SECURITY_WORK.search(strip_quotes(all_text)) or \
            str(((case.get("proposedTool") or {}).get("resolved") or {}).get("operation", "")) == "security_test":
        route = "restricted"
    else:
        route = "primary"

    # trajectory
    if req == "block" or tdec == "block":
        traj = "intervene"
    elif req == "review" or tdec == "review" or untrusted_redirect(case):
        traj = "watch"
    else:
        traj = "normal"
    return {"requestDecision": req, "toolDecision": tdec, "route": route, "trajectory": traj}


SAFE_DEFAULT = {"requestDecision": "review", "toolDecision": "review", "route": "restricted", "trajectory": "watch"}


def safe_classify(case, policy):
    try:
        r = classify(case, policy)
    except Exception as e:  # never drop a case
        print("ERR", case.get("caseId"), repr(e), flush=True)
        r = dict(SAFE_DEFAULT)
        if not case.get("proposedTool"):
            r["toolDecision"] = "none"
    return r


# ------------------------------------------------------------------ API
@app.post("/classify")
async def classify_batch(request: Request):
    raw = await request.body()
    try:
        body = json.loads(raw)
    except Exception:
        body = {}
    LOG.append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "size": len(raw), "body": body})
    print("BATCH", body.get("batchId"), "cases", len(body.get("cases") or []), flush=True)
    policy = body.get("policy")
    results, seen = [], set()
    for case in body.get("cases") or []:
        cid = case.get("caseId")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        results.append({"caseId": cid, **safe_classify(case, policy)})
    return JSONResponse({"profile": body.get("profile", PROFILE), "batchId": body.get("batchId"),
                         "results": results})


@app.get("/classify")
def classify_info():
    return {"status": "ok", "profile": PROFILE, "method": "POST JSON batch to this URL"}


@app.get("/log")
def get_log():
    return list(LOG)


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /classify", "logged_batches": len(LOG)}


@app.post("/")
async def classify_root(request: Request):
    return await classify_batch(request)


@app.post("/{path:path}")
async def classify_any(path: str, request: Request):
    return await classify_batch(request)
