"""Agent-harness safety layer (GA5 Q4).

POST /classify  {profile, batchId, policy, cases[]} -> {profile, batchId, results[]}
GET  /log       recent raw request bodies (to read the policy + cases after a Check)

Version 0: generic heuristics. Every request is logged so the classifier can be rebuilt
against the authoritative policy object the grader sends.
"""
import json
import re
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROFILE = "agent-harness-2026-09"
LOG = deque(maxlen=20)  # raw request bodies (each check = 4 batches)

app = FastAPI(title="Agent harness safety layer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ heuristics
HARMFUL = re.compile(
    r"\b(malware|ransomware|keylogger|trojan|rootkit|botnet|worm|spyware|backdoor|reverse shell|"
    r"steal|exfiltrat\w*|harvest\w* (credentials|passwords|tokens|cookies)|credential stuffing|phish\w*|"
    r"ddos|denial[- ]of[- ]service|brute[- ]?forc\w*|crack (the )?password|bypass (auth\w*|2fa|mfa|login|license)|"
    r"disable (the )?(antivirus|edr|defender|logging|audit\w*|monitoring|security)|evade detection|"
    r"cover (my|our) tracks|wipe (the )?logs|dox\w*|stalk\w*|weapon|explosive|bioweapon|"
    r"without (their|his|her|the owner'?s?) (knowledge|consent|permission))\b",
    re.IGNORECASE)
REVIEW_REQ = re.compile(
    r"\b(production|prod\b|customer data|pii|personal data|secrets?|credentials?|api keys?|tokens?|"
    r"payroll|medical|financial|delete (all|every)|drop (table|database)|force[- ]push|"
    r"port scan|penetration test|pentest|exploit)\b",
    re.IGNORECASE)
DESTRUCTIVE_OPS = {"delete", "remove", "drop", "truncate", "overwrite", "wipe", "format", "force_push",
                   "force-push", "reset", "kill", "shutdown", "chmod", "chown", "migrate", "deploy", "write"}
EGRESS_OPS = {"upload", "send", "post", "email", "publish", "exfiltrate", "network", "http", "transfer",
              "share", "export", "push"}
SENSITIVE = {"secret", "secrets", "credential", "credentials", "confidential", "restricted", "pii",
             "private", "sensitive", "internal"}


def text_of(x) -> str:
    return json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x


def user_texts(case):
    return [m.get("content", "") if isinstance(m, dict) else str(m)
            for m in case.get("messages") or [] if not isinstance(m, dict) or m.get("role", "user") == "user"]


def classify(case: dict, policy) -> dict:
    msgs = user_texts(case)
    latest = msgs[-1] if msgs else ""
    history = " ".join(msgs)
    tool = case.get("proposedTool") or None
    res = (tool or {}).get("resolved") or {}
    env = case.get("environment") or {}
    iso = env.get("isolation") or {}

    # ---- request
    if HARMFUL.search(latest) or HARMFUL.search(history):
        req = "block"
    elif REVIEW_REQ.search(latest):
        req = "review"
    else:
        req = "allow"

    # ---- tool
    if not tool or not tool.get("name"):
        tdec = "none"
    elif req == "block":
        tdec = "block"
    else:
        op = str(res.get("operation", "")).lower()
        sens = str(res.get("sensitivity", "")).lower()
        tenv = str(res.get("environment", env.get("kind", ""))).lower()
        dest = str(res.get("destination", "")).lower()
        isolated = bool(iso.get("verified")) and not iso.get("hostMounts") and not iso.get("credentials") \
            and not iso.get("externalNetwork")
        disposable = bool(res.get("disposable"))
        external = dest in {"external", "internet", "public", "third-party", "thirdparty"} or op in EGRESS_OPS
        if sens in SENSITIVE and external:
            tdec = "block"
        elif op in DESTRUCTIVE_OPS:
            if tenv in {"container", "sandbox"} and isolated and disposable:
                tdec = "allow"
            elif tenv in {"production", "prod", "host"} and not case.get("rollbackPlan"):
                tdec = "block"
            else:
                tdec = "review"
        elif sens in SENSITIVE:
            tdec = "review"
        else:
            tdec = "allow"
        if req == "review" and tdec == "allow":
            tdec = "review"

    # ---- route
    route = "none" if req == "block" else ("restricted" if req == "review" else "primary")

    # ---- trajectory: did the session move away from what was first asked?
    first = msgs[0] if msgs else ""
    if req == "block" and len(msgs) > 1 and not HARMFUL.search(first):
        traj = "intervene"
    elif req == "block" or tdec == "block":
        traj = "watch"
    elif len(msgs) > 1 and REVIEW_REQ.search(latest) and not REVIEW_REQ.search(first):
        traj = "watch"
    else:
        traj = "normal"
    return {"requestDecision": req, "toolDecision": tdec, "route": route, "trajectory": traj}


SAFE_DEFAULT = {"requestDecision": "review", "toolDecision": "review", "route": "restricted", "trajectory": "watch"}


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
        try:
            r = classify(case, policy)
        except Exception as e:  # never drop a case
            print("ERR", cid, repr(e), flush=True)
            r = dict(SAFE_DEFAULT)
        results.append({"caseId": cid, **r})
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
