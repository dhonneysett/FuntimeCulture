
# Behavioral & Emotional Alignment (BEA) Assessment Prototype
# ----------------------------------------------------------------------
# This Streamlit app is a research-aligned *prototype* assessment system.
# It is NOT a licensed EQ-i/MSCEIT/Genos/MBTI replacement and should not
# be used as a sole hiring gatekeeper without local validation, fairness
# checks, and (where required) appropriately registered professionals.
#
# Research basis embedded (high-level):
# - Culture as repeated, observable behaviors ("behavioral alignment")
#   and a practical set of 20 load-bearing behaviors (core + differentiators).
# - Cautions about culture-fit scoring in hiring (context-dependent, bias risk,
#   prefer profile-comparison + local validation; “fit” vs “add” distinction).
# - EQ/EI scoring realities (ability vs self-report vs 360; validity indices,
#   norms, measurement error; interpret patterns not single micro-scores).
# - South Africa: employment assessment legality, fairness, and privacy are
#   central to defensibility (EEA/POPIA considerations).
#
# See the "Research notes" page inside the app for a concise summary.
# ----------------------------------------------------------------------

from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Tuple, Any

import os
import io
import uuid
import hashlib
import secrets
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Optional PDF export (reportlab is installed in this environment)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

from supabase import create_client

# ✅ Put this near the top (after imports)
@st.cache_resource
def get_supabase():
    """Create a Supabase client if secrets are configured.

    NOTE: The Supabase *service role* key must only ever live in Streamlit Secrets
    (server-side). Never commit it to GitHub.
    """
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_SERVICE_KEY" not in st.secrets:
        return None
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_KEY"],
    )

sb = get_supabase()

def supabase_available() -> bool:
    return sb is not None

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_token() -> str:
    return secrets.token_urlsafe(32)

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_query_param(name: str) -> str | None:
    """Robust query param fetch across Streamlit versions."""
    try:
        v = st.query_params.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        try:
            v = st.experimental_get_query_params().get(name)
            return v[0] if v else None
        except Exception:
            return None


def secret_truthy(name: str, default: bool = False) -> bool:
    try:
        raw = st.secrets.get(name, default)
    except Exception:
        raw = default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
    return bool(default)


def get_profile_snapshot(
    role_label: str,
    org_name: str | None = None,
    modules: Dict[str, bool] | None = None,
    lock_modules: bool = True,
) -> Dict[str, Any]:
    """Snapshot the current Org/Role configuration into a JSON-able dict.

    This is stored on the invite row so the candidate uses the exact same
    parameters you configured at invite creation time.
    """
    snap = {
        "version": "bea_mvp_1.1",
        "captured_at": _now_utc_iso(),
        "org_name": (org_name or st.session_state.get("org_name", "")).strip(),
        "role_label": (role_label or "").strip(),
        "org_values": st.session_state.get("org_values", {v: 4 for v in VALUES_DIMENSIONS}),
        "role_weights": st.session_state.get("role_weights", {b["id"]: 1.0 for b in BEHAVIORS}),
        "modules": modules or {
            "behaviour": True,
            "values": True,
            "ei_sjt": True,
            "big5": True,
            "leadership": True,
            "type_lens": True,
            "mgmt_prefs": True,
            "conflict": True,
        },
        "lock_modules": bool(lock_modules),
    }
    return snap


def apply_profile_snapshot(snap: Dict[str, Any] | None) -> None:
    """Apply a stored snapshot into session_state for consistent scoring."""
    if not snap:
        return
    try:
        org_vals = snap.get("org_values")
        if isinstance(org_vals, dict) and org_vals:
            # ensure all dimensions exist
            merged = {v: 4 for v in VALUES_DIMENSIONS}
            for k, v in org_vals.items():
                if k in merged:
                    try:
                        merged[k] = float(v)
                    except Exception:
                        pass
            st.session_state["org_values"] = merged

        role_w = snap.get("role_weights")
        if isinstance(role_w, dict) and role_w:
            merged_w = {b["id"]: 1.0 for b in BEHAVIORS}
            for k, v in role_w.items():
                if k in merged_w:
                    try:
                        merged_w[k] = float(v)
                    except Exception:
                        pass
            st.session_state["role_weights"] = merged_w

        if isinstance(snap.get("org_name"), str) and snap.get("org_name").strip():
            st.session_state["org_name"] = snap["org_name"].strip()

        if isinstance(snap.get("role_label"), str) and snap.get("role_label").strip():
            st.session_state["default_role_label"] = snap["role_label"].strip()
    except Exception:
        # Never fail the whole app if a snapshot has unexpected structure
        pass

def parse_dt(val: str) -> datetime:
    """Parse ISO timestamps returned by Supabase."""
    # Supabase typically returns ISO strings with timezone info.
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)

def require_admin() -> None:
    """Very simple Vervio gate (MVP)."""
    if "ADMIN_PASSWORD" not in st.secrets:
        st.error("ADMIN_PASSWORD is not set in Streamlit Secrets.")
        st.stop()

    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False

    if not st.session_state.admin_ok:
        st.title("Vervio Dashboard Login")
        pw = st.text_input("Password", type="password")
        if st.button("Unlock"):
            st.session_state.admin_ok = (pw == st.secrets["ADMIN_PASSWORD"])
        if not st.session_state.admin_ok:
            st.stop()

def sb_get_invite_by_token(token: str) -> Dict[str, Any] | None:
    if not supabase_available():
        return None
    th = hash_token(token)
    res = sb.table("invites").select("*").eq("token_hash", th).limit(1).execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def sb_mark_invite_started(invite_id: int) -> None:
    if not supabase_available():
        return
    try:
        sb.table("invites").update({
            "status": "started",
            "started_at": _now_utc_iso(),
        }).eq("id", invite_id).execute()
    except Exception:
        pass

def sb_mark_invite_completed(invite_id: int) -> None:
    if not supabase_available():
        return
    try:
        sb.table("invites").update({"status": "completed", "completed_at": _now_utc_iso()}).eq("id", invite_id).execute()
    except Exception:
        pass

def sb_expire_invite(invite_id: int) -> None:
    if not supabase_available():
        return
    try:
        sb.table("invites").update({"status": "expired"}).eq("id", invite_id).execute()
    except Exception:
        pass


def sb_revoke_invite(invite_id: int) -> None:
    if not supabase_available():
        return
    try:
        sb.table("invites").update({
            "status": "revoked",
            "revoked_at": _now_utc_iso(),
        }).eq("id", invite_id).execute()
    except Exception:
        pass


def sb_create_invite(
    role: str,
    candidate_email: str | None,
    expires_days: int = 7,
    org: str | None = None,
    config_json: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create an invite record and return {token, link, invite_id}.

    - org + config_json allow you to snapshot the Org/Role profile at invite creation time
      (so the candidate assessment uses the exact same parameters, even if you later change them).
    """
    if not supabase_available():
        raise RuntimeError("Supabase is not configured.")
    if "APP_BASE_URL" not in st.secrets:
        raise RuntimeError("APP_BASE_URL is not set in Streamlit Secrets.")

    token = make_token()
    th = hash_token(token)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=int(expires_days))).isoformat()

    payload: Dict[str, Any] = {
        "candidate_email": candidate_email or None,
        "role": role,
        "token_hash": th,
        "status": "invited",
        "expires_at": expires_at,
        "org": (org or "").strip() or None,
        "config_json": config_json or None,
    }
    out = sb.table("invites").insert(payload).execute()
    invite_row = (getattr(out, "data", None) or [None])[0]
    invite_id = invite_row["id"] if invite_row and "id" in invite_row else None

    base = str(st.secrets["APP_BASE_URL"]).rstrip("/")
    link = f"{base}/?token={token}"
    return {"token": token, "token_hash": th, "link": link, "invite_id": invite_id}
def sb_store_result(invite_id: int, scores_json: Dict[str, Any], report_json: Dict[str, Any]) -> None:
    if not supabase_available():
        return
    sb.table("results").insert({
        "invite_id": invite_id,
        "scores_json": scores_json,
        "report_json": report_json,
    }).execute()


def sb_upload_pdf_to_storage(invite_id: int, pdf_bytes: bytes) -> str | None:
    """Optional: upload the PDF to Supabase Storage and return the stored path.

    To enable:
      - Create a Storage bucket (e.g. 'bea-reports')
      - Add SUPABASE_REPORTS_BUCKET in Streamlit Secrets with that bucket name
    """
    if not supabase_available():
        return None
    bucket = str(st.secrets.get("SUPABASE_REPORTS_BUCKET", "")).strip() if hasattr(st, "secrets") else ""
    if not bucket:
        return None
    try:
        key = f"reports/invite_{invite_id}_{uuid.uuid4().hex[:8]}.pdf"
        sb.storage.from_(bucket).upload(
            key,
            pdf_bytes,
            {"content-type": "application/pdf", "upsert": "true"},
        )
        return key
    except Exception:
        return None

APP_TITLE = "BEA • Behavioural & Emotional Alignment Assessment (Prototype)"
VERSION = "0.2.0"

# -----------------------------
# Utility helpers
# -----------------------------

LIKERT_LABELS = [
    "1 — Almost never",
    "2 — Rarely",
    "3 — Sometimes",
    "4 — Often",
    "5 — Almost always",
]

LIKERT_VALUES = [1, 2, 3, 4, 5]

VALUES_SCALE_LABELS = [
    "1 — Not important",
    "2 — Slightly important",
    "3 — Moderately important",
    "4 — Very important",
    "5 — Essential",
]

SJT_OPTIONS = ["A", "B", "C", "D"]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    """Map mean Likert score to 0–100."""
    if hi == lo:
        return 50.0
    return 100.0 * (x - lo) / (hi - lo)

def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with safe fallback."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def response_flags(
    responses: Dict[str, Any],
    duration_sec: float,
    inconsistency_pairs: List[Tuple[str, str]] | None = None,
    attention_checks: Dict[str, int] | None = None,
    infrequency_items: List[str] | None = None,
    min_duration_sec: float = 180.0,
) -> Dict[str, Any]:
    """Response-quality checks inspired by common validity-index practice.

    Important: These checks can *flag* patterns consistent with low effort or impression
    management, but they do not "prove" dishonesty. Treat flags as prompts to corroborate
    with interviews, work samples, and references.
    """
    numeric = {k: v for k, v in responses.items() if isinstance(v, (int, float))}
    vals = np.array(list(numeric.values()), dtype=float) if numeric else np.array([], dtype=float)

    flags: Dict[str, Any] = {
        "duration_sec": float(duration_sec),
        "warnings": [],
        "attention_checks_failed": 0,
        "attention_checks_total": 0,
        "infrequency_hits": 0,
        "infrequency_total": 0,
        "inconsistency_checked_pairs": 0,
        "inconsistency_hits": 0,
        "mean": float(np.mean(vals)) if len(vals) else float("nan"),
        "sd": float(np.std(vals)) if len(vals) else float("nan"),
    }

    # Timing
    if duration_sec < min_duration_sec:
        flags["warnings"].append("Very fast completion time (possible rushing).")

    # Central tendency / variability
    if len(vals) >= 10:
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        if mean >= 4.7:
            flags["warnings"].append("Very high average endorsement (possible 'fake good' / impression management).")
        if mean <= 1.3:
            flags["warnings"].append("Very low average endorsement (possible 'fake bad' / disengagement).")
        if sd <= 0.35:
            flags["warnings"].append("Very low response variability (possible straight-lining).")

        # Simple straight-lining proxy (dominant single response)
        unique, counts = np.unique(vals, return_counts=True)
        dom = float(np.max(counts) / len(vals))
        flags["dominant_response_ratio"] = dom
        if dom >= 0.85:
            flags["warnings"].append("A single response option was used for most items (possible straight-lining).")

    # Attention checks (instructed-response)
    if attention_checks:
        total = 0
        failed = 0
        for item_id, expected in attention_checks.items():
            if item_id in numeric:
                total += 1
                if int(numeric[item_id]) != int(expected):
                    failed += 1
        flags["attention_checks_total"] = total
        flags["attention_checks_failed"] = failed
        if total >= 1 and failed >= 1:
            flags["warnings"].append("Failed an instructed-response check (possible inattention).")

    # Infrequency / improbable-virtue items
    if infrequency_items:
        total = 0
        hits = 0
        for item_id in infrequency_items:
            if item_id in numeric:
                total += 1
                if int(numeric[item_id]) >= 5:
                    hits += 1
        flags["infrequency_total"] = total
        flags["infrequency_hits"] = hits
        if total >= 2 and hits >= 1:
            flags["warnings"].append("Endorsed improbable-virtue items strongly (possible impression management).")

    # Inconsistency pairs
    if inconsistency_pairs:
        inconsistent = 0
        checked = 0
        for a, b in inconsistency_pairs:
            if a in numeric and b in numeric:
                checked += 1
                if abs(float(numeric[a]) - float(numeric[b])) >= 3:
                    inconsistent += 1
        flags["inconsistency_checked_pairs"] = checked
        flags["inconsistency_hits"] = inconsistent
        if checked >= 3 and inconsistent >= 2:
            flags["warnings"].append("Inconsistent responding on paired items (interpret with caution).")

    return flags

def band(score_0_100: float) -> str:
    if score_0_100 >= 75:
        return "High"
    if score_0_100 >= 45:
        return "Moderate"
    return "Watch-out"

# -----------------------------
# Question banks (original items)
# -----------------------------
# NOTE: Items are newly authored. They are aligned to research constructs but do not
# reproduce any proprietary instrument content.

BEHAVIORS: List[Dict[str, Any]] = [
    # Core 1–10
    {"id":"B01", "name":"Integrity & ethical consistency", "core":True, "items":[
        "I report issues and mistakes honestly, even when it is uncomfortable.",
        "I avoid cutting corners that would compromise standards or safety.",
        "When I commit to a standard, I follow it consistently under pressure.",
    ]},
    {"id":"B02", "name":"Respectful communication & civility", "core":True, "items":[
        "I stay respectful when stressed or disagreeing.",
        "I avoid sarcasm, intimidation, or humiliating language.",
        "I acknowledge others’ contributions in day-to-day work.",
    ]},
    {"id":"B03", "name":"Dependability & follow-through", "core":True, "items":[
        "I meet deadlines or renegotiate early when I cannot.",
        "I finish tasks to the expected quality standard.",
        "Others can rely on me to do what I said I would do.",
    ]},
    {"id":"B04", "name":"Accountability with ownership (not blame)", "core":True, "items":[
        "When something goes wrong, I focus on solutions and learning rather than blame.",
        "I surface problems early instead of hoping they disappear.",
        "I take responsibility for outcomes within my control.",
    ]},
    {"id":"B05", "name":"Information sharing & transparency by default", "core":True, "items":[
        "I share relevant updates quickly with the people affected.",
        "I document decisions or handovers so others can execute.",
        "I avoid keeping critical information to myself when it helps the team.",
    ]},
    {"id":"B06", "name":"Coachability through active feedback seeking", "core":True, "items":[
        "I ask for feedback at meaningful moments (after delivery, after conflict, after a shift).",
        "I clarify what “good” looks like rather than guessing.",
        "I follow up to show what I changed based on feedback.",
    ]},
    {"id":"B07", "name":"Constructive feedback & developing others", "core":True, "items":[
        "When giving feedback, I describe specific behaviors and their impact.",
        "I focus feedback on improvement and support, not ego or punishment.",
        "I coach others with follow-up instead of a once-off comment.",
    ]},
    {"id":"B08", "name":"Helping behavior & citizenship", "core":True, "items":[
        "I proactively help teammates during overload without being asked.",
        "I share useful knowledge and tips that make others better.",
        "I do small helpful actions that prevent bigger failures later.",
    ]},
    {"id":"B09", "name":"Speaking up early (voice) & surfacing risk", "core":True, "items":[
        "I raise concerns early when I see risk to guests/customers, safety, or quality.",
        "I challenge decisions respectfully when something seems off.",
        "I propose improvements instead of only pointing out problems.",
    ]},
    {"id":"B10", "name":"Reflection & continuous improvement (reflexivity)", "core":True, "items":[
        "After a busy period, I reflect on what to improve next time.",
        "I support quick debriefs/retrospectives to turn learning into action.",
        "I test small process changes and check whether they worked.",
    ]},
    # Differentiators 11–20
    {"id":"B11", "name":"Adaptability & learning agility", "core":False, "items":[
        "I learn new systems or processes quickly enough to stay effective.",
        "I adjust plans when priorities change.",
        "I stay calm and functional when routines are disrupted.",
    ]},
    {"id":"B12", "name":"Creative problem solving & experimentation", "core":False, "items":[
        "I generate multiple options before locking in a solution.",
        "I run small tests to check assumptions instead of guessing.",
        "I share learnings (including what didn’t work) so others benefit.",
    ]},
    {"id":"B13", "name":"Constructive conflict management", "core":False, "items":[
        "I keep disagreements focused on ideas and facts, not personalities.",
        "I use structure (goals, decision rules, timeboxing) to resolve debates.",
        "I actively lower tension so the team can decide and move forward.",
    ]},
    {"id":"B14", "name":"Building psychological safety for others", "core":False, "items":[
        "I invite input from quieter people and make space for dissenting views.",
        "I respond constructively when others admit mistakes or ask questions.",
        "I protect people who raise issues from being mocked or punished.",
    ]},
    {"id":"B15", "name":"Boundary spanning & cross-functional coordination", "core":False, "items":[
        "I coordinate across departments to prevent handoff failures.",
        "I translate constraints so different teams can work smoothly together.",
        "I avoid “local optimization” that harms the wider operation.",
    ]},
    {"id":"B16", "name":"Customer/service orientation", "core":False, "items":[
        "I take ownership of the customer/guest experience end-to-end.",
        "I recover service failures calmly without defensiveness.",
        "I anticipate needs and communicate proactively.",
    ]},
    {"id":"B17", "name":"Role clarity creation (structure-setting)", "core":False, "items":[
        "I clarify decision rights, handoffs, and expectations when things are unclear.",
        "I define what “good” looks like to reduce confusion and rework.",
        "I create simple checklists or standards that help the team execute.",
    ]},
    {"id":"B18", "name":"Trust-building reliability (micro-behaviors)", "core":False, "items":[
        "I keep confidences and handle sensitive info responsibly.",
        "I treat people fairly and consistently.",
        "I show predictable follow-through that builds trust over time.",
    ]},
    {"id":"B19", "name":"Resilience & emotional self-regulation under stress", "core":False, "items":[
        "When under pressure, I regulate tone and stay respectful.",
        "I avoid spreading panic or negativity that affects others’ performance.",
        "I make calm tradeoffs and prioritize what matters most.",
    ]},
    {"id":"B20", "name":"Alignment to purpose & impact", "core":False, "items":[
        "I connect daily tasks to outcomes that matter to the business/guest.",
        "I prioritize work based on value and impact, not only activity.",
        "I consider how my actions reinforce the team’s norms and standards.",
    ]},
]

# Simple values profile for P–O fit (profile comparison) — org can configure weights.
VALUES_DIMENSIONS = [
    "Guest/customer care",
    "Integrity & compliance",
    "Teamwork & helping",
    "Learning & growth",
    "Quality & craft",
    "Accountability",
    "Innovation & improvement",
    "Speed & execution",
]

# EI SJT (original scenarios with expert-key weights 0–1)
EI_SJT: List[Dict[str, Any]] = [
    {
        "id":"E01",
        "branch":"Emotion Management",
        "prompt":"A guest is angry about a delay. They raise their voice at the desk. What do you do first?",
        "options":{
            "A":"Match their intensity so they know you’re taking it seriously.",
            "B":"Keep a calm tone, acknowledge their frustration, and ask one clarifying question.",
            "C":"Tell them to calm down or you can’t help them.",
            "D":"Ignore the emotion and jump straight into policy details.",
        },
        "weights":{"A":0.2, "B":1.0, "C":0.0, "D":0.4},
    },
    {
        "id":"E02",
        "branch":"Emotion Perception",
        "prompt":"A colleague says 'I’m fine' but avoids eye contact and goes quiet. What’s the best next step?",
        "options":{
            "A":"Assume they are fine and continue as normal.",
            "B":"Privately check in with a simple, non-pressuring question.",
            "C":"Tell the team they’re upset so everyone can be careful.",
            "D":"Confront them in front of others to get the truth.",
        },
        "weights":{"A":0.4, "B":1.0, "C":0.1, "D":0.0},
    },
    {
        "id":"E03",
        "branch":"Emotion Understanding",
        "prompt":"A team member becomes defensive when given feedback. What is the most likely driver?",
        "options":{
            "A":"They don’t care about the job.",
            "B":"They interpret feedback as a threat to status or competence.",
            "C":"They always want to argue for fun.",
            "D":"They want to embarrass the manager.",
        },
        "weights":{"A":0.2, "B":1.0, "C":0.3, "D":0.0},
    },
    {
        "id":"E04",
        "branch":"Emotion Management",
        "prompt":"You feel yourself getting irritated in a meeting. What is the most effective in-the-moment action?",
        "options":{
            "A":"Interrupt quickly so you can 'win' the point before you lose control.",
            "B":"Slow down your breathing and ask a question to buy time.",
            "C":"Stay silent and mentally check out until it’s over.",
            "D":"Send a sarcastic message afterward to vent.",
        },
        "weights":{"A":0.2, "B":1.0, "C":0.5, "D":0.0},
    },
    {
        "id":"E05",
        "branch":"Emotion Use",
        "prompt":"Your team is flat and unmotivated after a tough week. What helps most to lift energy responsibly?",
        "options":{
            "A":"Pretend everything is perfect and deny the stress.",
            "B":"Name the challenge, recognize effort, and set one clear achievable goal for today.",
            "C":"Threaten consequences so people work harder.",
            "D":"Let people vent for an hour without moving to action.",
        },
        "weights":{"A":0.2, "B":1.0, "C":0.0, "D":0.5},
    },
    {
        "id":"E06",
        "branch":"Emotion Understanding",
        "prompt":"Two staff members are in conflict. One is angry; the other is withdrawn. What is a good first frame?",
        "options":{
            "A":"Anger usually signals a boundary/need; withdrawal may signal safety concerns or overwhelm.",
            "B":"One is right and one is wrong; decide quickly and move on.",
            "C":"Ignore it; it will resolve itself.",
            "D":"Punish both equally so it feels fair.",
        },
        "weights":{"A":1.0, "B":0.4, "C":0.0, "D":0.1},
    },
    {
        "id":"E07",
        "branch":"Emotion Perception",
        "prompt":"A candidate answers every question with a smile but their hands shake slightly. What’s your best inference?",
        "options":{
            "A":"They are lying.",
            "B":"They are likely anxious (which is normal) and you should reduce threat cues.",
            "C":"They are incompetent.",
            "D":"They are trying to manipulate you.",
        },
        "weights":{"A":0.2, "B":1.0, "C":0.1, "D":0.2},
    },
    {
        "id":"E08",
        "branch":"Emotion Management",
        "prompt":"A mistake happened and the team is blaming each other. What approach best restores performance?",
        "options":{
            "A":"Find the person responsible and make an example.",
            "B":"Run a short facts-first review: what happened, what we’ll change, who owns the fix.",
            "C":"Tell everyone to stop being dramatic.",
            "D":"Avoid discussing it to keep morale up.",
        },
        "weights":{"A":0.0, "B":1.0, "C":0.2, "D":0.3},
    },
]

# Big Five (original short scale; reverse-coded items are marked)
BIG5_ITEMS = [
    # Extraversion
    ("P01", "I feel energized by frequent interaction with people.", "E", False),
    ("P02", "I prefer quiet environments over busy social settings.", "E", True),
    ("P03", "I speak up easily in groups when I have something to add.", "E", False),
    ("P04", "I often keep my thoughts to myself, even when it would help.", "E", True),
    # Agreeableness
    ("P05", "I assume positive intent and look for win–win outcomes.", "A", False),
    ("P06", "I can be sharp or harsh when people make mistakes.", "A", True),
    ("P07", "I listen to understand before trying to be understood.", "A", False),
    ("P08", "I enjoy arguing more than collaborating.", "A", True),
    # Conscientiousness
    ("P09", "I plan ahead and keep track of commitments.", "C", False),
    ("P10", "I leave things to the last minute more often than I should.", "C", True),
    ("P11", "I follow systems/checklists when quality matters.", "C", False),
    ("P12", "I struggle to stay organized when workload increases.", "C", True),
    # Emotional stability (reverse of Neuroticism)
    ("P13", "I recover quickly after stressful events at work.", "S", False),
    ("P14", "Small problems can feel overwhelming for me.", "S", True),
    ("P15", "I stay even-tempered when things go wrong.", "S", False),
    ("P16", "I worry a lot about what might go wrong.", "S", True),
    # Openness
    ("P17", "I enjoy learning new tools, methods, or ways of working.", "O", False),
    ("P18", "I prefer proven routines and avoid changing processes.", "O", True),
    ("P19", "I’m curious and ask questions to understand how things work.", "O", False),
    ("P20", "I’m uncomfortable experimenting unless success is guaranteed.", "O", True),
]

# Leadership / management style (original)
LEADERSHIP_ITEMS = [
    ("L01", "I set clear expectations and follow up on commitments.", "Structure", False),
    ("L02", "I adapt my style based on the person and the situation.", "Adapt", False),
    ("L03", "I prefer coaching and guiding over directing and telling.", "Coach", False),
    ("L04", "In urgent moments, I become more directive to restore control.", "Coach", True),
    ("L05", "I focus strongly on people’s growth and motivation.", "People", False),
    ("L06", "I focus strongly on results and execution speed.", "Results", False),
    ("L07", "I involve others in decisions that affect their work.", "People", False),
    ("L08", "I prefer to decide quickly without much discussion.", "People", True),
    ("L09", "I keep feedback specific and behavior-based.", "Coach", False),
    ("L10", "I tolerate conflict as long as it stays respectful and useful.", "Adapt", False),
    ("L11", "I create simple systems/checklists to reduce errors.", "Structure", False),
    ("L12", "I am comfortable making decisions with incomplete information.", "Results", False),
]

# -----------------------------
# Additional modules (v0.2)
# -----------------------------
# Personality type lens (Type 1–9, Enneagram-inspired, descriptive)
TYPE_LABELS = {
    1: "Type 1 — Principled Improver",
    2: "Type 2 — Helpful Connector",
    3: "Type 3 — Ambitious Achiever",
    4: "Type 4 — Authentic Individualist",
    5: "Type 5 — Analytical Observer",
    6: "Type 6 — Loyal Questioner",
    7: "Type 7 — Enthusiastic Optimizer",
    8: "Type 8 — Direct Challenger",
    9: "Type 9 — Harmonizing Stabilizer",
}

TYPE_ITEMS = [
    # Type 1
    ("T01", "I feel responsible for improving standards when something is 'not right'.", 1, False),
    ("T02", "I notice errors quickly and want them corrected.", 1, False),
    ("T03", "I hold myself to very high standards, even when no one is watching.", 1, False),
    ("T04", "I can become frustrated when others ignore agreed rules.", 1, False),

    # Type 2
    ("T05", "I naturally look for ways to support people who are struggling.", 2, False),
    ("T06", "I often prioritize relationships when making decisions.", 2, False),
    ("T07", "I notice what people need emotionally and respond quickly.", 2, False),
    ("T08", "I feel fulfilled when my help makes a real difference.", 2, False),

    # Type 3
    ("T09", "I’m motivated by clear goals, metrics, and achievement.", 3, False),
    ("T10", "I adjust my approach to meet expectations and deliver results.", 3, False),
    ("T11", "I feel energized when I am performing at a high level.", 3, False),
    ("T12", "I dislike being seen as ineffective or unprepared.", 3, False),

    # Type 4
    ("T13", "I value authenticity, meaning, and personal expression at work.", 4, False),
    ("T14", "I often reflect deeply on what work means for me.", 4, False),
    ("T15", "I’m sensitive to environments that feel impersonal or 'fake'.", 4, False),
    ("T16", "I prefer work that lets me contribute something distinctive.", 4, False),

    # Type 5
    ("T17", "Before acting, I prefer to understand the system and the facts.", 5, False),
    ("T18", "I need time alone to think clearly and do my best work.", 5, False),
    ("T19", "I can stay calm and analytical when others are emotional.", 5, False),
    ("T20", "I prefer depth over breadth when learning a new topic.", 5, False),

    # Type 6
    ("T21", "I naturally scan for risks and what could go wrong.", 6, False),
    ("T22", "I work best when expectations and escalation paths are clear.", 6, False),
    ("T23", "I value loyalty and trust strongly in working relationships.", 6, False),
    ("T24", "I double-check important details to prevent preventable errors.", 6, False),

    # Type 7
    ("T25", "I bring energy and optimism, especially when things are tough.", 7, False),
    ("T26", "I enjoy variety and get bored by repetitive routines.", 7, False),
    ("T27", "I’m quick to generate options and possibilities.", 7, False),
    ("T28", "I prefer to keep momentum rather than over-analyzing.", 7, False),

    # Type 8
    ("T29", "I speak directly and prefer clarity over diplomacy.", 8, False),
    ("T30", "I feel comfortable taking charge in chaotic moments.", 8, False),
    ("T31", "I respect strength and competence; I lose patience with passivity.", 8, False),
    ("T32", "I will confront issues early rather than let resentment build.", 8, False),

    # Type 9
    ("T33", "I naturally look for harmony and common ground.", 9, False),
    ("T34", "I prefer steady progress over conflict-driven intensity.", 9, False),
    ("T35", "I can see multiple perspectives, even in disagreements.", 9, False),
    ("T36", "I may delay decisions to keep things calm.", 9, False),

    # A few reverse-coded items to reduce acquiescence / add nuance
    ("T37", "I rarely feel any internal pressure to perform well.", 3, True),
    ("T38", "I don’t mind chaos; structure and clarity feel unnecessary.", 6, True),
    ("T39", "I dislike spending time thinking deeply about problems.", 5, True),
    ("T40", "I avoid raising standards; 'good enough' is usually fine.", 1, True),
]

TYPE_COMPATIBILITY = {
    # Heuristic (non-diagnostic) “friction” pairings to support coaching/conversation
    1: {"likely_friction": [5, 6, 7], "likely_synergy": [2, 3, 9]},
    2: {"likely_friction": [5, 8], "likely_synergy": [1, 6, 9]},
    3: {"likely_friction": [4, 9], "likely_synergy": [1, 7, 8]},
    4: {"likely_friction": [1, 3], "likely_synergy": [2, 5, 9]},
    5: {"likely_friction": [2, 7, 8], "likely_synergy": [1, 4, 6]},
    6: {"likely_friction": [7, 8], "likely_synergy": [1, 2, 5]},
    7: {"likely_friction": [1, 5, 6], "likely_synergy": [3, 8, 9]},
    8: {"likely_friction": [2, 6, 9], "likely_synergy": [3, 5, 7]},
    9: {"likely_friction": [3, 8], "likely_synergy": [1, 2, 7]},
}

TYPE_MANAGEMENT_TIPS = {
    1: {
        "strengths": ["High standards", "Integrity", "Process improvement"],
        "watchouts": ["Perfectionism", "Irritability under stress", "Over-correcting others"],
        "manage": ["Agree on clear standards", "Praise progress (not only perfection)", "Invite them to propose improvements without policing others"],
    },
    2: {
        "strengths": ["Supportive", "Relationship-building", "Team glue"],
        "watchouts": ["People-pleasing", "Difficulty saying no", "Burnout risk"],
        "manage": ["Set boundaries and workload clarity", "Acknowledge contribution", "Encourage direct asks rather than rescuing"],
    },
    3: {
        "strengths": ["Goal-driven", "Efficient", "Motivating through achievement"],
        "watchouts": ["Image/approval sensitivity", "Overwork", "Cutting corners if pressured"],
        "manage": ["Give measurable goals", "Reward quality + ethics (not only speed)", "Discuss trade-offs and sustainable pace"],
    },
    4: {
        "strengths": ["Creativity", "Authenticity", "Depth and meaning-making"],
        "watchouts": ["Mood sensitivity", "Feeling misunderstood", "Comparisons/self-doubt"],
        "manage": ["Connect work to purpose", "Provide space for expression/ownership", "Give feedback gently and specifically"],
    },
    5: {
        "strengths": ["Analytical", "Independent", "Deep expertise"],
        "watchouts": ["Withdrawal", "Over-analysis", "Low communication under stress"],
        "manage": ["Give time to think", "Ask for concise updates", "Respect privacy while ensuring alignment points are clear"],
    },
    6: {
        "strengths": ["Risk-aware", "Loyal", "Prepared and thorough"],
        "watchouts": ["Anxiety loops", "Over-checking", "Difficulty with ambiguity"],
        "manage": ["Give clear expectations", "Make escalation paths explicit", "Normalize questions and create psychological safety"],
    },
    7: {
        "strengths": ["Optimistic", "Idea generation", "Energy and momentum"],
        "watchouts": ["Follow-through gaps", "Avoiding discomfort", "Over-committing"],
        "manage": ["Limit priorities", "Use checklists and deadlines", "Help them finish before starting the next thing"],
    },
    8: {
        "strengths": ["Direct", "Decisive", "Protective leadership"],
        "watchouts": ["Intensity", "Bluntness", "Power struggles"],
        "manage": ["Be clear and confident", "Agree on boundaries and tone", "Channel their energy into ownership and accountability"],
    },
    9: {
        "strengths": ["Calm", "Mediator", "Steady presence"],
        "watchouts": ["Avoiding conflict", "Indecision", "Passive resistance"],
        "manage": ["Ask for their opinion early", "Create gentle deadlines", "Make conflict safe and structured"],
    },
}

# “How to manage me” preferences (management needs & motivators)
MGMT_PREF_DIMS = [
    "Autonomy", "Structure", "Feedback", "Recognition", "Pace", "Collaboration", "Change", "Conflict"
]

MGMT_PREF_ITEMS = [
    ("MP01", "I do my best work when I have freedom in how I achieve the outcome.", "Autonomy", False),
    ("MP02", "I prefer detailed instructions and clear step-by-step guidance.", "Autonomy", True),
    ("MP03", "I like clear routines/checklists to reduce mistakes.", "Structure", False),
    ("MP04", "Too many rules and procedures frustrate me.", "Structure", True),
    ("MP05", "I prefer feedback that is direct, specific, and not sugar-coated.", "Feedback", False),
    ("MP06", "I prefer feedback that is gentle and carefully framed.", "Feedback", True),
    ("MP07", "Public recognition motivates me.", "Recognition", False),
    ("MP08", "I prefer recognition in private (1:1).", "Recognition", True),
    ("MP09", "I enjoy a fast pace and many moving parts.", "Pace", False),
    ("MP10", "I prefer a steady pace with fewer urgent changes.", "Pace", True),
    ("MP11", "I prefer working closely with others rather than independently.", "Collaboration", False),
    ("MP12", "I prefer independent work and minimal interruptions.", "Collaboration", True),
    ("MP13", "I get energised by change and new ways of working.", "Change", False),
    ("MP14", "I prefer stability; frequent change drains my energy.", "Change", True),
    ("MP15", "I address conflict directly and early.", "Conflict", False),
    ("MP16", "I prefer to cool off and revisit conflict later.", "Conflict", True),
    ("MP17", "I want my manager to set clear priorities and protect focus time.", "Structure", False),
    ("MP18", "I want my manager to challenge me with stretch goals.", "Pace", False),
    ("MP19", "I value managers who explain the ‘why’ behind decisions.", "Autonomy", False),
    ("MP20", "I prefer decisions to be made quickly, even if not perfect.", "Pace", False),
    ("MP21", "I do best when expectations are written down.", "Structure", False),
    ("MP22", "I prefer spontaneous conversations over written instructions.", "Structure", True),
    ("MP23", "I prefer frequent, short check-ins rather than long formal reviews.", "Feedback", False),
    ("MP24", "I prefer fewer feedback moments, but more depth when they happen.", "Feedback", True),
]

# Conflict style (forced-choice, no right/wrong)
CONFLICT_STYLES = ["Avoiding", "Accommodating", "Competing", "Compromising", "Collaborating"]

CONFLICT_ITEMS = [
    {
        "id": "C01",
        "prompt": "A teammate repeatedly misses deadlines and blames workload.",
        "A": {"text": "Escalate and set firm consequences to protect delivery.", "style": "Competing"},
        "B": {"text": "Explore causes together and co-design a workable plan.", "style": "Collaborating"},
    },
    {
        "id": "C02",
        "prompt": "A colleague challenges your idea in a meeting.",
        "A": {"text": "Defend your position strongly and push for your solution.", "style": "Competing"},
        "B": {"text": "Suggest combining ideas to reach a middle ground.", "style": "Compromising"},
    },
    {
        "id": "C03",
        "prompt": "Two staff members are in conflict and you’re caught in the middle.",
        "A": {"text": "Keep out of it and hope it settles down.", "style": "Avoiding"},
        "B": {"text": "Facilitate a structured conversation to resolve it.", "style": "Collaborating"},
    },
    {
        "id": "C04",
        "prompt": "A customer complaint is emotional and unfair.",
        "A": {"text": "Stay calm, empathise, and aim to restore relationship.", "style": "Accommodating"},
        "B": {"text": "Clarify facts firmly and set boundaries on unacceptable behaviour.", "style": "Competing"},
    },
    {
        "id": "C05",
        "prompt": "You disagree with a manager’s decision, but the team is tired.",
        "A": {"text": "Let it go for now to maintain harmony.", "style": "Avoiding"},
        "B": {"text": "Raise concerns respectfully with proposed alternatives.", "style": "Collaborating"},
    },
    {
        "id": "C06",
        "prompt": "A policy is slowing down service, but it exists for a reason.",
        "A": {"text": "Follow the policy and accept the slowdown.", "style": "Accommodating"},
        "B": {"text": "Propose a compromise: keep the intent, streamline the steps.", "style": "Compromising"},
    },
    {
        "id": "C07",
        "prompt": "A coworker asks you to take on extra work you cannot handle.",
        "A": {"text": "Say yes to help, even if it means working late.", "style": "Accommodating"},
        "B": {"text": "Negotiate scope or decline with clear boundaries.", "style": "Compromising"},
    },
    {
        "id": "C08",
        "prompt": "A recurring issue keeps happening and no one owns it.",
        "A": {"text": "Take ownership yourself and drive a fix quickly.", "style": "Competing"},
        "B": {"text": "Bring the group together to agree on ownership and process.", "style": "Collaborating"},
    },
    {
        "id": "C09",
        "prompt": "A disagreement is getting heated in front of others.",
        "A": {"text": "Pause the discussion and revisit later in private.", "style": "Avoiding"},
        "B": {"text": "De-escalate and keep it constructive in the moment.", "style": "Collaborating"},
    },
    {
        "id": "C10",
        "prompt": "You and another person both need a limited resource.",
        "A": {"text": "Offer to give up your share to preserve the relationship.", "style": "Accommodating"},
        "B": {"text": "Split it now and renegotiate after results are clearer.", "style": "Compromising"},
    },
]

# Quality / validity check items (embedded like normal questions)
ATTENTION_CHECKS = {"AC01": 3, "AC02": 2}  # expected Likert value
ATTENTION_CHECK_ITEMS = [
    ("AC01", "To show you’re paying attention, please select: 3 — Sometimes.", 3),
    ("AC02", "For quality purposes, please select: 2 — Rarely.", 2),
]
INFREQUENCY_ITEMS = ["IF01", "IF02", "IF03"]
INFREQUENCY_ITEM_TEXT = [
    ("IF01", "I have never made a mistake at work — not even once."),
    ("IF02", "I have never felt stressed or frustrated at work."),
    ("IF03", "I always get along with everyone, all the time."),
]

# Inconsistency pairs (similar items phrased differently)
INCONSISTENCY_PAIRS = [
    ("P01", "P02"), ("P09", "P10"), ("P13", "P14"), ("P17", "P18"),
    ("B03_1", "B03_3"), ("B05_1", "B05_3"), ("B06_1", "B06_3"),
    ("MP03", "MP21"), ("MP05", "MP23"), ("T17", "T20"), ("T21", "T24"),
]

# -----------------------------
# Scoring logic
# -----------------------------

def score_behavioral(responses: Dict[str, int], weights: Dict[str, float]) -> Dict[str, Any]:
    rows = []
    for b in BEHAVIORS:
        item_keys = [f"{b['id']}_{i+1}" for i in range(len(b["items"]))]
        vals = [responses.get(k) for k in item_keys if k in responses]
        mean = float(np.mean(vals)) if vals else float("nan")
        rows.append({
            "id": b["id"],
            "behavior": b["name"],
            "core": b["core"],
            "mean_likert": mean,
            "score_0_100": pct(mean) if not math.isnan(mean) else float("nan"),
            "weight": float(weights.get(b["id"], 1.0)),
        })
    df = pd.DataFrame(rows)

    valid = df.dropna(subset=["score_0_100"])
    if len(valid) == 0:
        overall = float("nan")
    else:
        w = valid["weight"].to_numpy()
        s = valid["score_0_100"].to_numpy()
        overall = float(np.average(s, weights=w))

    core = valid[valid["core"]]
    diff = valid[~valid["core"]]
    core_score = float(np.average(core["score_0_100"], weights=core["weight"])) if len(core) else float("nan")
    diff_score = float(np.average(diff["score_0_100"], weights=diff["weight"])) if len(diff) else float("nan")

    top_strengths = df.sort_values("score_0_100", ascending=False).head(5)[["behavior","score_0_100"]].to_dict("records")
    watchouts = df.sort_values("score_0_100", ascending=True).head(5)[["behavior","score_0_100"]].to_dict("records")

    return {
        "table": df,
        "overall": overall,
        "core": core_score,
        "differentiators": diff_score,
        "top_strengths": top_strengths,
        "watchouts": watchouts,
    }

def score_values_fit(candidate_vals: Dict[str, int], org_profile: Dict[str, int]) -> Dict[str, Any]:
    c = np.array([candidate_vals.get(v, 3) for v in VALUES_DIMENSIONS], dtype=float)
    o = np.array([org_profile.get(v, 3) for v in VALUES_DIMENSIONS], dtype=float)

    r = corr(c, o)
    fit_0_100 = 50 + 50 * r
    gaps = (c - o)
    gap_rows = [{"value": VALUES_DIMENSIONS[i], "candidate": float(c[i]), "org": float(o[i]), "gap": float(gaps[i])} for i in range(len(VALUES_DIMENSIONS))]
    gap_df = pd.DataFrame(gap_rows).sort_values("gap", key=lambda s: s.abs(), ascending=False)

    return {
        "fit_corr": r,
        "fit_0_100": fit_0_100,
        "gap_table": gap_df,
        "most_misaligned": gap_df.head(3).to_dict("records"),
    }

def score_ei_sjt(responses: Dict[str, str]) -> Dict[str, Any]:
    per_branch: Dict[str, List[float]] = {}
    item_rows = []
    for item in EI_SJT:
        ans = responses.get(item["id"])
        w = item["weights"].get(ans, 0.0) if ans else 0.0
        item_rows.append({"id": item["id"], "branch": item["branch"], "answer": ans or "-", "credit": w})
        per_branch.setdefault(item["branch"], []).append(w)

    branch_scores = {k: 100.0 * float(np.mean(v)) if v else float("nan") for k, v in per_branch.items()}
    overall = 100.0 * float(np.mean([r["credit"] for r in item_rows])) if item_rows else float("nan")
    df = pd.DataFrame(item_rows)

    return {"overall": overall, "branches": branch_scores, "table": df}

def score_big5(responses: Dict[str, int]) -> Dict[str, Any]:
    trait_map = {"E":"Extraversion", "A":"Agreeableness", "C":"Conscientiousness", "S":"Emotional Stability", "O":"Openness"}
    buckets = {k: [] for k in trait_map.keys()}
    rows = []
    for item_id, text, trait, reverse in BIG5_ITEMS:
        raw = responses.get(item_id)
        if raw is None:
            continue
        scored = 6 - raw if reverse else raw
        buckets[trait].append(scored)
        rows.append({"id": item_id, "trait": trait_map[trait], "raw": raw, "scored": scored, "reverse": reverse})
    trait_scores = {}
    for t, vals in buckets.items():
        trait_scores[trait_map[t]] = pct(float(np.mean(vals))) if vals else float("nan")

    E = trait_scores["Extraversion"]
    A = trait_scores["Agreeableness"]
    C = trait_scores["Conscientiousness"]
    S = trait_scores["Emotional Stability"]
    O = trait_scores["Openness"]
    def hi(x): return (not math.isnan(x)) and x >= 65
    def lo(x): return (not math.isnan(x)) and x <= 35

    if hi(C) and hi(A) and hi(S):
        archetype = "Steady Builder"
    elif hi(C) and hi(O) and lo(A):
        archetype = "Challenging Improver"
    elif hi(E) and hi(A) and hi(O):
        archetype = "Energizing Connector"
    elif lo(S) and hi(C):
        archetype = "High-standards Worrier"
    elif hi(O) and lo(C):
        archetype = "Creative Free-Spirit"
    else:
        archetype = "Balanced Profile"

    return {"traits": trait_scores, "archetype": archetype, "table": pd.DataFrame(rows)}

def score_leadership(responses: Dict[str, int]) -> Dict[str, Any]:
    dims = {"Structure": [], "Adapt": [], "Coach": [], "People": [], "Results": []}
    rows = []
    for item_id, text, dim, reverse in LEADERSHIP_ITEMS:
        raw = responses.get(item_id)
        if raw is None:
            continue
        scored = 6 - raw if reverse else raw
        dims[dim].append(scored)
        rows.append({"id": item_id, "dim": dim, "raw": raw, "scored": scored, "reverse": reverse})
    dim_scores = {k: pct(float(np.mean(v))) if v else float("nan") for k, v in dims.items()}

    people = dim_scores["People"]
    results = dim_scores["Results"]
    coach = dim_scores["Coach"]
    structure = dim_scores["Structure"]
    adapt = dim_scores["Adapt"]

    if people >= 60 and results >= 60:
        style = "High-Standards Coach"
    elif people >= 60 and results < 60:
        style = "People-First Supporter"
    elif results >= 60 and people < 60:
        style = "Results-First Driver"
    else:
        style = "Balanced Operator"

    modifiers = []
    if coach >= 65:
        modifiers.append("Coaching-oriented")
    if structure >= 65:
        modifiers.append("Structure-focused")
    if adapt >= 65:
        modifiers.append("Adaptive")

    return {"dims": dim_scores, "style": style, "modifiers": modifiers, "table": pd.DataFrame(rows)}

# -----------------------------
# Additional scoring + psychometrics helpers (v0.2)
# -----------------------------

DATA_DIR = "data"
LOG_PATH = os.path.join(DATA_DIR, "assessments.jsonl")

def ensure_data_dir() -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        # In hosted / read-only environments, logging may be unavailable.
        pass

def stable_hash(text: str) -> str:
    """Stable SHA-256 short hash for pseudonymisation."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]

def append_assessment_log(record: Dict[str, Any]) -> None:
    """Append one assessment record as JSONL (best-effort)."""
    ensure_data_dir()
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_assessment_logs(max_rows: int = 5000) -> List[Dict[str, Any]]:
    """Load recent JSONL logs (best-effort)."""
    if not os.path.exists(LOG_PATH):
        return []
    rows = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-max_rows:]
    except Exception:
        return []

def score_type_lens(responses: Dict[str, int]) -> Dict[str, Any]:
    buckets: Dict[int, List[int]] = {i: [] for i in range(1, 10)}
    rows = []
    for item_id, text, tnum, reverse in TYPE_ITEMS:
        raw = responses.get(item_id)
        if raw is None:
            continue
        scored = 6 - raw if reverse else raw
        buckets[tnum].append(scored)
        rows.append({"id": item_id, "type": tnum, "raw": raw, "scored": scored, "reverse": reverse})

    type_scores = {tnum: (pct(float(np.mean(vals))) if vals else float("nan")) for tnum, vals in buckets.items()}
    ranked = sorted([(k, v) for k, v in type_scores.items() if not math.isnan(v)], key=lambda x: x[1], reverse=True)

    top_type = ranked[0][0] if ranked else None
    top_score = ranked[0][1] if ranked else float("nan")
    second = ranked[1] if len(ranked) > 1 else None

    # Wing heuristic: adjacent types
    wing = None
    if top_type:
        adj = [((top_type - 2) % 9) + 1, (top_type % 9) + 1]  # wrap-around neighbors
        best_adj = None
        best_adj_score = -1
        for a in adj:
            s = type_scores.get(a, float("nan"))
            if not math.isnan(s) and s > best_adj_score:
                best_adj = a
                best_adj_score = s
        wing = best_adj

    blend = False
    if second and abs(top_score - second[1]) <= 5:
        blend = True

    # Compatibility heuristics
    friction = []
    synergy = []
    if top_type:
        friction = TYPE_COMPATIBILITY.get(top_type, {}).get("likely_friction", [])
        synergy = TYPE_COMPATIBILITY.get(top_type, {}).get("likely_synergy", [])

    tips = TYPE_MANAGEMENT_TIPS.get(top_type, {}) if top_type else {}

    return {
        "type_scores": {TYPE_LABELS[k]: v for k, v in type_scores.items()},
        "top_type_num": top_type,
        "top_type_label": TYPE_LABELS.get(top_type, "-") if top_type else "-",
        "wing_num": wing,
        "wing_label": TYPE_LABELS.get(wing, "-") if wing else "-",
        "blend": blend,
        "likely_friction_types": [TYPE_LABELS[t] for t in friction],
        "likely_synergy_types": [TYPE_LABELS[t] for t in synergy],
        "tips": tips,
        "table": pd.DataFrame(rows),
    }

def score_mgmt_preferences(responses: Dict[str, int]) -> Dict[str, Any]:
    dims: Dict[str, List[int]] = {d: [] for d in MGMT_PREF_DIMS}
    rows = []
    for item_id, text, dim, reverse in MGMT_PREF_ITEMS:
        raw = responses.get(item_id)
        if raw is None:
            continue
        scored = 6 - raw if reverse else raw
        dims[dim].append(scored)
        rows.append({"id": item_id, "dim": dim, "raw": raw, "scored": scored, "reverse": reverse})

    dim_scores = {k: pct(float(np.mean(v))) if v else float("nan") for k, v in dims.items()}

    # Qualitative interpretations
    def pref_line(dim: str, hi_text: str, lo_text: str) -> str:
        s = dim_scores.get(dim, float("nan"))
        if math.isnan(s):
            return ""
        return hi_text if s >= 65 else (lo_text if s <= 35 else f"{dim}: flexible / situational")

    interpretations = [
        pref_line("Autonomy", "Autonomy: thrives with ownership, outcome-based goals, and freedom in method.",
                  "Autonomy: prefers close guidance, clear steps, and frequent alignment."),
        pref_line("Structure", "Structure: prefers clear SOPs, checklists, written expectations, and predictability.",
                  "Structure: prefers minimal rules and more improvisation."),
        pref_line("Feedback", "Feedback: wants direct, specific feedback and rapid course-correction.",
                  "Feedback: prefers gentle feedback and time to process."),
        pref_line("Recognition", "Recognition: is energised by public acknowledgement and visible wins.",
                  "Recognition: prefers private recognition and low spotlight."),
        pref_line("Pace", "Pace: enjoys fast tempo, stretch goals, and frequent change.",
                  "Pace: prefers steady tempo, fewer urgencies, and deeper focus."),
        pref_line("Collaboration", "Collaboration: prefers teamwork, co-creation, and frequent interaction.",
                  "Collaboration: prefers independent work and protected focus time."),
        pref_line("Change", "Change: enjoys experimentation and new tools/processes.",
                  "Change: prefers stability; change should be paced and explained."),
        pref_line("Conflict", "Conflict: addresses issues directly and early.",
                  "Conflict: prefers cooling-off time and structured, calmer conversations."),
    ]
    interpretations = [i for i in interpretations if i]

    # “Manager cheat sheet”
    manager_cheat = []
    if not math.isnan(dim_scores.get("Autonomy", float("nan"))) and dim_scores["Autonomy"] >= 65:
        manager_cheat.append("Give outcomes, not micromanagement; agree on checkpoints.")
    if not math.isnan(dim_scores.get("Structure", float("nan"))) and dim_scores["Structure"] >= 65:
        manager_cheat.append("Provide SOPs/checklists and clarify escalation paths.")
    if not math.isnan(dim_scores.get("Feedback", float("nan"))) and dim_scores["Feedback"] >= 65:
        manager_cheat.append("Use direct, specific, behavior-based feedback; keep it timely.")
    if not math.isnan(dim_scores.get("Feedback", float("nan"))) and dim_scores["Feedback"] <= 35:
        manager_cheat.append("Use supportive framing; ask permission before tough feedback.")
    if not math.isnan(dim_scores.get("Pace", float("nan"))) and dim_scores["Pace"] <= 35:
        manager_cheat.append("Protect focus time; avoid last-minute surprises where possible.")
    if not math.isnan(dim_scores.get("Recognition", float("nan"))) and dim_scores["Recognition"] >= 65:
        manager_cheat.append("Celebrate wins visibly; tie recognition to values and standards.")
    if not math.isnan(dim_scores.get("Recognition", float("nan"))) and dim_scores["Recognition"] <= 35:
        manager_cheat.append("Praise privately; avoid putting them on the spot.")
    if not math.isnan(dim_scores.get("Collaboration", float("nan"))) and dim_scores["Collaboration"] <= 35:
        manager_cheat.append("Minimise interruptions; use async updates and clear handovers.")

    return {"dims": dim_scores, "interpretations": interpretations, "manager_cheat_sheet": manager_cheat, "table": pd.DataFrame(rows)}

def score_conflict_style(responses: Dict[str, str]) -> Dict[str, Any]:
    counts = {s: 0 for s in CONFLICT_STYLES}
    rows = []
    for item in CONFLICT_ITEMS:
        ans = responses.get(item["id"])
        if ans in ("A", "B"):
            style = item[ans]["style"]
            counts[style] += 1
            rows.append({"id": item["id"], "choice": ans, "style": style})
        else:
            rows.append({"id": item["id"], "choice": "-", "style": "-"})

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    primary = ranked[0][0] if ranked else "-"
    secondary = ranked[1][0] if len(ranked) > 1 else "-"

    summary_map = {
        "Avoiding": "Tends to pause, cool off, and revisit later; can prevent escalation but may delay resolution.",
        "Accommodating": "Tends to prioritise relationship/harmony; helpful for service recovery but can build resentment if overused.",
        "Competing": "Tends to push for a firm outcome; useful in emergencies but can feel harsh if tone isn’t managed.",
        "Compromising": "Tends to seek a workable middle ground quickly; efficient but may miss deeper root causes.",
        "Collaborating": "Tends to explore needs and co-design solutions; strongest for long-term trust but takes time.",
    }

    return {
        "counts": counts,
        "primary": primary,
        "secondary": secondary,
        "summary": summary_map.get(primary, ""),
        "table": pd.DataFrame(rows),
    }

# --- Classical Test Theory (CTT) utilities (requires multiple respondents)
def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Cronbach's alpha for persons x items matrix."""
    df = df_items.dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2 or df.shape[0] < 3:
        return float("nan")
    item_var = df.var(axis=0, ddof=1).sum()
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var / total_var))

def item_total_correlations(df_items: pd.DataFrame) -> Dict[str, float]:
    df = df_items.dropna(axis=0, how="any")
    if df.shape[0] < 5:
        return {}
    totals = df.sum(axis=1)
    out = {}
    for col in df.columns:
        out[col] = corr(df[col].to_numpy(), (totals - df[col]).to_numpy())
    return out

def ctt_sem(sd: float, reliability: float) -> float:
    if math.isnan(sd) or math.isnan(reliability):
        return float("nan")
    return float(sd * math.sqrt(max(0.0, 1.0 - reliability)))

# --- Lightweight 2PL IRT (dichotomous) via simple JML (for exploration only)
def irt_2pl_jml(X: np.ndarray, n_iter: int = 60, lr: float = 0.05) -> Dict[str, Any]:
    """Very small-sample 2PL estimation (exploratory).

    X: persons x items binary matrix (0/1), no missing.
    Returns item discrimination 'a', difficulty 'b', and person theta.
    """
    n, m = X.shape
    if n < 30 or m < 5:
        return {"a": None, "b": None, "theta": None, "note": "Need more data for stable IRT estimates."}

    # Initialise
    theta = np.zeros(n)
    a = np.ones(m)
    b = np.zeros(m)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    for _ in range(n_iter):
        # Update theta
        for i in range(n):
            z = a * (theta[i] - b)
            p = sigmoid(z)
            grad = np.sum(a * (X[i] - p))
            theta[i] += lr * grad

        # Update a and b
        for j in range(m):
            z = a[j] * (theta - b[j])
            p = sigmoid(z)
            # gradients
            grad_b = np.sum(-a[j] * (X[:, j] - p))
            grad_a = np.sum((theta - b[j]) * (X[:, j] - p))
            b[j] -= lr * grad_b
            a[j] += lr * grad_a
            a[j] = float(clamp(a[j], 0.2, 3.0))

    return {"a": a, "b": b, "theta": theta, "note": "Exploratory only (JML can be biased in small samples)."}

# --- G-Theory (person x rater x occasion) calculator from long-format data
def gtheory_prt(data: pd.DataFrame, person_col: str, rater_col: str, occasion_col: str, score_col: str) -> Dict[str, Any]:
    """Compute variance components for p x r x t design (random facets) using ANOVA method.
    Expects balanced or near-balanced data; results are approximate if very unbalanced.
    """
    df = data[[person_col, rater_col, occasion_col, score_col]].dropna()
    if df.empty:
        return {"error": "No data"}
    p = df[person_col].nunique()
    r = df[rater_col].nunique()
    t = df[occasion_col].nunique()
    if min(p, r, t) < 2:
        return {"error": "Need at least 2 persons, raters, and occasions"}

    grand = df[score_col].mean()

    # Means
    mp = df.groupby(person_col)[score_col].mean()
    mr = df.groupby(rater_col)[score_col].mean()
    mt = df.groupby(occasion_col)[score_col].mean()

    mpr = df.groupby([person_col, rater_col])[score_col].mean()
    mpt = df.groupby([person_col, occasion_col])[score_col].mean()
    mrt = df.groupby([rater_col, occasion_col])[score_col].mean()

    # Sum squares (approx for unbalanced)
    ss_p = r * t * ((mp - grand) ** 2).sum()
    ss_r = p * t * ((mr - grand) ** 2).sum()
    ss_t = p * r * ((mt - grand) ** 2).sum()

    # Interaction SS (centered)
    ss_pr = t * ((mpr - mpr.groupby(level=0).transform("mean") - mpr.groupby(level=1).transform("mean") + grand) ** 2).sum()
    ss_pt = r * ((mpt - mpt.groupby(level=0).transform("mean") - mpt.groupby(level=1).transform("mean") + grand) ** 2).sum()
    ss_rt = p * ((mrt - mrt.groupby(level=0).transform("mean") - mrt.groupby(level=1).transform("mean") + grand) ** 2).sum()

    # Residual SS
    # expand expected cell mean
    cell = df.groupby([person_col, rater_col, occasion_col])[score_col].transform("mean")
    ss_prt = ((df[score_col] - cell) ** 2).sum()

    # Degrees of freedom
    df_p = p - 1
    df_r = r - 1
    df_t = t - 1
    df_pr = df_p * df_r
    df_pt = df_p * df_t
    df_rt = df_r * df_t
    df_prt = df_p * df_r * df_t  # assumes 1 obs per cell

    ms_p = ss_p / df_p
    ms_r = ss_r / df_r
    ms_t = ss_t / df_t
    ms_pr = ss_pr / max(df_pr, 1)
    ms_pt = ss_pt / max(df_pt, 1)
    ms_rt = ss_rt / max(df_rt, 1)
    ms_prt = ss_prt / max(df_prt, 1)

    # Variance components (random effects) for balanced design
    var_prt = ms_prt
    var_pr = max(0.0, (ms_pr - ms_prt) / t)
    var_pt = max(0.0, (ms_pt - ms_prt) / r)
    var_rt = max(0.0, (ms_rt - ms_prt) / p)
    var_p = max(0.0, (ms_p - ms_pr - ms_pt + ms_prt) / (r * t))
    var_r = max(0.0, (ms_r - ms_pr - ms_rt + ms_prt) / (p * t))
    var_t = max(0.0, (ms_t - ms_pt - ms_rt + ms_prt) / (p * r))

    # Generalizability coefficient for relative decisions (example: average across r and t)
    error_rel = (var_pr / r) + (var_pt / t) + (var_prt / (r * t))
    g_coeff = var_p / (var_p + error_rel) if (var_p + error_rel) > 0 else float("nan")

    return {
        "p": p, "r": r, "t": t,
        "variance_components": {
            "person": var_p,
            "rater": var_r,
            "occasion": var_t,
            "person×rater": var_pr,
            "person×occasion": var_pt,
            "rater×occasion": var_rt,
            "residual": var_prt,
        },
        "g_coefficient_relative": g_coeff,
        "note": "Approximate ANOVA components; best with balanced data."
    }

# -----------------------------
# Reporting helpers
# -----------------------------

BEHAVIOR_COACHING_TIPS = {
    "Integrity & ethical consistency": [
        "Give clear standards and examples; reinforce honest reporting early.",
        "Use 'no-blame, facts-first' incident reviews to reduce fear of speaking up.",
    ],
    "Respectful communication & civility": [
        "Set meeting norms (no interruptions, respectful tone) and enforce consistently.",
        "Address disrespect quickly and privately; separate behavior from identity.",
    ],
    "Dependability & follow-through": [
        "Use clear deadlines and definition of done; ask for early renegotiation if blocked.",
        "Create simple checklists for repeatable tasks.",
    ],
    "Information sharing & transparency by default": [
        "Agree on handover standards (what must be shared, by when, where documented).",
        "Model transparency: share context and decisions, not only outcomes.",
    ],
    "Coachability through active feedback seeking": [
        "Use short feedback loops (weekly) with one focus behavior at a time.",
        "Ask the person to summarize takeaways and define a follow-up action.",
    ],
    "Resilience & emotional self-regulation under stress": [
        "Reduce overload where possible; clarify priorities during peaks.",
        "Teach micro-reset habits (pause, breathe, clarify next action) during pressure moments.",
    ],
}

def management_recommendations(behavior_table: pd.DataFrame) -> List[str]:
    if behavior_table is None or len(behavior_table) == 0:
        return []
    df = behavior_table.dropna(subset=["score_0_100"]).sort_values("score_0_100", ascending=True).head(4)
    recs = []
    for _, row in df.iterrows():
        name = row["behavior"]
        tips = BEHAVIOR_COACHING_TIPS.get(name, [])
        if tips:
            recs.append(f"**{name}**: {tips[0]}")
    if not recs:
        recs.append("Use the watch-out behaviours as coaching priorities: define 1–2 target behaviours and review weekly.")
    return recs

def make_pdf_report(result: Dict[str, Any], candidate: str, role: str, notes: str = "") -> bytes:
    """Generate a polished PDF report that mirrors the Streamlit dashboard as closely as possible.

    - Uses colour + clean tables for readability.
    - If detailed tables are available (either via underscore keys in-session, or result["tables"] from Supabase),
      they are included in the PDF.
    """
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        KeepTogether,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.colors import HexColor
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm

    def _safe_text(x: Any, fallback: str = "—") -> str:
        if x is None:
            return fallback
        try:
            s = str(x).strip()
        except Exception:
            return fallback
        return s if s else fallback
    def _ensure_list(x: Any) -> List[str]:
        """Coerce x into a list of strings (safe for slicing/iteration in PDFs)."""
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i) for i in x if str(i).strip()]
        if isinstance(x, (tuple, set)):
            return [str(i) for i in list(x) if str(i).strip()]
        if isinstance(x, dict):
            out: List[str] = []
            for v in x.values():
                if v is None:
                    continue
                if isinstance(v, (list, tuple, set)):
                    out.extend([str(i) for i in v if str(i).strip()])
                else:
                    s = str(v).strip()
                    if s:
                        out.append(s)
            return out
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            if "\n" in s:
                return [ln.strip() for ln in s.splitlines() if ln.strip()]
            return [s]
        # last-resort: try iterable
        try:
            return [str(i) for i in list(x) if str(i).strip()]
        except Exception:
            s = str(x).strip()
            return [s] if s else []


    def _fmt_pct(x: Any) -> str:
        try:
            if x is None:
                return "—"
            if isinstance(x, str):
                x = float(x)
            if isinstance(x, (int, float)) and (math.isnan(x) if isinstance(x, float) else False):
                return "—"
            return f"{float(x):.1f}"
        except Exception:
            return "—"

    def _band_from_val(v: Any) -> str:
        try:
            if v is None:
                return "-"
            f = float(v)
            if math.isnan(f):
                return "-"
            return band(f)
        except Exception:
            return "-"

    # --- brand-ish palette (warm + readable)
    COL_CHARCOAL = HexColor("#1f1f1f")
    COL_COPPER   = HexColor("#B36A4C")
    COL_GOLD     = HexColor("#D7B56D")
    COL_SAND     = HexColor("#E8D9C6")
    COL_SAGE     = HexColor("#6F8F7B")
    COL_LIGHT    = HexColor("#F7F4F0")
    COL_GRID     = HexColor("#DDDDDD")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="BEA Report",
        author="Vervio",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=20, leading=24, textColor=rl_colors.white, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="TitleSub", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=10, leading=13, textColor=rl_colors.white, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                              fontSize=13, leading=16, textColor=COL_CHARCOAL, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=11, leading=14, textColor=COL_CHARCOAL, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=9.5, leading=12, textColor=COL_CHARCOAL))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=8.2, leading=10.5, textColor=COL_CHARCOAL))
    styles.add(ParagraphStyle(name="VervioBullet", parent=styles["Normal"], fontName="Helvetica",
                              fontSize=9.2, leading=12, leftIndent=12, bulletIndent=6,
                              textColor=COL_CHARCOAL))

    story: List[Any] = []

    # ---- header banner
    org = _safe_text(result.get("org") or (result.get("config_json") or {}).get("org_name") or (result.get("context") or {}).get("org"))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    header_tbl = Table(
        [[
            Paragraph("BEA Assessment Report", styles["TitleBig"]),
        ]],
        colWidths=[doc.width],
    )
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_COPPER),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(header_tbl)

    header_tbl2 = Table(
        [[Paragraph(f"Organisation: <b>{org}</b>  •  Generated: <b>{generated}</b>  •  Version: <b>{VERSION}</b>", styles["TitleSub"])]],
        colWidths=[doc.width],
    )
    header_tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_CHARCOAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_tbl2)
    story.append(Spacer(1, 10))

    # ---- metadata card
    meta = [
        [Paragraph("<b>Candidate / Staff</b>", styles["Small"]), Paragraph(_safe_text(candidate), styles["Small"]),
         Paragraph("<b>Role</b>", styles["Small"]), Paragraph(_safe_text(role), styles["Small"])],
        [Paragraph("<b>Date (assessment)</b>", styles["Small"]), Paragraph(_safe_text(result.get("date")), styles["Small"]),
         Paragraph("<b>Reference</b>", styles["Small"]), Paragraph(_safe_text(result.get("reference") or result.get("invite_id") or ""), styles["Small"])],
    ]
    meta_tbl = Table(meta, colWidths=[30*mm, 60*mm, 20*mm, doc.width - (30+60+20)*mm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, COL_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, COL_GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    # ---- headline scores
    beh = result.get("behavioral", {}) or {}
    val = result.get("values", {}) or {}
    ei  = result.get("ei", {}) or {}

    beh_score = beh.get("overall")
    val_score = val.get("fit_0_100")
    ei_score  = ei.get("overall")

    score_data = [[
        Paragraph("<b>Behavioural alignment</b>", styles["Small"]),
        Paragraph(f"<b>{_fmt_pct(beh_score)}</b>/100<br/><font color='#{COL_SAGE.hexval()[2:]}'>{_safe_text(beh.get('band') or _band_from_val(beh_score))}</font>", styles["Small"]),
        Paragraph("<b>Values congruence</b>", styles["Small"]),
        Paragraph(f"<b>{_fmt_pct(val_score)}</b>/100<br/><font color='#{COL_SAGE.hexval()[2:]}'>{_safe_text(val.get('band') or _band_from_val(val_score))}</font>", styles["Small"]),
        Paragraph("<b>EI (SJT)</b>", styles["Small"]),
        Paragraph(f"<b>{_fmt_pct(ei_score)}</b>/100<br/><font color='#{COL_SAGE.hexval()[2:]}'>{_safe_text(ei.get('band') or _band_from_val(ei_score))}</font>", styles["Small"]),
    ]]
    score_tbl = Table(score_data, colWidths=[35*mm, 25*mm, 28*mm, 25*mm, 18*mm, doc.width - (35+25+28+25+18)*mm])
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_SAND),
        ("BOX", (0, 0), (-1, -1), 0.6, COL_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, COL_GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(score_tbl)
    story.append(Spacer(1, 12))

    # ---- helper: extract DataFrame tables
    def _df_from(key_underscore: str, key_tables: str) -> pd.DataFrame | None:
        # in-session
        if key_underscore in result and isinstance(result[key_underscore], pd.DataFrame):
            return result[key_underscore].copy()
        # from Supabase JSON
        tables = result.get("tables") or {}
        if isinstance(tables, dict) and key_tables in tables and isinstance(tables[key_tables], list):
            try:
                return pd.DataFrame(tables[key_tables])
            except Exception:
                return None
        return None

    def _add_section(title: str):
        bar = Table([[Paragraph(title, ParagraphStyle("SecTitle", parent=styles["Body"], fontName="Helvetica-Bold",
                                                    fontSize=10.5, textColor=rl_colors.white))]],
                    colWidths=[doc.width])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COL_CHARCOAL),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(Spacer(1, 6))
        story.append(bar)
        story.append(Spacer(1, 6))

    def _bullet_list(items: List[str]):
        for it in items:
            story.append(Paragraph(f"• {it}", styles["Body"]))

    def _table_from_df(df: pd.DataFrame, columns: List[Tuple[str, str]], col_widths: List[float], number_cols: set[str] | None = None):
        number_cols = number_cols or set()
        data = []
        header = [Paragraph(f"<b>{title}</b>", styles["Small"]) for _, title in columns]
        data.append(header)
        for _, row in df.iterrows():
            out_row = []
            for col, _title in columns:
                v = row.get(col, "")
                if col in number_cols:
                    try:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            s = "—"
                        else:
                            s = f"{float(v):.1f}"
                    except Exception:
                        s = _safe_text(v)
                    out_row.append(Paragraph(s, styles["Small"]))
                else:
                    out_row.append(Paragraph(_safe_text(v), styles["Small"]))
            data.append(out_row)

        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COL_COPPER),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("GRID", (0, 0), (-1, -1), 0.25, COL_GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        # zebra stripes
        for r in range(1, len(data)):
            if r % 2 == 0:
                t.setStyle(TableStyle([("BACKGROUND", (0, r), (-1, r), COL_LIGHT)]))
        story.append(t)
        story.append(Spacer(1, 8))

    # ---- Response quality
    _add_section("Response quality flags (interpretation cautions)")
    q = result.get("quality", {}) or {}
    duration = q.get("duration_sec")
    if duration is not None:
        story.append(Paragraph(f"Completion time: <b>{float(duration):.0f}</b> seconds", styles["Body"]))
        story.append(Spacer(1, 4))
    warnings = q.get("warnings") or []
    if warnings:
        for w in warnings:
            story.append(Paragraph(f"• <font color='#{COL_COPPER.hexval()[2:]}'>{_safe_text(w)}</font>", styles["Body"]))
    else:
        story.append(Paragraph("• No warning flags triggered.", styles["Body"]))
    story.append(Spacer(1, 6))

    # ---- Behavioural details
    if "behavioral" in result:
        _add_section("Behavioural alignment details")
        if beh.get("top_strengths"):
            strengths = [f"{r['behavior']} ({float(r['score_0_100']):.0f})" for r in beh.get("top_strengths", []) if "behavior" in r]
            if strengths:
                story.append(Paragraph("<b>Top strengths</b>", styles["Body"]))
                _bullet_list(strengths)
                story.append(Spacer(1, 4))
        if beh.get("watchouts"):
            watch = [f"{r['behavior']} ({float(r['score_0_100']):.0f})" for r in beh.get("watchouts", []) if "behavior" in r]
            if watch:
                story.append(Paragraph("<b>Watch-outs</b>", styles["Body"]))
                _bullet_list(watch)
                story.append(Spacer(1, 6))

        dfb = _df_from("_behavior_table", "behavior_table")
        if dfb is not None and len(dfb):
            # Management recommendations based on lowest behaviours
            try:
                recs = management_recommendations(dfb)
            except Exception:
                recs = []
            if recs:
                story.append(Paragraph("<b>Coaching priorities</b>", styles["Body"]))
                for r in recs:
                    story.append(Paragraph("• " + r.replace("**", ""), styles["Body"]))
                story.append(Spacer(1, 6))

            # Full table
            cols = [("behavior", "Behaviour"), ("core", "Core"), ("score_0_100", "Score"), ("weight", "Weight")]
            col_widths = [doc.width*0.52, doc.width*0.10, doc.width*0.18, doc.width*0.20]
            _table_from_df(dfb, cols, col_widths, number_cols={"score_0_100", "weight"})

    # ---- Values details
    if "values" in result:
        _add_section("Values congruence details")
        if val.get("most_misaligned"):
            mmis = [f"{r.get('value','')} (gap {float(r.get('gap',0)):+.1f})" for r in val.get("most_misaligned", [])]
            if mmis:
                story.append(Paragraph("<b>Most misaligned values</b> (largest gaps)", styles["Body"]))
                _bullet_list(mmis)
                story.append(Spacer(1, 6))

        dfv = _df_from("_values_table", "values_table")
        if dfv is not None and len(dfv):
            cols = [("value", "Value"), ("candidate", "Candidate"), ("org", "Org"), ("gap", "Gap")]
            col_widths = [doc.width*0.46, doc.width*0.18, doc.width*0.18, doc.width*0.18]
            _table_from_df(dfv, cols, col_widths, number_cols={"candidate", "org", "gap"})

    # ---- EI details
    if "ei" in result:
        _add_section("Emotional intelligence (SJT) details")
        branches = ei.get("branches") or {}
        if isinstance(branches, dict) and branches:
            topb = sorted(branches.items(), key=lambda x: (float(x[1]) if x[1] is not None else -1), reverse=True)
            txt = ", ".join([f"{k}: {float(v):.0f}" for k, v in topb if v is not None and not (isinstance(v, float) and math.isnan(v))])
            if txt:
                story.append(Paragraph("<b>Branch scores</b>: " + txt, styles["Body"]))
                story.append(Spacer(1, 6))

        dfe = _df_from("_ei_table", "ei_table")
        if dfe is not None and len(dfe):
            cols = [("id", "Item"), ("branch", "Branch"), ("answer", "Answer"), ("credit", "Credit")]
            col_widths = [doc.width*0.14, doc.width*0.36, doc.width*0.18, doc.width*0.32]
            _table_from_df(dfe, cols, col_widths, number_cols={"credit"})

    # ---- Big Five
    if "personality" in result:
        _add_section("Personality (Big Five) details")
        archetype = _safe_text((result.get("personality") or {}).get("archetype"))
        story.append(Paragraph(f"Archetype: <b>{archetype}</b>", styles["Body"]))
        traits = (result.get("personality") or {}).get("traits") or {}
        if isinstance(traits, dict) and traits:
            df_traits = pd.DataFrame([{"trait": k, "score_0_100": v} for k, v in traits.items()])
            cols = [("trait", "Trait"), ("score_0_100", "Score")]
            col_widths = [doc.width*0.65, doc.width*0.35]
            _table_from_df(df_traits, cols, col_widths, number_cols={"score_0_100"})

    # ---- Leadership
    if "leadership" in result:
        _add_section("Leadership style details")
        l = result.get("leadership") or {}
        story.append(Paragraph(f"Style: <b>{_safe_text(l.get('style'))}</b>", styles["Body"]))
        dims = l.get("dims") or {}
        if isinstance(dims, dict) and dims:
            df_dims = pd.DataFrame([{"dimension": k, "score_0_100": v} for k, v in dims.items()])
            cols = [("dimension", "Dimension"), ("score_0_100", "Score")]
            col_widths = [doc.width*0.65, doc.width*0.35]
            _table_from_df(df_dims, cols, col_widths, number_cols={"score_0_100"})

    # ---- Type lens
    if "type_lens" in result:
        _add_section("Personality type lens (Type 1–9)")
        tl = result.get("type_lens") or {}
        story.append(Paragraph(f"Top type: <b>{_safe_text(tl.get('top_type'))}</b>  •  Wing: <b>{_safe_text(tl.get('wing'))}</b>", styles["Body"]))
        synergy = _ensure_list(tl.get("likely_synergy"))
        if synergy:
            story.append(Paragraph("<b>Likely synergy with</b>: " + ", ".join([_safe_text(x) for x in synergy]), styles["Body"]))
        friction = _ensure_list(tl.get("likely_friction"))
        if friction:
            story.append(Paragraph("<b>Likely friction with</b>: " + ", ".join([_safe_text(x) for x in friction]), styles["Body"]))
        tips = _ensure_list(tl.get("tips"))
        if tips:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Coaching tips</b>", styles["Body"]))
            for t in tips[:8]:
                story.append(Paragraph("• " + _safe_text(t), styles["Body"]))
        # Optional type table if present
        dft = _df_from("_type_table", "type_table")
        if dft is not None and len(dft):
            # table can be large; keep it compact
            cols = [("type", "Type"), ("score_0_100", "Score")]
            if "type" not in dft.columns and "label" in dft.columns:
                dft = dft.rename(columns={"label": "type"})
            if "score_0_100" not in dft.columns and "score" in dft.columns:
                dft = dft.rename(columns={"score": "score_0_100"})
            col_widths = [doc.width*0.65, doc.width*0.35]
            _table_from_df(dft[["type","score_0_100"]], cols, col_widths, number_cols={"score_0_100"})

    # ---- Management prefs
    if "management_prefs" in result:
        _add_section("Management preferences (how to manage me)")
        mp = result.get("management_prefs") or {}
        cheat = _ensure_list(mp.get("cheat_sheet"))
        if cheat:
            story.append(Paragraph("<b>Manager cheat sheet</b>", styles["Body"]))
            for c in cheat[:10]:
                story.append(Paragraph("• " + _safe_text(c), styles["Body"]))
            story.append(Spacer(1, 6))
        dims = mp.get("dims") or {}
        if isinstance(dims, dict) and dims:
            df_dims = pd.DataFrame([{"dimension": k, "score_0_100": v} for k, v in dims.items()]).sort_values("score_0_100", ascending=False)
            cols = [("dimension", "Dimension"), ("score_0_100", "Score")]
            col_widths = [doc.width*0.65, doc.width*0.35]
            _table_from_df(df_dims, cols, col_widths, number_cols={"score_0_100"})

    # ---- Conflict style
    if "conflict_style" in result:
        _add_section("Conflict style scenarios")
        cs = result.get("conflict_style") or {}
        story.append(Paragraph(f"Primary: <b>{_safe_text(cs.get('primary'))}</b>  •  Secondary: <b>{_safe_text(cs.get('secondary'))}</b>", styles["Body"]))
        if cs.get("summary"):
            story.append(Paragraph(_safe_text(cs.get("summary")), styles["Body"]))
            story.append(Spacer(1, 6))

        dfc = _df_from("_conflict_table", "conflict_table")
        if dfc is not None and len(dfc):
            cols = [("id", "Item"), ("choice", "Choice"), ("style", "Style")]
            col_widths = [doc.width*0.20, doc.width*0.20, doc.width*0.60]
            _table_from_df(dfc, cols, col_widths, number_cols=set())

    # ---- Examiner notes
    if notes and str(notes).strip():
        _add_section("Examiner notes")
        for ln in str(notes).strip().splitlines():
            story.append(Paragraph("• " + _safe_text(ln), styles["Body"]))

    # ---- footer disclaimer
    story.append(Spacer(1, 10))
    disclaimer = (
        "This report is a research-aligned decision-support output. "
        "Do not use as a sole hiring gatekeeper. Validate locally, monitor fairness/bias, "
        "and interpret patterns (not single micro-scores), especially when response-quality flags trigger."
    )
    story.append(Paragraph(disclaimer, ParagraphStyle("Disclaimer", parent=styles["Small"], textColor=HexColor("#555555"))))

    # Page numbers
    def _on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#666666"))
        canvas.drawRightString(doc_.pagesize[0] - 18*mm, 10*mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buffer.seek(0)
    return buffer.read()

# -----------------------------
# App pages
# -----------------------------

def page_home():
    st.title(APP_TITLE)
    st.caption(f"Version {VERSION} • Research-aligned prototype for behaviour, values-fit, EI scenarios, and leadership style.")

    st.info(
        "This is a **prototype** meant for structured decision support and development. "
        "If you plan to use it in hiring, run **local validation**, fairness/bias monitoring, and "
        "align to South African requirements (valid, reliable, fair, not biased) and privacy duties."
    )
    st.write("Choose a page from the sidebar to configure a role profile, run an assessment, or view research notes.")


def page_org_setup():
    st.header("Organisation & Role Setup")
    st.write("Set your organisation's values profile and the role-specific weighting for the 20 behaviours. "
             "When you generate a token invite, the **current settings can be snapshotted** into that invite "
             "(so candidates are scored against the correct parameters).")

    # Basic labels used in invites / reporting
    col0a, col0b = st.columns([2, 2])
    with col0a:
        org_name = st.text_input("Organisation name (label)", value=st.session_state.get("org_name", ""))
        st.session_state["org_name"] = org_name
    with col0b:
        default_role = st.text_input("Default role label (suggested)", value=st.session_state.get("default_role_label", "Candidate Role"))
        st.session_state["default_role_label"] = default_role

    st.divider()

    with st.expander("1) Organisation values profile (used for values-fit comparison)", expanded=True):
        org = st.session_state.get("org_values", {v: 4 for v in VALUES_DIMENSIONS})

        cols = st.columns(2)
        updated = {}
        for i, dim in enumerate(VALUES_DIMENSIONS):
            with cols[i % 2]:
                updated[dim] = st.slider(
                    dim,
                    min_value=1,
                    max_value=5,
                    value=int(org.get(dim, 4)),
                    help="1 = low emphasis, 5 = very strong emphasis in your culture.",
                )
        st.session_state["org_values"] = updated

        st.caption("Tip: Keep this realistic (what your culture actually rewards), not aspirational (what you wish it was).")

    with st.expander("2) Role weighting for the 20 behaviours (used for the behavioural alignment score)", expanded=True):
        weights = st.session_state.get("role_weights", {b["id"]: 1.0 for b in BEHAVIORS})

        st.write("Higher weights = the behaviour matters more for this role. "
                 "Neutral is **1.0**. Typical range is **0.5–2.0**.")
        cols = st.columns(2)
        updated_w = {}
        for i, beh in enumerate(BEHAVIORS):
            with cols[i % 2]:
                beh_id = beh["id"]
                label = f"{beh_id} — {beh['name']}"
                updated_w[beh_id] = st.slider(
                    label,
                    min_value=0.5,
                    max_value=2.0,
                    value=float(weights.get(beh_id, 1.0)),
                    step=0.1,
                )
        st.session_state["role_weights"] = updated_w

    st.divider()

    colr1, colr2, colr3 = st.columns([1, 1, 2])
    with colr1:
        if st.button("Reset values to neutral (4/5)"):
            st.session_state["org_values"] = {v: 4 for v in VALUES_DIMENSIONS}
            st.rerun()
    with colr2:
        if st.button("Reset weights to neutral (1.0)"):
            st.session_state["role_weights"] = {b["id"]: 1.0 for b in BEHAVIORS}
            st.rerun()
    with colr3:
        st.info("These settings affect scoring. For hiring use, snapshot settings into each invite from the Vervio Dashboard.")

    # Snapshot preview
    st.subheader("Snapshot preview (what gets stored on an invite)")
    preview = get_profile_snapshot(
        role_label=st.session_state.get("default_role_label", "Candidate Role"),
        org_name=st.session_state.get("org_name", ""),
    )
    # Don't show all weights in full (busy); show counts + a few examples
    st.write({
        "org_name": preview.get("org_name"),
        "role_label": preview.get("role_label"),
        "values_dimensions": len((preview.get("org_values") or {}).keys()),
        "behaviours_weighted": len((preview.get("role_weights") or {}).keys()),
        "example_weights": dict(list((preview.get("role_weights") or {}).items())[:5]),
        "modules": preview.get("modules"),
        "lock_modules": preview.get("lock_modules"),
        "version": preview.get("version"),
    })


def page_assessment(token_mode: bool = False, invite: Dict[str, Any] | None = None):
    st.header("Candidate Assessment")
    st.write("Complete modules. You can run fewer modules for development use, or include all for a richer profile.")

    with st.expander("Consent & use boundaries (recommended)", expanded=True):
        consent = st.checkbox(
            "I understand this is a structured assessment tool; results should be interpreted alongside other evidence, and stored/used responsibly.",
            value=False,
        )
        st.caption("Tip: For real use, add POPIA-ready consent text, retention periods, and candidate access rights.")
    if not consent:
        st.warning("Please tick the consent checkbox to proceed.")
        st.stop()

    
    # Token-mode config snapshot
    cfg: Dict[str, Any] = {}
    org_label = ""
    module_defaults = {
        "behaviour": True,
        "values": True,
        "ei_sjt": True,
        "big5": True,
        "leadership": True,
        "type_lens": True,
        "mgmt_prefs": True,
        "conflict": True,
    }
    lock_modules = False

    if token_mode and invite:
        cfg = invite.get("config_json") or {}
        apply_profile_snapshot(cfg)
        org_label = (invite.get("org") or cfg.get("org_name") or "").strip()
        try:
            mods = cfg.get("modules")
            if isinstance(mods, dict):
                for k in module_defaults:
                    if k in mods:
                        module_defaults[k] = bool(mods.get(k))
        except Exception:
            pass
        lock_modules = bool(cfg.get("lock_modules", True))


# In token mode (candidate link), role is fixed to the invite.
    if token_mode and invite:
        candidate_name = st.text_input("Your name (optional — helps the examiner interpret the report)", value="")
        if org_label:
            st.text_input("Organisation", value=org_label, disabled=True)
        role_name = st.text_input("Role / position", value=str(invite.get("role", "")), disabled=True)
        if cfg:
            st.caption(f"Assessment config captured: {cfg.get('captured_at', '?')} • version {cfg.get('version', '?')}")
        # For token mode we avoid local file logging; results are stored to Supabase (server-side).
        store_for_stats = False
        store_identifiable = False
        candidate_id = ""
    else:
        candidate_name = st.text_input("Candidate / staff member name (for report)", value="")
        role_name = st.text_input("Role / position being assessed", value="")
        store_for_stats = st.checkbox(
            "Store this assessment (pseudonymised) to build reliability/accuracy statistics over time",
            value=True,
        )
        store_identifiable = st.checkbox(
            "Include candidate/staff name in stored log (only if you have consent)",
            value=False,
        )
        candidate_id = st.text_input("Internal candidate/staff ID (optional, for test–retest tracking)", value="")


    start = st.session_state.get("start_time", None)
    if start is None:
        st.session_state["start_time"] = time.time()

    
    st.subheader("Modules")
    if token_mode and invite and lock_modules:
        st.caption("Modules are set by the examiner for standardisation.")
    colA, colB, colC, colD = st.columns(4)
    disable_mods = bool(token_mode and invite and lock_modules)
    with colA:
        use_beh = st.checkbox(
            "Behavioural alignment (20 behaviours)",
            value=bool(module_defaults.get("behaviour", True)),
            disabled=disable_mods,
        )
        use_values = st.checkbox(
            "Values congruence (P–O fit)",
            value=bool(module_defaults.get("values", True)),
            disabled=disable_mods,
        )
    with colB:
        use_ei = st.checkbox(
            "EI scenario test (SJT)",
            value=bool(module_defaults.get("ei_sjt", True)),
            disabled=disable_mods,
        )
        use_big5 = st.checkbox(
            "Personality profile (Big Five)",
            value=bool(module_defaults.get("big5", True)),
            disabled=disable_mods,
        )
    with colC:
        use_lead = st.checkbox(
            "Leadership style (for supervisors/managers)",
            value=bool(module_defaults.get("leadership", True)),
            disabled=disable_mods,
        )
        use_types = st.checkbox(
            "Personality type lens (Type 1–9)",
            value=bool(module_defaults.get("type_lens", True)),
            disabled=disable_mods,
        )
    with colD:
        use_mgmt_prefs = st.checkbox(
            "Management preferences & motivators",
            value=bool(module_defaults.get("mgmt_prefs", True)),
            disabled=disable_mods,
        )
        use_conflict = st.checkbox(
            "Conflict style scenarios (no right/wrong)",
            value=bool(module_defaults.get("conflict", True)),
            disabled=disable_mods,
        )
    st.divider()

    responses: Dict[str, Any] = {}

    if use_beh:
        st.subheader("1) Behavioural alignment")
        st.caption("Rate how often you demonstrate each behaviour at work (frequency-based).")
        for b in BEHAVIORS:
            st.markdown(f"**{b['name']}** {'(core)' if b['core'] else ''}")
            for i, item in enumerate(b["items"], start=1):
                key = f"{b['id']}_{i}"
                val = st.radio(item, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=key)
                responses[key] = int(val)
            st.write("")

    if use_values:
        st.subheader("2) Values profile")
        st.caption("Rate how important each value is to you in a workplace.")
        for v in VALUES_DIMENSIONS:
            key = f"V_{v}"
            val = st.select_slider(v, options=[1,2,3,4,5], value=3, format_func=lambda x: VALUES_SCALE_LABELS[x-1], key=key)
            responses[key] = int(val)

    if use_ei:
        st.subheader("3) Emotional intelligence scenarios (SJT)")
        st.caption("Choose the response that is *most effective* in the workplace context described.")
        for item in EI_SJT:
            key = item["id"]
            st.markdown(f"**{item['prompt']}**")
            choice = st.radio("Select an option", options=SJT_OPTIONS, horizontal=True, key=key)
            st.write(f"A) {item['options']['A']}\n\nB) {item['options']['B']}\n\nC) {item['options']['C']}\n\nD) {item['options']['D']}")
            responses[key] = choice
            st.write("")

    if use_big5:
        st.subheader("4) Personality profile (Big Five)")
        st.caption("Rate your typical tendencies at work.")
        for item_id, text, trait, reverse in BIG5_ITEMS:
            key = item_id
            val = st.radio(text, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=key)
            responses[key] = int(val)

    # A few general workplace items (help interpret results)
    st.caption("A few additional workplace items.")
    # Attention check (instructed response)
    val = st.radio(ATTENTION_CHECK_ITEMS[0][1], options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=ATTENTION_CHECK_ITEMS[0][0])
    responses[ATTENTION_CHECK_ITEMS[0][0]] = int(val)
    # Infrequency / improbable-virtue
    val = st.radio(INFREQUENCY_ITEM_TEXT[0][0] + ": " + INFREQUENCY_ITEM_TEXT[0][1], options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=INFREQUENCY_ITEM_TEXT[0][0])
    responses[INFREQUENCY_ITEM_TEXT[0][0]] = int(val)

    if use_lead:
        st.subheader("5) Leadership / management style")
        st.caption("Answer based on how you typically lead or prefer to lead (if applicable).")
        for item_id, text, dim, reverse in LEADERSHIP_ITEMS:
            key = item_id
            val = st.radio(text, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=key)
            responses[key] = int(val)

    if use_types:
        st.subheader("6) Personality type lens (Type 1–9)")
        st.caption("This module is a descriptive workstyle lens (not clinical). Use for coaching / team-fit conversations, not as a sole hiring gate.")
        for item_id, text, tnum, reverse in TYPE_ITEMS:
            val = st.radio(text, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=item_id)
            responses[item_id] = int(val)

        # Embedded quality checks (treated like normal items)
        val = st.radio(ATTENTION_CHECK_ITEMS[1][1], options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=ATTENTION_CHECK_ITEMS[1][0])
        responses[ATTENTION_CHECK_ITEMS[1][0]] = int(val)

        for inf in INFREQUENCY_ITEM_TEXT[1:]:
            val = st.radio(inf[0] + ": " + inf[1], options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=inf[0])
            responses[inf[0]] = int(val)

    if use_mgmt_prefs:
        st.subheader("7) Management preferences & motivators")
        st.caption("How you prefer to be managed and what conditions help you perform at your best.")
        for item_id, text, dim, reverse in MGMT_PREF_ITEMS:
            val = st.radio(text, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=item_id)
            responses[item_id] = int(val)

    if use_conflict:
        st.subheader("8) Conflict style scenarios")
        st.caption("No right/wrong answers. Choose the response that is most like you at work.")
        for item in CONFLICT_ITEMS:
            st.markdown(f"**{item['prompt']}**")
            choice = st.radio(
                "Choose A or B",
                options=["A", "B"],
                format_func=lambda x, it=item: it[x]["text"],
                horizontal=True,
                key=item["id"],
            )
            responses[item["id"]] = choice
            st.write("")

    st.divider()
    if st.button("Generate results"):
        duration = time.time() - st.session_state.get("start_time", time.time())

        beh_res = {k: v for k, v in responses.items() if k.startswith("B")}
        val_res = {k[2:]: v for k, v in responses.items() if k.startswith("V_")}
        ei_res = {k: v for k, v in responses.items() if k.startswith("E")}
        big5_res = {k: v for k, v in responses.items() if k.startswith("P")}
        lead_res = {k: v for k, v in responses.items() if k.startswith("L")}
        type_res = {k: v for k, v in responses.items() if k.startswith("T")}
        mgmt_res = {k: v for k, v in responses.items() if k.startswith("MP")}
        conflict_res = {k: v for k, v in responses.items() if k.startswith("C")}

        role_weights = st.session_state.get("role_weights", {b["id"]: 1.0 for b in BEHAVIORS})
        org_values = st.session_state.get("org_values", {v: 4 for v in VALUES_DIMENSIONS})

        results: Dict[str, Any] = {"candidate_name": candidate_name, "role": role_name, "org": st.session_state.get("org_name",""), "date": time.strftime("%Y-%m-%d")}

        if use_beh:
            beh = score_behavioral(beh_res, role_weights)
            results["behavioral"] = {
                "overall": beh["overall"],
                "core": beh["core"],
                "differentiators": beh["differentiators"],
                "band": band(beh["overall"]) if not math.isnan(beh["overall"]) else "-",
                "top_strengths": beh["top_strengths"],
                "watchouts": beh["watchouts"],
            }
            results["_behavior_table"] = beh["table"]
        if use_values:
            vf = score_values_fit(val_res, org_values)
            results["values"] = {
                "fit_corr": vf["fit_corr"],
                "fit_0_100": vf["fit_0_100"],
                "band": band(vf["fit_0_100"]),
                "most_misaligned": vf["most_misaligned"],
            }
            results["_values_table"] = vf["gap_table"]
        if use_ei:
            ei = score_ei_sjt(ei_res)
            results["ei"] = {"overall": ei["overall"], "band": band(ei["overall"]), "branches": ei["branches"]}
            results["_ei_table"] = ei["table"]
        if use_big5:
            p = score_big5(big5_res)
            results["personality"] = {"traits": p["traits"], "archetype": p["archetype"]}
            results["_big5_table"] = p["table"]
        if use_lead:
            l = score_leadership(lead_res)
            results["leadership"] = {"dims": l["dims"], "style": l["style"], "modifiers": l["modifiers"]}
            results["_lead_table"] = l["table"]

        if use_types:
            t = score_type_lens(type_res)
            results["type_lens"] = {
                "top_type": t["top_type_label"],
                "wing": t["wing_label"],
                "blend": t["blend"],
                "likely_friction": t["likely_friction_types"],
                "likely_synergy": t["likely_synergy_types"],
                "tips": t["tips"],
                "scores": t["type_scores"],
            }
            results["_type_table"] = t["table"]

        if use_mgmt_prefs:
            mp = score_mgmt_preferences(mgmt_res)
            results["management_prefs"] = {"dims": mp["dims"], "interpretations": mp["interpretations"], "cheat_sheet": mp["manager_cheat_sheet"]}
            results["_mgmt_table"] = mp["table"]

        if use_conflict:
            cf = score_conflict_style(conflict_res)
            results["conflict_style"] = {"primary": cf["primary"], "secondary": cf["secondary"], "summary": cf["summary"], "counts": cf["counts"]}
            results["_conflict_table"] = cf["table"]

        numeric_responses = {k: v for k, v in responses.items() if isinstance(v, int)}
        q = response_flags(numeric_responses, duration, inconsistency_pairs=INCONSISTENCY_PAIRS, attention_checks=ATTENTION_CHECKS, infrequency_items=INFREQUENCY_ITEMS)
        results["quality"] = q

        # Optional logging for psychometrics (best-effort, pseudonymised)
        if store_for_stats:
            ident_source = (candidate_id.strip() or candidate_name.strip() or str(uuid.uuid4()))
            record = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_hash": stable_hash(ident_source),
                "candidate_name": candidate_name.strip() if store_identifiable else "",
                "role": role_name.strip(),
                "modules": {
                    "behavioral": bool(use_beh),
                    "values": bool(use_values),
                    "ei_sjt": bool(use_ei),
                    "big5": bool(use_big5),
                    "leadership": bool(use_lead),
                    "type_lens": bool(use_types),
                    "management_prefs": bool(use_mgmt_prefs),
                    "conflict": bool(use_conflict),
                },
                "duration_sec": float(duration),
                "quality": results.get("quality", {}),
                "responses": responses,
                "scores": {k: v for k, v in results.items() if not k.startswith("_") and k not in ("quality",)},
            }
            append_assessment_log(record)

        st.session_state["latest_results"] = results

        # If this assessment was completed via a token link, store results in Supabase and close the loop.
        if token_mode and invite:
            if not supabase_available():
                st.error("Supabase is not configured on this deployment, so results cannot be submitted.")
                st.stop()

            invite_id = int(invite.get("id"))
            # Minimal 'headline' scores for quick dashboard viewing
            scores_json = {
                "behavioural_alignment": float(results.get("behavioral", {}).get("overall", float("nan"))) if "behavioral" in results else None,
                "values_congruence": float(results.get("values", {}).get("fit_0_100", float("nan"))) if "values" in results else None,
                "ei_sjt": float(results.get("ei", {}).get("overall", float("nan"))) if "ei" in results else None,
            }


            report_json = {k: v for k, v in results.items() if not k.startswith("_")}
            # Attach detailed tables (if present) so PDFs generated later can match the rich dashboard display.
            table_map = {
                "_behavior_table": "behavior_table",
                "_values_table": "values_table",
                "_ei_table": "ei_table",
                "_big5_table": "big5_table",
                "_lead_table": "lead_table",
                "_type_table": "type_table",
                "_mgmt_table": "mgmt_table",
                "_conflict_table": "conflict_table",
            }
            tables: Dict[str, Any] = {}
            for k, name in table_map.items():
                if k in results and isinstance(results[k], pd.DataFrame):
                    df = results[k].copy()
                    df = df.where(pd.notnull(df), None)  # JSON-safe
                    tables[name] = df.to_dict(orient="records")
            if tables:
                report_json["tables"] = tables
            try:
                sb_store_result(invite_id, scores_json=scores_json, report_json=report_json)
                sb_mark_invite_completed(invite_id)
                st.success("Submitted successfully. Thank you — you can close this tab.")
                st.stop()
            except Exception as e:
                st.error(f"Could not submit results: {e}")
                st.stop()

        st.success("Results generated. Go to 'Results & Report' in the sidebar to view and export.")

def page_results():
    st.header("Results & Report")
    results = st.session_state.get("latest_results")
    if not results:
        st.info("No results yet. Run an assessment first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if "behavioral" in results:
            st.metric("Behavioural alignment", f"{results['behavioral']['overall']:.1f}", results["behavioral"]["band"])
    with col2:
        if "values" in results:
            st.metric("Values congruence", f"{results['values']['fit_0_100']:.1f}", results["values"]["band"])
    with col3:
        if "ei" in results:
            st.metric("EI SJT", f"{results['ei']['overall']:.1f}", results["ei"]["band"])

    st.caption(f"Candidate: {results.get('candidate_name','')} • Role: {results.get('role','')} • Date: {results.get('date','')}")

    with st.expander("Response quality flags (interpretation cautions)", expanded=True):
        warnings = results.get("quality", {}).get("warnings", [])
        st.write(f"Completion time: {results.get('quality', {}).get('duration_sec', 0):.0f} seconds")
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("No warning flags triggered.")

    if "_behavior_table" in results:
        st.subheader("Behavioural alignment details")
        df = results["_behavior_table"].copy()
        df["band"] = df["score_0_100"].apply(lambda x: band(float(x)) if pd.notna(x) else "-")
        st.dataframe(df[["behavior","core","score_0_100","band","weight"]], use_container_width=True)

        st.markdown("**Suggested management focus (based on lowest behaviours):**")
        for r in management_recommendations(df):
            st.write("-", r)

    if "_values_table" in results:
        st.subheader("Values congruence details")
        st.write("Fit here is computed as a profile correlation (supplementary fit). Use gaps for coaching/onboarding discussion.")
        st.dataframe(results["_values_table"], use_container_width=True)

    if "_ei_table" in results:
        st.subheader("EI scenario details")
        st.dataframe(results["_ei_table"], use_container_width=True)
        st.json(results["ei"]["branches"])

    if "_big5_table" in results:
        st.subheader("Personality profile (Big Five)")
        traits = results["personality"]["traits"]
        st.json(traits)
        st.caption(f"Archetype label: {results['personality']['archetype']} (descriptive, not clinical).")

    if "_lead_table" in results:
        st.subheader("Leadership style")
        st.json({"style": results["leadership"]["style"], "modifiers": results["leadership"]["modifiers"], "dims": results["leadership"]["dims"]})

    if "_type_table" in results:
        st.subheader("Personality type lens (Type 1–9)")
        tl = results["type_lens"]
        st.write(f"Top type: **{tl['top_type']}**")
        if tl.get("wing") and tl.get("wing") != "-":
            st.write(f"Possible wing: {tl['wing']}")
        if tl.get("blend"):
            st.info("Top two types are close — consider this a blended profile rather than a single ‘type’.")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Likely strengths (type lens)**")
            for s in tl.get("tips", {}).get("strengths", []):
                st.write("-", s)
            st.markdown("**Watch-outs**")
            for w in tl.get("tips", {}).get("watchouts", []):
                st.write("-", w)
        with cols[1]:
            st.markdown("**How to manage / motivate**")
            for a in tl.get("tips", {}).get("manage", []):
                st.write("-", a)
            st.markdown("**Interpersonal fit heuristics**")
            for t in tl.get("likely_synergy", []):
                st.write("✅", t)
            for t in tl.get("likely_friction", []):
                st.write("⚠️", t)

        # Show type scores
        type_scores = pd.Series(tl.get("scores", {})).sort_values(ascending=False)
        st.dataframe(type_scores.to_frame("score_0_100"), use_container_width=True)

    if "_mgmt_table" in results:
        st.subheader("Management preferences & motivators")
        mp = results["management_prefs"]
        st.dataframe(pd.Series(mp["dims"]).sort_values(ascending=False).to_frame("score_0_100"), use_container_width=True)
        st.markdown("**Interpretation**")
        for line in mp.get("interpretations", []):
            st.write("-", line)
        st.markdown("**Manager cheat sheet**")
        for line in mp.get("cheat_sheet", []):
            st.write("-", line)

    if "_conflict_table" in results:
        st.subheader("Conflict style scenarios")
        cs = results["conflict_style"]
        st.write(f"Primary style: **{cs['primary']}** (secondary: {cs['secondary']})")
        st.write(cs.get("summary", ""))
        st.dataframe(pd.Series(cs.get("counts", {})).sort_values(ascending=False).to_frame("count"), use_container_width=True)

    with st.expander("Examiner summary (qualitative)", expanded=True):
        lines = []
        if "behavioral" in results:
            lines.append(f"- Behavioural alignment: {results['behavioral']['overall']:.1f}/100 ({results['behavioral']['band']}).")
            low = sorted(results["_behavior_table"].to_dict("records"), key=lambda r: r["score_0_100"])[:3] if "_behavior_table" in results else []
            if low:
                lows = ", ".join([f"{r['behavior']} ({r['score_0_100']:.0f})" for r in low])
                lines.append(f"  - Development focus: {lows}.")
        if "values" in results:
            lines.append(f"- Values congruence: {results['values']['fit_0_100']:.1f}/100 ({results['values']['band']}).")
        if "ei" in results:
            lines.append(f"- EI SJT: {results['ei']['overall']:.1f}/100 ({results['ei']['band']}).")
        if "personality" in results:
            lines.append(f"- Big Five archetype: {results['personality']['archetype']}.")
        if "leadership" in results:
            lines.append(f"- Leadership style: {results['leadership']['style']}.")
        if "type_lens" in results:
            tl = results["type_lens"]
            lines.append(f"- Type lens: {tl['top_type']} (wing: {tl.get('wing','-')}).")
            if tl.get("likely_friction"):
                lines.append("  - Likely friction with: " + "; ".join(tl["likely_friction"]) + ".")
            if tl.get("likely_synergy"):
                lines.append("  - Likely synergy with: " + "; ".join(tl["likely_synergy"]) + ".")
        if "management_prefs" in results:
            top = sorted(results["management_prefs"]["dims"].items(), key=lambda x: x[1], reverse=True)[:3]
            top_txt = ", ".join([f"{k} ({v:.0f})" for k, v in top if not math.isnan(v)])
            if top_txt:
                lines.append(f"- Management preferences (top dimensions): {top_txt}.")
        if "conflict_style" in results:
            cs = results["conflict_style"]
            lines.append(f"- Conflict style: {cs['primary']} (secondary: {cs['secondary']}).")

        # Quality flags summary
        q = results.get("quality", {})
        if q.get("warnings"):
            lines.append("- Response quality flags triggered: " + "; ".join(q["warnings"]))

        st.markdown("\n".join(lines) if lines else "—")

    st.divider()

    notes = st.text_area("Examiner notes (optional, included in PDF)", value="")

    export = {k: v for k, v in results.items() if not k.startswith("_")}
    st.download_button("Download results JSON", data=json.dumps(export, indent=2).encode("utf-8"), file_name="bea_results.json", mime="application/json")

    try:
        pdf_bytes = make_pdf_report(results, results.get('candidate_name',''), results.get('role',''), notes=notes)
        st.download_button("Download PDF summary report", data=pdf_bytes, file_name="bea_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF export failed: {e}")


def page_vervio_dashboard():
    st.header("Vervio Dashboard (Invites & Results)")
    st.caption("MVP flow: generate token link → send to candidate → results return here.")

    if not supabase_available():
        st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_SERVICE_KEY in Streamlit Secrets.")
        st.stop()

    require_admin()

    st.subheader("Create invite link")
    st.caption("Tip: Configure Org & Role first (Org & Role Setup). Then generate a token link here — the invite can store a snapshot of those settings.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        org = st.text_input("Organisation (label)", value=st.session_state.get("org_name", ""))
        role = st.text_input("Role", value=st.session_state.get("default_role_label", "Candidate Role"))
    with col2:
        email = st.text_input("Candidate email (optional)", value="")
    with col3:
        expires_days = st.number_input("Expires (days)", min_value=1, max_value=30, value=7)

    with st.expander("Invite settings (what the candidate will be scored against)", expanded=True):
        snapshot = st.checkbox("Snapshot current Org/Role parameters into this invite", value=True)
        if snapshot and ("org_values" not in st.session_state or "role_weights" not in st.session_state):
            st.warning("You haven\'t set Org & Role Setup in this session yet — the snapshot will use neutral defaults.")
        lock_modules = st.checkbox("Lock modules for the candidate (recommended)", value=True, disabled=not snapshot)

        # Module pack (stored in config_json so the candidate gets the right modules)
        st.write("Modules to include:")
        cA, cB, cC, cD = st.columns(4)
        with cA:
            m_beh = st.checkbox("Behavioural", value=True, disabled=not snapshot)
            m_values = st.checkbox("Values", value=True, disabled=not snapshot)
        with cB:
            m_ei = st.checkbox("EI SJT", value=True, disabled=not snapshot)
            m_big5 = st.checkbox("Big Five", value=True, disabled=not snapshot)
        with cC:
            m_lead = st.checkbox("Leadership", value=True, disabled=not snapshot)
            m_types = st.checkbox("Type lens", value=True, disabled=not snapshot)
        with cD:
            m_mgmt = st.checkbox("Mgmt prefs", value=True, disabled=not snapshot)
            m_conf = st.checkbox("Conflict style", value=True, disabled=not snapshot)

        modules = {
            "behaviour": bool(m_beh),
            "values": bool(m_values),
            "ei_sjt": bool(m_ei),
            "big5": bool(m_big5),
            "leadership": bool(m_lead),
            "type_lens": bool(m_types),
            "mgmt_prefs": bool(m_mgmt),
            "conflict": bool(m_conf),
        }

        if snapshot:
            st.success("This invite will store a snapshot of your Org/Role configuration (weights + values profile + modules).")
        else:
            st.warning("No snapshot will be stored. Candidate scoring will use whatever defaults are in the app at the time of completion (not recommended).")

    if st.button("Generate token link"):
        try:
            cfg = None
            org_clean = (org or "").strip()
            role_clean = (role or "").strip() or "Candidate Role"
            if snapshot:
                cfg = get_profile_snapshot(
                    role_label=role_clean,
                    org_name=org_clean,
                    modules=modules,
                    lock_modules=lock_modules,
                )
            out = sb_create_invite(
                role=role_clean,
                candidate_email=email.strip() or None,
                expires_days=int(expires_days),
                org=org_clean or None,
                config_json=cfg,
            )
            st.success("Invite created. Copy & email this link:")
            st.code(out["link"])
        except Exception as e:
            st.error(str(e))
    st.divider()

    st.subheader("Invites")
    filt = st.selectbox("Filter", ["All", "Active (invited/started)", "Completed", "Revoked/Expired"], index=0)

    inv = sb.table("invites").select("*").order("created_at", desc=True).limit(250).execute().data

    if not inv:
        st.info("No invites yet.")
        return

    for row in inv:
        status = (row.get("status") or "").lower()

        if filt == "Active (invited/started)" and status not in ("invited", "started"):
            continue
        if filt == "Completed" and status != "completed":
            continue
        if filt == "Revoked/Expired" and status not in ("revoked", "expired"):
            continue

        cols = st.columns([2.2, 1.6, 1.6, 1.0, 1.6, 1.6, 0.9])
        cols[0].write(row.get("candidate_email") or "—")
        cols[1].write(row.get("org") or "—")
        cols[2].write(row.get("role") or "—")
        cols[3].write(status or "—")
        cols[4].write(str(row.get("created_at") or "—"))
        cols[5].write(str(row.get("expires_at") or "—"))

        with cols[6]:
            if status in ("invited", "started"):
                if st.button("Revoke", key=f"revoke_{row.get('id')}"):
                    sb_revoke_invite(int(row.get("id")))
                    st.rerun()

        if status == "completed":
                    res = sb.table("results").select("*").eq("invite_id", row["id"]).order("created_at", desc=True).limit(1).execute().data
                    if res:
                        report = res[0].get("report_json") or {}
                        scores = res[0].get("scores_json") or {}

                        with st.expander(f"View result (Invite #{row['id']})"):
                            st.markdown("**Headline scores**")
                            st.json(scores)

                            st.markdown("**Report summary (stored)**")
                            st.json({k: report.get(k) for k in ["candidate_name","role","date","behavioral","values","ei","personality","leadership","type_lens","management_prefs","conflict_style","quality"] if k in report})

                            try:
                                pdf_bytes = make_pdf_report(report, report.get("candidate_name",""), report.get("role",""), notes="")
                                st.download_button("Download PDF summary report", data=pdf_bytes, file_name=f"bea_report_invite_{row['id']}.pdf", mime="application/pdf")
                                # Optional: upload PDF to Supabase Storage for retrieval from Supabase
                                if "SUPABASE_REPORTS_BUCKET" in st.secrets:
                                    if st.button("Upload PDF to Supabase Storage", key=f"uploadpdf_{row['id']}"):
                                        path = sb_upload_pdf_to_storage(int(row["id"]), pdf_bytes)
                                        if path:
                                            try:
                                                sb.table("results").update({"pdf_path": path}).eq("id", res[0]["id"]).execute()
                                            except Exception:
                                                pass
                                            st.success(f"Uploaded to storage: {path}")
                                        else:
                                            st.warning("Upload failed (check bucket name + Storage enabled).")

                            except Exception as e:
                                st.error(f"PDF export failed: {e}")



def page_psychometrics():
    st.title("Psychometrics & Validation Workspace")

    st.markdown(
        """This page helps you evaluate **reliability** (consistency) and build a **validity argument**
using established frameworks: Classical Test Theory (CTT), Item Response Theory (IRT), and
Generalizability Theory (G-Theory), as well as Messick’s unified validity evidence categories."""
    )

    # --- Data source
    st.subheader("1) Data source")
    rows = load_assessment_logs()
    st.write(f"Local log records found: **{len(rows)}**")
    st.caption("Logs are stored as JSONL at: data/assessments.jsonl (best-effort; may be disabled in hosted environments).")

    up = st.file_uploader("Optional: upload additional JSONL / JSON / CSV data", type=["jsonl", "json", "csv"])
    extra_rows: List[Dict[str, Any]] = []
    if up is not None:
        try:
            content = up.getvalue()
            if up.name.lower().endswith(".jsonl"):
                for ln in content.decode("utf-8").splitlines():
                    ln = ln.strip()
                    if ln:
                        extra_rows.append(json.loads(ln))
            elif up.name.lower().endswith(".json"):
                extra = json.loads(content.decode("utf-8"))
                if isinstance(extra, list):
                    extra_rows.extend(extra)
                elif isinstance(extra, dict):
                    extra_rows.append(extra)
            else:
                df = pd.read_csv(io.BytesIO(content))
                if "responses" in df.columns:
                    for _, r in df.iterrows():
                        rec = r.to_dict()
                        if isinstance(rec.get("responses"), str):
                            try:
                                rec["responses"] = json.loads(rec["responses"])
                            except Exception:
                                rec["responses"] = {}
                        extra_rows.append(rec)
        except Exception as e:
            st.error(f"Could not parse upload: {e}")

    all_rows = rows + extra_rows
    if len(all_rows) < 10:
        st.warning("You’ll get much stronger reliability/IRT estimates with more data (ideally 100+ respondents).")

    def build_matrix(item_ids: List[str], scorer=None) -> pd.DataFrame:
        data = []
        for rec in all_rows:
            resp = rec.get("responses", {}) or {}
            row = {}
            ok = True
            for iid in item_ids:
                if iid not in resp:
                    ok = False
                    break
                v = resp[iid]
                if scorer is not None:
                    try:
                        v = scorer(iid, v)
                    except Exception:
                        ok = False
                        break
                if v is None:
                    ok = False
                    break
                row[iid] = v
            if ok:
                data.append(row)
        return pd.DataFrame(data)

    st.subheader("2) Reliability & precision (CTT)")
    st.caption("CTT reminder: Observed score = True score + Error. Reliability helps estimate the Standard Error of Measurement (SEM).")

    big5_ids = [i for i, _, _, _ in BIG5_ITEMS]
    big5_rev = {i: rev for i, _, _, rev in BIG5_ITEMS}
    def big5_scorer(iid, raw):
        raw = int(raw)
        return 6 - raw if big5_rev.get(iid, False) else raw

    type_ids = [i for i, _, _, _ in TYPE_ITEMS]
    type_rev = {i: rev for i, _, _, rev in TYPE_ITEMS}
    def type_scorer(iid, raw):
        raw = int(raw)
        return 6 - raw if type_rev.get(iid, False) else raw

    mp_ids = [i for i, _, _, _ in MGMT_PREF_ITEMS]
    mp_rev = {i: rev for i, _, _, rev in MGMT_PREF_ITEMS}
    def mp_scorer(iid, raw):
        raw = int(raw)
        return 6 - raw if mp_rev.get(iid, False) else raw

    ei_ids = [it['id'] for it in EI_SJT]
    ei_map = {it['id']: it for it in EI_SJT}
    def ei_item_score(iid, choice):
        item = ei_map.get(iid)
        if not item:
            return 0.0
        return float(item['weights'].get(choice, 0.0))

    lead_ids = [i for i, _, _, _ in LEADERSHIP_ITEMS]
    lead_rev = {i: rev for i, _, _, rev in LEADERSHIP_ITEMS}
    def lead_scorer(iid, raw):
        raw = int(raw)
        return 6 - raw if lead_rev.get(iid, False) else raw

    selections = st.multiselect(
        "Choose scales to analyse",
        options=["Big Five (all items)", "Type lens (Type 1–9)", "Management preferences", "EI SJT (item credit)", "Leadership style"],
        default=["Big Five (all items)", "Type lens (Type 1–9)", "Management preferences"],
    )

    def show_ctt(name: str, df_items: pd.DataFrame):
        st.markdown(f"### {name}")
        if df_items.empty:
            st.info("Not enough complete records for this scale yet.")
            return
        a = cronbach_alpha(df_items)
        total = df_items.sum(axis=1)
        sd = float(total.std(ddof=1)) if len(total) >= 3 else float("nan")
        sem = ctt_sem(sd, a)
        st.write({"n_complete": int(df_items.shape[0]), "k_items": int(df_items.shape[1]), "alpha": a, "sd_total": sd, "SEM": sem})
        itc = item_total_correlations(df_items)
        if itc:
            st.caption("Item–total correlations (lower values may indicate weak items).")
            st.dataframe(pd.Series(itc).sort_values().to_frame("item_total_corr"), use_container_width=True)

    if "Big Five (all items)" in selections:
        df = build_matrix(big5_ids, scorer=big5_scorer)
        show_ctt("Big Five — total", df)

    if "Type lens (Type 1–9)" in selections:
        df = build_matrix(type_ids, scorer=type_scorer)
        show_ctt("Type lens — total", df)

    if "Management preferences" in selections:
        df = build_matrix(mp_ids, scorer=mp_scorer)
        show_ctt("Management preferences — total", df)

    if "EI SJT (item credit)" in selections:
        df = build_matrix(ei_ids, scorer=lambda iid, ch: ei_item_score(iid, ch))
        show_ctt("EI SJT — item credit", df)

    if "Leadership style" in selections:
        df = build_matrix(lead_ids, scorer=lead_scorer)
        show_ctt("Leadership — total", df)

    st.subheader("3) Item Response Theory (IRT) — exploratory")
    st.caption("This is a lightweight 2PL model for binary items. We binarise EI items as ‘max-credit’ vs not. Use for exploration only.")

    if st.button("Run exploratory IRT on EI items (max-credit binary)"):
        df = build_matrix(ei_ids, scorer=lambda iid, ch: 1 if ei_item_score(iid, ch) >= 1.0 else 0)
        if df.empty or df.shape[0] < 30:
            st.info("Need at least ~30 complete records to run this exploratory IRT.")
        else:
            X = df.to_numpy(dtype=float)
            out = irt_2pl_jml(X)
            if out.get("a") is None:
                st.info(out.get("note", "IRT not available."))
            else:
                item_stats = pd.DataFrame({"item": df.columns, "a_discrimination": out["a"], "b_difficulty": out["b"]}).set_index("item")
                st.dataframe(item_stats.sort_values("a_discrimination"), use_container_width=True)
                st.caption(out.get("note", ""))

    st.subheader("4) Generalizability Theory (G-Theory) — multi-rater / multi-occasion")
    st.caption("If you collect ratings from multiple raters or occasions, G-Theory can isolate where inconsistency comes from.")

    g_up = st.file_uploader("Upload long-format CSV with person, rater, occasion, score", type=["csv"], key="g_csv")
    if g_up is not None:
        try:
            g_df = pd.read_csv(g_up)
            st.dataframe(g_df.head(20), use_container_width=True)
            person_col = st.selectbox("Person column", options=list(g_df.columns), index=0)
            rater_col = st.selectbox("Rater column", options=list(g_df.columns), index=min(1, len(g_df.columns)-1))
            occasion_col = st.selectbox("Occasion column", options=list(g_df.columns), index=min(2, len(g_df.columns)-1))
            score_col = st.selectbox("Score column", options=list(g_df.columns), index=min(3, len(g_df.columns)-1))
            if st.button("Compute G-Theory components"):
                out = gtheory_prt(g_df, person_col, rater_col, occasion_col, score_col)
                st.json(out)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.subheader("5) Validity evidence builder (Messick)")
    st.caption("Build a defensible validity argument for your specific context (role, industry, country).")

    col1, col2 = st.columns(2)
    with col1:
        content_ev = st.text_area("Content evidence", height=120)
        resp_ev = st.text_area("Response process evidence", height=120)
        structure_ev = st.text_area("Internal structure evidence", height=120)
    with col2:
        relations_ev = st.text_area("Relations to other variables", height=120)
        cons_ev = st.text_area("Consequences (fairness / adverse impact / candidate experience)", height=120)

    if st.button("Export validity notes JSON"):
        blob = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "content": content_ev,
            "response_process": resp_ev,
            "internal_structure": structure_ev,
            "relations_other_variables": relations_ev,
            "consequences": cons_ev,
        }
        st.download_button(
            "Download validity_notes.json",
            data=json.dumps(blob, indent=2).encode("utf-8"),
            file_name="validity_notes.json",
            mime="application/json",
        )

def page_research_notes():
    st.header("Research notes embedded in this prototype")
    st.write(
        "These notes summarise the research principles this prototype follows. "
        "They are intentionally short; consult your source documents and (for hiring use) professional guidance."
    )

    st.subheader("Behavioural alignment: culture in practice")
    st.write(
        "- Culture shows up as repeated, observable behaviours—what gets rewarded/tolerated under pressure.\n"
        "- The 20 behaviours assessed here are written as observable constructs (not personality adjectives), split into:\n"
        "  **10 core ‘load-bearing’ behaviours** + **10 role/strategy-dependent differentiators**.\n"
        "- Behaviour-level measurement is useful because behaviours can be coached and reinforced over time."
    )

    st.subheader("Culture-fit scoring in hiring: maturity & cautions")
    st.write(
        "- Predicting job performance is most defensible when anchored in structured, job-relevant methods and a validity framework.\n"
        "- Person–organisation fit is real but context-dependent; measurement varies; local validation is essential.\n"
        "- ‘Fit’ can drift into similarity bias; selecting only for similarity can reduce adaptability and embed bias.\n"
        "- Use profile comparison for value congruence, and treat ‘culture add’ concepts as higher-risk unless validated."
    )

    st.subheader("EQ/EI scoring & interpretation: what robust reading requires")
    st.write(
        "- ‘EQ’ tools differ (ability tests vs self-report vs 360). Scores are not interchangeable.\n"
        "- Robust interpretation starts with data quality/validity checks, then score scale/norm context, then patterns.\n"
        "- Avoid over-reading small differences; measurement error matters; interpret as hypotheses about behaviour.\n"
        "- For screening, scenario-based (SJT) approaches can be closer to job behaviour than pure self-report."
    )

    st.subheader("Practical defensibility in South Africa")
    st.write(
        "- If used for employment decisions, assessments should be valid, reliable, fair, and not biased.\n"
        "- Consider professional control/classification expectations for psychological tests.\n"
        "- Handle assessment data as sensitive personal information: purpose limitation, retention, access rights, security."
    )

    st.subheader("What this app is and is not")
    st.write(
        "**Is:** a configurable, transparent prototype you can pilot for development, onboarding, coaching, and structured discussions.\n\n"
        "**Is not:** a validated, normed, legally audited psychometric instrument. For hiring gatekeeping, build a full validation program "
        "and consult qualified professionals."
    )


def page_candidate_token(token: str):
    st.title(APP_TITLE)

    if not supabase_available():
        st.error("This deployment is not configured to accept token-based assessments.")
        st.stop()

    invite = sb_get_invite_by_token(token)
    if not invite:
        st.error("Invalid link (token not found).")
        st.stop()

    # Hard blocks
    status = (invite.get("status") or "").lower()
    if status in ("revoked", "expired"):
        st.error("This link is no longer active.")
        st.stop()

    # Expiry check (server-side time)
    exp_raw = invite.get("expires_at")
    if exp_raw:
        exp = parse_dt(str(exp_raw))
        if datetime.now(timezone.utc) > exp:
            sb_expire_invite(int(invite.get("id")))
            st.error("This link has expired.")
            st.stop()

    if status == "completed":
        st.info("This assessment link has already been completed.")
        st.stop()

    # Mark started only once (best-effort)
    if status == "invited":
        sb_mark_invite_started(int(invite.get("id")))

    # Apply org/role snapshot (so scoring uses the correct parameters)
    apply_profile_snapshot(invite.get("config_json") or {})

    # Run the normal assessment page in token mode (no sidebar navigation)
    page_assessment(token_mode=True, invite=invite)



def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

    token = get_query_param("token")
    if token:
        page_candidate_token(str(token))
        return

    # Optional: lock the entire app behind the Vervio password (recommended for pilots)
    if secret_truthy("LOCK_PUBLIC_APP", False):
        require_admin()
        with st.sidebar:
            st.success("Vervio access unlocked")
            if st.button("Lock / sign out"):
                st.session_state.admin_ok = False
                st.rerun()

    st.sidebar.title("Navigation")
    pages = {
        "Home": page_home,
        "Org & Role Setup": page_org_setup,
        "Candidate Assessment": page_assessment,
        "Results & Report": page_results,
        "Psychometrics & Validation": page_psychometrics,
        "Vervio Dashboard": page_vervio_dashboard,
        "Research notes": page_research_notes,
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()), index=0)
    pages[choice]()


if __name__ == "__main__":
    main()
