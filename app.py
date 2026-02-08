
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

import numpy as np
import pandas as pd
import streamlit as st

# Optional PDF export (reportlab is installed in this environment)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm


APP_TITLE = "BEA • Behavioural & Emotional Alignment Assessment (Prototype)"
VERSION = "0.1.0"

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
    min_duration_sec: float = 180.0,
) -> Dict[str, Any]:
    """Simple response-quality checks inspired by validity-index practices."""
    vals = [v for v in responses.values() if isinstance(v, (int, float))]
    flags = {"duration_sec": duration_sec, "warnings": []}

    if duration_sec < min_duration_sec:
        flags["warnings"].append("Very fast completion time (possible rushing).")

    if len(vals) >= 10:
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        if mean >= 4.7:
            flags["warnings"].append("Very high average endorsement (possible 'fake good' / impression management).")
        if sd <= 0.35:
            flags["warnings"].append("Very low response variability (possible straight-lining).")

    if inconsistency_pairs:
        inconsistent = 0
        checked = 0
        for a, b in inconsistency_pairs:
            if a in responses and b in responses and isinstance(responses[a], (int, float)) and isinstance(responses[b], (int, float)):
                checked += 1
                if abs(responses[a] - responses[b]) >= 3:
                    inconsistent += 1
        if checked >= 3 and inconsistent >= 2:
            flags["warnings"].append("Inconsistent responding on paired items (interpret with caution).")
        flags["inconsistency_checked_pairs"] = checked
        flags["inconsistency_hits"] = inconsistent

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

# Inconsistency pairs (similar items phrased differently)
INCONSISTENCY_PAIRS = [
    ("P01", "P02"), ("P09", "P10"), ("P13", "P14"), ("P17", "P18"),
    ("B03_1", "B03_3"), ("B05_1", "B05_3"), ("B06_1", "B06_3"),
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

def make_pdf_report(payload: Dict[str, Any]) -> bytes:
    import io
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x = 18 * mm
    y = height - 18 * mm

    def line(txt: str, dy: float = 6.0):
        nonlocal y
        c.drawString(x, y, txt)
        y -= dy * mm

    c.setFont("Helvetica-Bold", 14)
    line("BEA Assessment Report (Prototype)", dy=8)
    c.setFont("Helvetica", 10)
    line(f"Candidate: {payload.get('candidate_name','-')} | Date: {payload.get('date','-')} | Version: {VERSION}", dy=7)

    c.setFont("Helvetica-Bold", 11)
    line("Overall summary", dy=7)
    c.setFont("Helvetica", 10)
    if "behavioral" in payload:
        line(f"Behavioural alignment (0–100): {payload['behavioral']['overall']:.1f} ({payload['behavioral']['band']})")
    if "values" in payload:
        line(f"Values congruence (0–100): {payload['values']['fit_0_100']:.1f} ({payload['values']['band']})")
    if "ei" in payload:
        line(f"EI SJT (0–100): {payload['ei']['overall']:.1f} ({payload['ei']['band']})")
    if "personality" in payload:
        line(f"Personality archetype: {payload['personality']['archetype']}")
    if "leadership" in payload:
        line(f"Leadership style: {payload['leadership']['style']}")

    y -= 3 * mm
    if "behavioral" in payload:
        c.setFont("Helvetica-Bold", 11)
        line("Key strengths (top 3 behaviours)", dy=7)
        c.setFont("Helvetica", 10)
        for s in payload["behavioral"]["top_strengths"][:3]:
            line(f"- {s['behavior']}: {s['score_0_100']:.1f}")

        y -= 2 * mm
        c.setFont("Helvetica-Bold", 11)
        line("Watch-outs (bottom 3 behaviours)", dy=7)
        c.setFont("Helvetica", 10)
        for w in payload["behavioral"]["watchouts"][:3]:
            line(f"- {w['behavior']}: {w['score_0_100']:.1f}")

    y -= 2 * mm
    c.setFont("Helvetica-Bold", 11)
    line("Response quality flags", dy=7)
    c.setFont("Helvetica", 10)
    warnings = payload.get("quality", {}).get("warnings", [])
    if warnings:
        for w in warnings[:4]:
            line(f"- {w}")
    else:
        line("- None flagged.")

    y -= 4 * mm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(x, y, "Note: Prototype tool. Do not use as a sole hiring gatekeeper without local validation and fairness checks.")
    c.showPage()
    c.save()
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
    st.write("Define an organisation values profile and role-specific weighting for the 20 behaviours.")

    with st.expander("Organisation values profile (used for values-fit profile comparison)", expanded=True):
        org = st.session_state.get("org_values", {v: 4 for v in VALUES_DIMENSIONS})
        cols = st.columns(2)
        new_org = {}
        for i, v in enumerate(VALUES_DIMENSIONS):
            with cols[i % 2]:
                new_org[v] = st.select_slider(v, options=[1,2,3,4,5], value=int(org.get(v, 4)), format_func=lambda x: VALUES_SCALE_LABELS[x-1])
        st.session_state["org_values"] = new_org

    with st.expander("Role weights for behaviour alignment (optional)", expanded=True):
        st.write("Weight behaviours that are more critical for this role. Default is 1.0.")
        role_weights = st.session_state.get("role_weights", {b["id"]: 1.0 for b in BEHAVIORS})
        new_weights = {}
        for b in BEHAVIORS:
            new_weights[b["id"]] = st.slider(
                f"{b['id']} • {b['name']}", min_value=0.0, max_value=3.0, value=float(role_weights.get(b["id"], 1.0)), step=0.25
            )
        st.session_state["role_weights"] = new_weights

    profile = {
        "org_values": st.session_state["org_values"],
        "role_weights": st.session_state["role_weights"],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
    }
    st.download_button("Download profile JSON", data=json.dumps(profile, indent=2).encode("utf-8"), file_name="bea_profile.json", mime="application/json")

def page_assessment():
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

    candidate_name = st.text_input("Candidate / staff member name (for report)", value="")
    role_name = st.text_input("Role / position being assessed", value="")

    start = st.session_state.get("start_time", None)
    if start is None:
        st.session_state["start_time"] = time.time()

    st.subheader("Modules")
    colA, colB, colC = st.columns(3)
    with colA:
        use_beh = st.checkbox("Behavioural alignment (20 behaviours)", value=True)
        use_values = st.checkbox("Values congruence (P–O fit)", value=True)
    with colB:
        use_ei = st.checkbox("EI scenario test (SJT)", value=True)
        use_big5 = st.checkbox("Personality profile (Big Five)", value=True)
    with colC:
        use_lead = st.checkbox("Leadership style (for supervisors/managers)", value=True)

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

    if use_lead:
        st.subheader("5) Leadership / management style")
        st.caption("Answer based on how you typically lead or prefer to lead (if applicable).")
        for item_id, text, dim, reverse in LEADERSHIP_ITEMS:
            key = item_id
            val = st.radio(text, options=LIKERT_VALUES, format_func=lambda x: LIKERT_LABELS[x-1], horizontal=True, key=key)
            responses[key] = int(val)

    st.divider()
    if st.button("Generate results"):
        duration = time.time() - st.session_state.get("start_time", time.time())

        beh_res = {k: v for k, v in responses.items() if k.startswith("B")}
        val_res = {k[2:]: v for k, v in responses.items() if k.startswith("V_")}
        ei_res = {k: v for k, v in responses.items() if k.startswith("E")}
        big5_res = {k: v for k, v in responses.items() if k.startswith("P")}
        lead_res = {k: v for k, v in responses.items() if k.startswith("L")}

        role_weights = st.session_state.get("role_weights", {b["id"]: 1.0 for b in BEHAVIORS})
        org_values = st.session_state.get("org_values", {v: 4 for v in VALUES_DIMENSIONS})

        results: Dict[str, Any] = {"candidate_name": candidate_name, "role": role_name, "date": time.strftime("%Y-%m-%d")}

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

        numeric_responses = {k: v for k, v in responses.items() if isinstance(v, int)}
        q = response_flags(numeric_responses, duration, inconsistency_pairs=INCONSISTENCY_PAIRS)
        results["quality"] = q

        st.session_state["latest_results"] = results
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

    st.divider()

    export = {k: v for k, v in results.items() if not k.startswith("_")}
    st.download_button("Download results JSON", data=json.dumps(export, indent=2).encode("utf-8"), file_name="bea_results.json", mime="application/json")

    try:
        pdf_bytes = make_pdf_report(results)
        st.download_button("Download PDF summary report", data=pdf_bytes, file_name="bea_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"PDF export failed: {e}")

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

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    st.sidebar.title("Navigation")
    pages = {
        "Home": page_home,
        "Org & Role Setup": page_org_setup,
        "Candidate Assessment": page_assessment,
        "Results & Report": page_results,
        "Research notes": page_research_notes,
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.caption("Prototype • not a sole hiring gatekeeper")
    pages[choice]()

if __name__ == "__main__":
    main()
