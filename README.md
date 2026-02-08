
# BEA • Behavioural & Emotional Alignment Assessment (Prototype)

A simple **web-based** assessment app built with **Python + Streamlit**.

It provides:
- Behavioural alignment scoring across 20 research-aligned behaviours (core + differentiators)
- Values congruence (P–O fit) via profile comparison (correlation)
- EI scenario module (SJT-style) with transparent scoring keys (prototype)
- Big Five personality profile (original items; descriptive archetype)
- Leadership style profile (original items; descriptive type)
- Response-quality flags (time, straight-lining, high endorsement, inconsistency pairs)
- JSON export + 1-page PDF summary report

> ⚠️ Important: This is a **prototype** for structured decision support and development.  
> Do not use as a sole hiring gatekeeper without local validation, fairness/bias monitoring, and (where required) appropriately registered professionals.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Configure
Use **Org & Role Setup** to:
- set your organisation values profile
- weight behaviours per role
Download the profile JSON and keep it with your hiring/assessment documentation.

## Next steps for real-world use (recommended)
- Do a role-based job analysis and specify intended score interpretations.
- Pilot with a sample, compute reliability/precision, and validate against criteria (performance, retention, etc.).
- Run subgroup/fairness checks and document governance (privacy, retention, access rights).
