"""
Loan Origination — Credit Scoring Tool
Loan officer decision support. Enter applicant details, get score + decision.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Origination", page_icon="📋", layout="wide")

st.title("Loan Origination — Credit Scoring")
st.caption("Loan officer decision support | OSFI E-23 | PIPEDA compliant")


# ── Scoring logic ──────────────────────────────────────────────────────────
def score_application(inputs):
    """
    Simple WoE-inspired scorecard. Maps applicant inputs to a 300–850 score.
    Based on logistic regression with PDO=20, base score=600, base odds=50:1.
    """
    score = 600

    # DEBTINC contribution
    dti = inputs["dti"]
    if   dti < 20:  score += 45
    elif dti < 30:  score += 25
    elif dti < 40:  score +=  0
    elif dti < 50:  score -= 35
    else:           score -= 70

    # Derogatory reports
    d = inputs["derog"]
    if   d == 0:  score += 30
    elif d == 1:  score -= 45
    elif d == 2:  score -= 85
    else:         score -= 120

    # Delinquent lines
    dl = inputs["delinq"]
    if   dl == 0:  score += 25
    elif dl == 1:  score -= 38
    elif dl == 2:  score -= 60
    else:          score -= 90

    # Years on job
    yoj = inputs["yoj"]
    if   yoj > 10:  score += 20
    elif yoj >  5:  score += 10
    elif yoj >  2:  score +=  0
    elif yoj >  1:  score -= 12
    else:           score -= 28

    # Credit line age
    cl = inputs["clage"]
    if   cl > 240:  score += 18
    elif cl > 120:  score += 10
    elif cl >  60:  score +=  0
    elif cl >  24:  score -=  8
    else:           score -= 22

    # LTV
    if inputs["pval"] > 0:
        ltv = inputs["loan"] / inputs["pval"]
        if   ltv < 0.60:  score += 15
        elif ltv < 0.75:  score +=  5
        elif ltv < 0.85:  score -= 12
        else:             score -= 28

    # Inquiries
    ni = inputs["ninq"]
    if   ni == 0:  score += 10
    elif ni <= 2:  score +=  0
    elif ni <= 4:  score -= 15
    else:          score -= 32

    score = int(np.clip(score, 300, 850))

    # Convert score → PD via inverse log-odds scaling
    log_odds = (score - 600) / 28.85
    pd12     = float(1 / (1 + np.exp(log_odds)))
    pdlt     = float(min(pd12 * 2.0, 0.99))

    # IFRS 9 staging
    dpd   = dl * 30
    stage = 3 if (dpd >= 90 or pdlt > 0.40) else (2 if (dpd >= 30 or pd12 > 0.10) else 1)

    # Risk grade
    if   score >= 780:  grade = "A+"
    elif score >= 740:  grade = "A"
    elif score >= 700:  grade = "B+"
    elif score >= 660:  grade = "B"
    elif score >= 620:  grade = "C+"
    elif score >= 580:  grade = "C"
    else:               grade = "D"

    # Decision
    if   score >= 660:  decision = "APPROVE"
    elif score >= 620:  decision = "CONDITIONAL APPROVE"
    elif score >= 580:  decision = "REFER TO CREDIT OFFICER"
    else:               decision = "DECLINE"

    # LGD (collateral-based)
    if inputs["pval"] > 0:
        net_coll = inputs["pval"] * 0.75   # 25% OSFI haircut
        lgd = float(max(0.0, 1 - min(net_coll, inputs["loan"]) * 0.92 / inputs["loan"]))
    else:
        lgd = 0.65

    ead  = float(inputs["loan"])
    ecl  = float(pd12 * lgd * ead)
    ltv  = float(inputs["loan"] / max(inputs["pval"], 1))
    cltv = float((inputs["loan"] + inputs["mortdue"]) / max(inputs["pval"], 1))

    # Top adverse action reason codes
    reasons = []
    if d > 0:
        reasons.append(f"Derogatory reports ({d} on file) — major negative signal")
    if dl > 0:
        reasons.append(f"Delinquent credit lines ({dl}) — elevated behavioural risk")
    if dti > 43:
        reasons.append(f"Debt-to-income {dti:.1f}% exceeds OSFI B-20 TDS cap (44%)")
    if yoj < 2:
        reasons.append(f"Employment tenure {yoj:.1f} yr — short-term instability risk")
    if ni >= 4:
        reasons.append(f"Recent inquiries ({ni}) — credit-seeking behaviour")
    if ltv > 0.80:
        reasons.append(f"LTV {ltv*100:.1f}% — high-ratio, limited equity cushion")

    return dict(
        score=score, grade=grade, decision=decision,
        pd12=pd12, pdlt=pdlt, lgd=lgd, ead=ead, ecl=ecl,
        stage=stage, ltv=ltv, cltv=cltv, reasons=reasons[:4]
    )


# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Applicant Details")

st.sidebar.subheader("Loan")
loan    = st.sidebar.number_input("Loan Amount (CAD $)", 5_000, 500_000, 35_000, 5_000)
purpose = st.sidebar.selectbox("Purpose", ["DebtCon — Debt Consolidation", "HomeImp — Home Improvement"])

st.sidebar.subheader("Property")
pval    = st.sidebar.number_input("Property Value (CAD $)", 0, 3_000_000, 280_000, 10_000)
mortdue = st.sidebar.number_input("Existing Mortgage (CAD $)", 0, 2_000_000, 145_000, 10_000)

st.sidebar.subheader("Employment")
job  = st.sidebar.selectbox("Occupation", ["Mgr", "ProfExe", "Office", "Sales", "Self", "Other"])
yoj  = st.sidebar.slider("Years at Current Employer", 0.0, 30.0, 4.5, 0.5)

st.sidebar.subheader("Financial Profile")
dti   = st.sidebar.slider("Debt-to-Income Ratio (%)", 5.0, 80.0, 32.0, 0.5)
clage = st.sidebar.slider("Age of Oldest Credit Line (months)", 0, 600, 180, 6)
ninq  = st.sidebar.slider("Recent Credit Inquiries", 0, 20, 1, 1)

st.sidebar.subheader("Adverse History")
derog  = st.sidebar.selectbox("Major Derogatory Reports", [0, 1, 2, 3, 4, 5])
delinq = st.sidebar.selectbox("Delinquent Credit Lines",  [0, 1, 2, 3, 4, 5])

# ── Score ──────────────────────────────────────────────────────────────────
r = score_application(dict(
    loan=loan, pval=pval, mortdue=mortdue,
    dti=dti, derog=derog, delinq=delinq,
    yoj=yoj, clage=clage, ninq=ninq
))

# ── Output ─────────────────────────────────────────────────────────────────
col_score, col_decision = st.columns([1, 2])

with col_score:
    st.metric("Credit Score", r["score"], help="300–850 scale. PDO=20, Base=600, Odds=50:1")
    st.metric("Risk Grade", r["grade"])
    st.metric("IFRS 9 Stage", f"Stage {r['stage']}")

with col_decision:
    decision_colour = {
        "APPROVE":                  "✅",
        "CONDITIONAL APPROVE":      "🟡",
        "REFER TO CREDIT OFFICER":  "🔄",
        "DECLINE":                  "❌",
    }
    icon = decision_colour.get(r["decision"], "⚠️")
    st.subheader(f"{icon}  {r['decision']}")

    stage_labels = {
        1: "Stage 1 — Performing (12-month ECL)",
        2: "Stage 2 — Underperforming (Lifetime ECL)",
        3: "Stage 3 — Non-Performing (Lifetime ECL)",
    }
    if r["stage"] == 1:
        st.success(stage_labels[1])
    elif r["stage"] == 2:
        st.warning(stage_labels[2])
    else:
        st.error(stage_labels[3])

st.divider()

# ── Risk metrics ───────────────────────────────────────────────────────────
st.subheader("Risk Parameters")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("PD (12-Month)",   f"{r['pd12']*100:.2f}%")
c2.metric("PD (Lifetime)",   f"{r['pdlt']*100:.2f}%")
c3.metric("LGD",             f"{r['lgd']*100:.1f}%")
c4.metric("EAD",             f"${r['ead']:,.0f}")
c5.metric("ECL (12M Est.)",  f"${r['ecl']:,.0f}")
c6.metric("LTV",             f"{r['ltv']*100:.1f}%",
          delta="⚠️ High LTV" if r["ltv"] > 0.80 else "OK")

st.divider()

# ── Adverse action reasons + summary table ─────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Adverse Action Reasons")
    st.caption("Top risk factors driving score — required under PIPEDA for declined applicants")

    if r["reasons"]:
        for i, reason in enumerate(r["reasons"], 1):
            st.warning(f"**{i}.** {reason}")
    else:
        st.success("No major adverse factors identified.")

with col_right:
    st.subheader("Application Summary")

    summary = pd.DataFrame({
        "Field":  ["Loan Amount", "Property Value", "Existing Mortgage",
                   "LTV", "CLTV", "Debt-to-Income",
                   "Employment", "Derogatory Reports", "OSFI B-20 Flag"],
        "Value":  [f"${loan:,.0f}", f"${pval:,.0f}", f"${mortdue:,.0f}",
                   f"{r['ltv']*100:.1f}%", f"{r['cltv']*100:.1f}%",
                   f"{dti:.1f}%", f"{yoj:.1f} years",
                   str(derog),
                   "⚠️ BREACH" if dti > 44 else "✓ OK"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Score distribution chart ───────────────────────────────────────────────
st.divider()
st.subheader("Score Bands — Decision Thresholds")

fig, ax = plt.subplots(figsize=(10, 2.5))
bands = [
    (300, 580, "#ef4444", "Decline"),
    (580, 620, "#f97316", "Refer"),
    (620, 660, "#eab308", "Conditional"),
    (660, 850, "#22c55e", "Approve"),
]
for lo, hi, colour, label in bands:
    ax.barh(0, hi - lo, left=lo, color=colour, height=0.5, alpha=0.75)
    ax.text((lo + hi) / 2, 0, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white")

ax.axvline(r["score"], color="navy", lw=2.5, label=f"Score: {r['score']}")
ax.set_xlim(300, 850)
ax.set_yticks([])
ax.set_xlabel("Credit Score")
ax.set_title(f"Applicant Score: {r['score']} ({r['grade']})")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="x", alpha=0.3)
st.pyplot(fig, use_container_width=True)
plt.close()

st.caption(
    "⚖️ Model governed by OSFI E-23. Credit decisions must comply with the "
    "Canadian Human Rights Act, PIPEDA, and OSFI B-20. "
    "This score is not the sole basis for any credit decision."
)
