"""
01_Loan_Origination.py
──────────────────────
Loan Officer Simulation Tool — Real-time credit scoring.
Mirrors origination decisioning tools at RBC, TD, BMO.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Loan Origination | CreditIQ", page_icon="📋", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#003366 0%,#1a3a5c 100%); }
[data-testid="stSidebar"] * { color: #fff !important; }
h1,h2 { color: #003366 !important; }
.stButton > button { background:#003366 !important; color:white !important;
    border:none !important; border-radius:6px !important; font-weight:600 !important; }
[data-testid="metric-container"] { background:white; border:1px solid #e0e7ff;
    border-radius:10px; padding:12px; box-shadow:0 2px 8px rgba(0,51,102,0.08); }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size:1.5rem !important; font-weight:700 !important; color:#003366 !important; }
</style>
""", unsafe_allow_html=True)

# ── Scoring Engine ────────────────────────────────────────────
def compute_score(inp):
    score = 600
    dti = inp["dti"]
    score += 45 if dti < 20 else 25 if dti < 30 else 0 if dti < 40 else -35 if dti < 50 else -70
    d = inp["derog"]
    score += 30 if d == 0 else -45 if d == 1 else -85 if d == 2 else -120
    dl = inp["delinq"]
    score += 25 if dl == 0 else -38 if dl == 1 else -60 if dl == 2 else -90
    yoj = inp["yoj"]
    score += 20 if yoj > 10 else 10 if yoj > 5 else 0 if yoj > 2 else -12 if yoj > 1 else -28
    cl = inp["clage"]
    score += 18 if cl > 240 else 10 if cl > 120 else 0 if cl > 60 else -8 if cl > 24 else -22
    pv = inp["pval"]
    if pv > 0:
        ltv = inp["loan"] / pv
        score += 15 if ltv < 0.60 else 5 if ltv < 0.75 else -12 if ltv < 0.85 else -28
    ni = inp["ninq"]
    score += 10 if ni == 0 else 0 if ni <= 2 else -15 if ni <= 4 else -32
    score = int(np.clip(score, 300, 850))

    lo = (score - 600) / 28.85
    pd12 = 1 / (1 + np.exp(lo))
    pdlt = min(pd12 * 3.5, 0.99)
    dpd  = dl * 30
    stage = 3 if dpd >= 90 or pdlt > 0.20 else 2 if dpd >= 30 or pd12 > 0.03 else 1

    grade = ("A+" if score >= 780 else "A" if score >= 740 else "B+" if score >= 700
             else "B" if score >= 660 else "C+" if score >= 620 else "C" if score >= 580 else "D")
    decision = ("APPROVE" if score >= 660 else "CONDITIONAL APPROVE" if score >= 620
                else "REFER TO CREDIT OFFICER" if score >= 580 else "DECLINE")

    lgd = max(0.0, 1 - min(pv * 0.75, inp["loan"]) * 0.92 / max(inp["loan"], 1)) if pv > 0 else 0.65
    ecl12 = pd12 * lgd * inp["loan"]
    eclt  = pdlt * lgd * inp["loan"]

    reasons = []
    if d > 0:  reasons.append(("Derogatory Reports",       f"{d} report(s) on bureau file",               "neg", d * 40))
    if dl > 0: reasons.append(("Delinquent Credit Lines",  f"{dl} delinquent line(s)",                    "neg", dl * 35))
    if dti > 43: reasons.append(("Debt-to-Income Ratio",   f"{dti:.1f}% — exceeds OSFI B-20 44% cap",    "neg", (dti - 43) * 2))
    if yoj < 2:  reasons.append(("Employment Stability",   f"Only {yoj:.1f} yr at current employer",      "neg", 25))
    if ni >= 4:  reasons.append(("Recent Credit Inquiries",f"{ni} inquiries in past 12 months",           "neg", ni * 8))
    if d == 0 and dl == 0:
        reasons.append(("Clean Credit History",   "No derogatory or delinquent records",              "pos", 30))
    if cl > 120:
        reasons.append(("Credit Seniority",       f"{cl} months of established credit history",       "pos", 18))
    if yoj > 5:
        reasons.append(("Stable Employment",      f"{yoj:.1f} years at current employer",             "pos", 15))
    reasons.sort(key=lambda x: x[3], reverse=True)

    return dict(score=score, pd12=pd12, pdlt=pdlt, lgd=lgd, ead=inp["loan"],
                ecl12=ecl12, eclt=eclt, stage=stage, grade=grade, decision=decision,
                reasons=reasons[:4], ltv=(inp["loan"] / max(pv, 1)))


def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(5, 3.2), subplot_kw=dict(polar=True))
    bands = [(300,580,"#dc2626"),(580,620,"#f97316"),(620,660,"#eab308"),
             (660,720,"#84cc16"),(720,850,"#16a34a")]
    for lo, hi, col in bands:
        t1 = np.pi*(1-(lo-300)/550); t2 = np.pi*(1-(hi-300)/550)
        ax.fill_between(np.linspace(t2, t1, 50), 0.55, 1.0, color=col, alpha=0.88)
    tn = np.pi*(1-(score-300)/550)
    ax.plot([tn, tn], [0, 0.88], color="#0f172a", lw=3, zorder=5)
    ax.plot(tn, 0, "o", color="#0f172a", markersize=7, zorder=6)
    ax.set_ylim(0, 1.15); ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1); ax.set_xlim(0, np.pi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    for sv, lbl in [(300,"300"),(580,"580"),(660,"660"),(720,"720"),(850,"850")]:
        ax.text(np.pi*(1-(sv-300)/550), 1.2, lbl, ha="center", va="center",
                fontsize=7.5, color="#374151", fontweight="500")
    ax.text(np.pi/2, -0.32, str(score), ha="center", va="center",
            fontsize=34, fontweight="bold", color="#003366", transform=ax.transData)
    ax.text(np.pi/2, -0.52, "Credit Score", ha="center", fontsize=10,
            color="#6b7280", transform=ax.transData)
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    plt.tight_layout(pad=0)
    return fig


# ── Sidebar Inputs ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 Application Form")
    st.divider()
    st.markdown("**💰 Loan**")
    loan    = st.number_input("Loan Amount (CAD)", 5_000, 500_000, 35_000, 5_000)
    purpose = st.selectbox("Purpose", ["DebtCon — Debt Consolidation", "HomeImp — Home Improvement"])
    st.markdown("**🏠 Property**")
    pval    = st.number_input("Property Value (CAD)", 0, 3_000_000, 280_000, 10_000)
    mortdue = st.number_input("Existing Mortgage (CAD)", 0, 2_000_000, 145_000, 10_000)
    st.markdown("**💼 Employment**")
    job     = st.selectbox("Occupation", ["Mgr","ProfExe","Office","Sales","Self","Other"])
    yoj     = st.slider("Years at Current Employer", 0.0, 30.0, 4.5, 0.5)
    st.markdown("**📊 Financials**")
    dti     = st.slider("Debt-to-Income Ratio (%)", 5.0, 80.0, 32.0, 0.5)
    clage   = st.slider("Oldest Credit Line (months)", 0, 600, 180, 6)
    clno    = st.slider("Number of Credit Lines", 0, 30, 6, 1)
    ninq    = st.slider("Recent Inquiries", 0, 20, 1, 1)
    st.markdown("**⚠️ Adverse History**")
    derog   = st.selectbox("Derogatory Reports", [0,1,2,3,4,5])
    delinq  = st.selectbox("Delinquent Lines",   [0,1,2,3,4,5])
    st.divider()
    st.button("🔍  Score Application", use_container_width=True)

inp = dict(loan=loan, pval=pval, dti=dti, derog=derog, delinq=delinq,
           yoj=yoj, clage=clage, ninq=ninq, mortdue=mortdue)
r   = compute_score(inp)

# ── Header ────────────────────────────────────────────────────
st.markdown("# 📋 Loan Origination — Real-Time Scoring")
st.caption("Loan Officer Decision Support | OSFI E-23 | Adverse Action Reason Codes | PIPEDA Compliant")
st.divider()

# ── Score + Decision ──────────────────────────────────────────
col_g, col_d, col_m = st.columns([2, 2, 3])

with col_g:
    st.pyplot(draw_gauge(r["score"]), use_container_width=True)
    plt.close("all")

with col_d:
    st.markdown("<br>", unsafe_allow_html=True)
    badge = {"APPROVE":("#d1fae5","#065f46","#34d399","✅"),
             "CONDITIONAL APPROVE":("#fef3c7","#92400e","#fbbf24","🔶"),
             "REFER TO CREDIT OFFICER":("#fffbeb","#92400e","#f59e0b","🔄"),
             "DECLINE":("#fee2e2","#991b1b","#f87171","❌")}
    bg,tc,bc,ic = badge.get(r["decision"], badge["REFER TO CREDIT OFFICER"])
    st.markdown(f"""
    <div style="text-align:center; padding:18px 10px;">
      <div style="background:{bg};color:{tc};border:2px solid {bc};border-radius:20px;
                  padding:8px 20px;font-weight:700;font-size:1rem;display:inline-block;">
        {ic} {r["decision"]}
      </div>
      <div style="font-size:2.4rem;font-weight:800;color:#003366;margin-top:12px;">
        Grade: {r["grade"]}
      </div>
      <div style="color:#6b7280;font-size:0.75rem;">OSFI Internal Risk Classification</div>
    </div>""", unsafe_allow_html=True)
    sc = {1:"#16a34a",2:"#d97706",3:"#dc2626"}
    sl = {1:"Stage 1 — Performing (12M ECL)",2:"Stage 2 — Watch (Lifetime ECL)",
          3:"Stage 3 — Non-Performing (Lifetime ECL)"}
    st.markdown(f"""
    <div style="background:{sc[r['stage']]}18;border:1.5px solid {sc[r['stage']]};
                border-radius:8px;padding:10px;text-align:center;margin-top:8px;">
      <span style="color:{sc[r['stage']]};font-weight:700;font-size:0.88rem;">
        🏛️ IFRS 9: {sl[r['stage']]}
      </span>
    </div>""", unsafe_allow_html=True)

with col_m:
    a,b,c = st.columns(3)
    a.metric("PD (12-Month)", f"{r['pd12']*100:.2f}%")
    b.metric("LGD",           f"{r['lgd']*100:.1f}%")
    c.metric("EAD",           f"${r['ead']:,.0f}")
    d,e,f = st.columns(3)
    d.metric("ECL (12M)",     f"${r['ecl12']:,.0f}")
    e.metric("ECL (Lifetime)",f"${r['eclt']:,.0f}")
    ltv_pct = r["ltv"]*100
    f.metric("LTV",           f"{ltv_pct:.1f}%",
             delta="⚠️ High" if ltv_pct > 80 else "✓ OK")

st.divider()

# ── Risk Factors + Detail ─────────────────────────────────────
col_r, col_t = st.columns(2)

with col_r:
    st.markdown("### 🔍 Risk Factor Analysis")
    st.caption("Top adverse action reason codes — PIPEDA §11 compliant")
    for feat, desc, direction, _ in r["reasons"]:
        bg2 = "#fee2e2" if direction=="neg" else "#d1fae5"
        bc2 = "#f87171" if direction=="neg" else "#34d399"
        tc2 = "#991b1b" if direction=="neg" else "#065f46"
        ic2 = "🔴" if direction=="neg" else "🟢"
        st.markdown(f"""
        <div style="background:{bg2};border:1px solid {bc2};border-radius:8px;
                    padding:10px 14px;margin:5px 0;">
          <strong style="color:{tc2};">{ic2} {feat}</strong>
          <div style="color:#374151;font-size:0.82rem;margin-top:2px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

with col_t:
    st.markdown("### 📐 Application Summary")
    cltv = (loan + mortdue) / max(pval, 1)
    rows = [
        ("Loan Amount",      f"${loan:,.0f}",        "—"),
        ("Property Value",   f"${pval:,.0f}",         "—"),
        ("Mortgage Due",     f"${mortdue:,.0f}",      "—"),
        ("LTV Ratio",        f"{r['ltv']*100:.1f}%",  "⚠️ High" if r['ltv']>0.80 else "✓ OK"),
        ("CLTV Ratio",       f"{cltv*100:.1f}%",      "🔴 High" if cltv>0.85 else "✓ OK"),
        ("OSFI B-20 Flag",   "⚠️ BREACH" if dti>44 else "✓ OK", "—"),
        ("Debt-to-Income",   f"{dti:.1f}%",           "—"),
        ("Employment",       f"{yoj:.1f} yrs",        "—"),
        ("Credit Age",       f"{clage} mths",         "—"),
        ("Derogatory Rpts",  str(derog),              "🔴 Adverse" if derog>0 else "✓ Clean"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric","Value","Status"]),
                 use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
<div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;
            font-size:0.74rem;color:#6b7280;">
⚖️ <strong>Regulatory Notice:</strong> This score is governed by OSFI E-23. Credit decisions
must comply with the <em>Canadian Human Rights Act</em>, PIPEDA, and OSFI B-20.
The model shall not be the sole basis for any credit decision.
</div>""", unsafe_allow_html=True)
