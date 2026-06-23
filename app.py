import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Engine Cycle Simulator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.4rem !important; }
    .stMetric label { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── ISA atmosphere ────────────────────────────────────────────────────────────
def isa_atmosphere(alt_ft: float) -> tuple[float, float]:
    """Return (T_K, P_kPa) for ISA standard atmosphere at given altitude (ft)."""
    h = alt_ft * 0.3048          # ft → m
    T0, P0, L, R, g = 288.15, 101.325, 0.0065, 287.05, 9.80665
    if h <= 11_000:
        T = T0 - L * h
        P = P0 * (T / T0) ** (g / (L * R))
    elif h <= 20_000:
        T = 216.65
        P11 = P0 * (216.65 / T0) ** (g / (L * R))
        P = P11 * np.exp(-g * (h - 11_000) / (R * T))
    else:
        T = 216.65
        P11 = P0 * (216.65 / T0) ** (g / (L * R))
        P20 = P11 * np.exp(-g * 9_000 / (R * T))
        P = P20 * np.exp(-g * (h - 20_000) / (R * T))
    return T, P

# ── Cycle computations ────────────────────────────────────────────────────────
def compute_otto(r, gamma, T1, P1, T3_max=2500.0):
    cv, cp = 718.0, 718.0 * gamma
    R_air = cp - cv
    T2 = T1 * r ** (gamma - 1);  P2 = P1 * r ** gamma
    T3 = T3_max                   # fixed peak temp (materials/knock limit)
    qin = cv * (T3 - T2) / 1e3   # heat addition varies with inlet conditions
    P3 = P2 * T3 / T2
    T4 = T3 / r ** (gamma - 1);  P4 = P3 / r ** gamma
    qout = cv * (T4 - T1) / 1e3
    wnet = qin - qout
    eta  = 1 - 1 / r ** (gamma - 1)
    v1   = R_air * T1 / (P1 * 1e3);  v2 = v1 / r
    mep  = wnet * 1e3 / (v1 - v2) / 1e3

    n = 60
    # 1→2 isentropic compression
    v_12 = np.linspace(v1, v2, n)
    p_12 = P1 * (v1 / v_12) ** gamma
    # 2→3 const-V heat addition
    p_23 = np.linspace(P2, P3, 10);  v_23 = np.full(10, v2)
    # 3→4 isentropic expansion
    v_34 = np.linspace(v2, v1, n)
    p_34 = P3 * (v2 / v_34) ** gamma
    # 4→1 const-V heat rejection
    p_41 = np.linspace(P4, P1, 10);  v_41 = np.full(10, v1)

    pv = dict(
        v=np.concatenate([v_12, v_23, v_34, v_41]) * 1e3,
        p=np.concatenate([p_12, p_23, p_34, p_41]),
        seg=(['1→2 Compress']*n + ['2→3 Heat Add']*10 +
             ['3→4 Expand']*n   + ['4→1 Heat Rej']*10)
    )
    return dict(eta=eta, wnet=wnet, T1=T1, T2=T2, T3=T3, T4=T4,
                P1=P1, P2=P2, P3=P3, P4=P4, qin=qin, qout=qout,
                mep=mep, pv=pv, v1=v1, v2=v2, R=R_air, cv=cv, cp=cp,
                phases=['1→2 Compress','2→3 Heat Add','3→4 Expand','4→1 Heat Rej'])


def compute_diesel(r, rc, gamma, T1, P1, T3_max=2500.0):
    cv, cp = 718.0, 718.0 * gamma
    R_air = cp - cv
    T2 = T1 * r ** (gamma - 1);      P2 = P1 * r ** gamma
    T3 = T3_max                        # fixed peak temp
    rc = T3 / T2                       # cutoff ratio now derived from T3 and T2
    rc = max(rc, 1.001)                # guard against rc < 1 at very high compression
    P3 = P2
    T4 = T3 * (rc / r) ** (gamma - 1); P4 = P3 * (rc / r) ** gamma
    qin  = cp * (T3 - T2) / 1e3
    qout = cv * (T4 - T1) / 1e3
    wnet = qin - qout
    eta  = 1 - (1 / r ** (gamma - 1)) * (rc**gamma - 1) / (gamma * (rc - 1))
    v1   = R_air * T1 / (P1 * 1e3);  v2 = v1 / r;  v3 = v2 * rc
    mep  = wnet * 1e3 / (v1 - v2) / 1e3

    n = 60
    v_12 = np.linspace(v1, v2, n);   p_12 = P1 * (v1 / v_12) ** gamma
    v_23 = np.linspace(v2, v3, 20);  p_23 = np.full(20, P2)
    v_34 = np.linspace(v3, v1, n);   p_34 = P3 * (v3 / v_34) ** gamma
    p_41 = np.linspace(P4, P1, 10);  v_41 = np.full(10, v1)

    pv = dict(
        v=np.concatenate([v_12, v_23, v_34, v_41]) * 1e3,
        p=np.concatenate([p_12, p_23, p_34, p_41]),
        seg=(['1→2 Compress']*n + ['2→3 Heat Add']*20 +
             ['3→4 Expand']*n   + ['4→1 Heat Rej']*10)
    )
    return dict(eta=eta, wnet=wnet, T1=T1, T2=T2, T3=T3, T4=T4,
                P1=P1, P2=P2, P3=P3, P4=P4, qin=qin, qout=qout,
                mep=mep, pv=pv, v1=v1, v2=v2, R=R_air, cv=cv, cp=cp,
                phases=['1→2 Compress','2→3 Heat Add','3→4 Expand','4→1 Heat Rej'])


def compute_brayton(pr, gamma, T1, P1, T3_max=1600.0):
    cp = 1005.0;  cv = cp / gamma;  R_air = cp - cv
    T2 = T1 * pr ** ((gamma - 1) / gamma);  P2 = P1 * pr
    T3 = T3_max                              # fixed turbine inlet temp limit
    qin = cp * (T3 - T2) / 1e3              # varies with compressor exit temp
    P3 = P2
    T4 = T3 / pr ** ((gamma - 1) / gamma);  P4 = P1
    qout = cp * (T4 - T1) / 1e3
    wnet = qin - qout
    eta  = 1 - 1 / pr ** ((gamma - 1) / gamma)

    N = 80
    def bray_pv(Ta, Tb, Pa, Pb, phase_const_p=False):
        if phase_const_p:
            Ts = np.linspace(Ta, Tb, N)
            Ps = np.full(N, Pa)
        else:
            Ts = np.linspace(Ta, Tb, N)
            Ps = Pa * (Ts / Ta) ** (gamma / (gamma - 1))
        Vs = R_air * Ts / (Ps * 1e3)
        return Vs * 1e3, Ps

    v12, p12 = bray_pv(T1, T2, P1, P2)
    v23, p23 = bray_pv(T2, T3, P2, P3, phase_const_p=True)
    v34, p34 = bray_pv(T3, T4, P3, P4)
    v41, p41 = bray_pv(T4, T1, P1, P1, phase_const_p=True)

    pv = dict(
        v=np.concatenate([v12, v23, v34, v41]),
        p=np.concatenate([p12, p23, p34, p41]),
        seg=(['1→2 Compress']*N + ['2→3 Heat Add']*N +
             ['3→4 Expand']*N   + ['4→1 Heat Rej']*N)
    )
    return dict(eta=eta, wnet=wnet, T1=T1, T2=T2, T3=T3, T4=T4,
                P1=P1, P2=P2, P3=P3, P4=P4, qin=qin, qout=qout,
                mep=None, pv=pv, v1=None, v2=None, R=R_air, cv=cv, cp=cp,
                phases=['1→2 Compress','2→3 Heat Add','3→4 Expand','4→1 Heat Rej'])


def compute_cycle(cycle, r, rc, pr, gamma, T1, P1, T3_max=None):
    if cycle == 'Otto':
        return compute_otto(r, gamma, T1, P1, T3_max or 2500.0)
    if cycle == 'Diesel':
        return compute_diesel(r, rc, gamma, T1, P1, T3_max or 2500.0)
    return compute_brayton(pr, gamma, T1, P1, T3_max or 1600.0)


def ts_traces(res, gamma):
    """Return list of (s_array, T_array, label) for each phase.

    Entropy reference: s=0 at state 1.
    Isentropic processes (1→2, 3→4) have ds=0.
    Heat addition (2→3) and rejection (4→1) are computed from cv/cp·ln(T).
    The 4→1 segment is forced to close back at s=0, T1 so the loop
    is exact regardless of floating-point accumulation.
    """
    cv, cp = res['cv'], res['cp']
    T1, T2, T3, T4 = res['T1'], res['T2'], res['T3'], res['T4']

    s1 = 0.0                                  # reference
    s2 = s1                                   # 1→2 isentropic: ds = 0
    s3 = s2 + cp * np.log(T3 / T2) / 1e3     # 2→3 heat addition
    s4 = s3                                   # 3→4 isentropic: ds = 0
    # 4→1 must close back to s1=0 exactly — force it rather than recompute
    s1_close = s1

    pts = [
        (s1, T1, s2, T2, '1→2 Compress'),
        (s2, T2, s3, T3, '2→3 Heat Add'),
        (s3, T3, s4, T4, '3→4 Expand'),
        (s4, T4, s1_close, T1, '4→1 Heat Rej'),
    ]
    out = []
    for (sa, Ta, sb, Tb, label) in pts:
        ss = np.linspace(sa, sb, 40)
        Ts = np.linspace(Ta, Tb, 40)
        out.append((ss, Ts, label))
    return out

# ── Color map ─────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    '1→2 Compress': '#58a6ff',
    '2→3 Heat Add':  '#3fb950',
    '3→4 Expand':    '#ff7b72',
    '4→1 Heat Rej':  '#d29922',
}
SL_COLOR = 'rgba(255,255,255,0.2)'

# ── Plotting ──────────────────────────────────────────────────────────────────
DARK = '#0d1117'
GRID = '#21262d'
TICK = '#8b949e'

def base_layout(xtitle, ytitle, title=""):
    return dict(
        paper_bgcolor=DARK, plot_bgcolor='#161b22',
        font=dict(color=TICK, size=11, family='monospace'),
        title=dict(text=title, font=dict(size=12, color='#c9d1d9'), x=0.02),
        xaxis=dict(title=xtitle, gridcolor=GRID, zerolinecolor=GRID, color=TICK),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor=GRID, color=TICK),
        margin=dict(l=55, r=15, t=35, b=45),
        showlegend=False,
    )


def pv_figure(res, sl=None, show_ref=False):
    fig = go.Figure()
    segs = {ph: {'v': [], 'p': []} for ph in res['phases']}
    for v, p, seg in zip(res['pv']['v'], res['pv']['p'], res['pv']['seg']):
        segs[seg]['v'].append(v); segs[seg]['p'].append(p)
    for ph in res['phases']:
        fig.add_trace(go.Scatter(x=segs[ph]['v'], y=segs[ph]['p'],
            mode='lines', name=ph, line=dict(color=PHASE_COLORS[ph], width=2)))
    if show_ref and sl:
        fig.add_trace(go.Scatter(x=sl['pv']['v'], y=sl['pv']['p'],
            mode='lines', name='Sea Level ref',
            line=dict(color=SL_COLOR, width=1.5, dash='dash')))
    fig.update_layout(**base_layout('V  (L/kg)', 'P  (kPa)', 'P-V Diagram'))
    return fig


def ts_figure(res, sl=None, gamma=1.4, show_ref=False):
    fig = go.Figure()
    for ss, Ts, label in ts_traces(res, gamma):
        fig.add_trace(go.Scatter(x=ss, y=Ts, mode='lines', name=label,
            line=dict(color=PHASE_COLORS[label], width=2)))
    if show_ref and sl:
        for ss, Ts, label in ts_traces(sl, gamma):
            fig.add_trace(go.Scatter(x=ss, y=Ts, mode='lines',
                name=f'{label} (SL)', line=dict(color=SL_COLOR, width=1.5, dash='dash')))
    fig.update_layout(**base_layout('ΔS  (kJ/kg·K)', 'T  (K)', 'T-S Diagram'))
    return fig


def eta_figure(gamma, rc, cycle):
    rs = np.arange(4, 25.1, 0.25)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs,
        y=(1 - 1/rs**(gamma-1))*100,
        mode='lines', name='Otto', line=dict(color='#58a6ff', width=2)))
    fig.add_trace(go.Scatter(x=rs,
        y=(1 - (1/rs**(gamma-1))*(rs**0/(gamma*(rc-1))*(rc**gamma-1)))*100,
        mode='lines', name='Diesel', line=dict(color='#3fb950', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=rs,
        y=(1 - 1/rs**((gamma-1)/gamma))*100,
        mode='lines', name='Brayton (PR)', line=dict(color='#d29922', width=2, dash='dot')))
    fig.update_layout(**base_layout('r  (or PR for Brayton)', 'η  (%)',
                                    'Thermal Efficiency vs Compression Ratio'))
    fig.update_layout(showlegend=True,
        legend=dict(bgcolor='#161b22', bordercolor=GRID, font=dict(size=10)),
        yaxis=dict(range=[0, 80], gridcolor=GRID, zerolinecolor=GRID, color=TICK,
                   title='η  (%)'))
    return fig


def energy_figure(res, sl=None, show_ref=False):
    labels = ['Q_in', 'W_net', 'Q_out']
    vals   = [res['qin'], res['wnet'], res['qout']]
    colors      = ['rgba(63,185,80,0.35)',  'rgba(88,166,255,0.35)',  'rgba(255,123,114,0.35)']
    line_colors = ['rgba(63,185,80,1)',     'rgba(88,166,255,1)',     'rgba(255,123,114,1)']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current', x=labels, y=vals,
        marker_color=colors,
        marker_line_color=line_colors, marker_line_width=2))
    if show_ref and sl:
        sl_vals = [sl['qin'], sl['wnet'], sl['qout']]
        fig.add_trace(go.Bar(name='Sea Level ref', x=labels, y=sl_vals,
            marker_color=['rgba(255,255,255,0.08)']*3,
            marker_line_color=['rgba(255,255,255,0.3)']*3,
            marker_line_width=1.5))
    fig.update_layout(**base_layout('', 'kJ/kg', 'Energy Balance'))
    fig.update_layout(showlegend=show_ref, barmode='group',
        legend=dict(bgcolor='#161b22', bordercolor=GRID, font=dict(size=10)))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Engine Cycle Simulator")
    st.markdown("*1D Thermodynamic Analysis*")
    st.divider()

    cycle = st.radio("Cycle", ['Otto', 'Diesel', 'Brayton'], horizontal=True)
    st.divider()

    st.markdown("**Cycle Parameters**")
    if cycle in ('Otto', 'Diesel'):
        r = st.slider("Compression Ratio (r)", 4.0, 25.0, 10.0, 0.5)
    else:
        r = 10.0
        st.slider("Compression Ratio (r)", 4.0, 25.0, 10.0, 0.5, disabled=True)

    if cycle == 'Diesel':
        rc = st.slider("Cutoff Ratio (r_c)", 1.1, 4.0, 2.0, 0.1)
    else:
        rc = 2.0
        st.slider("Cutoff Ratio (r_c)", 1.1, 4.0, 2.0, 0.1, disabled=True)

    if cycle == 'Brayton':
        pr = st.slider("Pressure Ratio (PR)", 3, 40, 10, 1)
    else:
        pr = 10
        st.slider("Pressure Ratio (PR)", 3, 40, 10, 1, disabled=True)

    st.divider()
    st.markdown("**Atmospheric Conditions**")
    alt = st.slider("Altitude (ft)", 0, 45_000, 0, 500)

    T_isa, P_isa = isa_atmosphere(alt)
    rho_ratio = (P_isa / 101.325) * (288.15 / T_isa)

    if alt > 500:
        st.info(
            f"**ISA @ {alt:,} ft**  \n"
            f"T₁ = {T_isa:.1f} K  \n"
            f"P₁ = {P_isa:.1f} kPa ({P_isa/101.325*100:.0f}% of SL)  \n"
            f"ρ/ρ₀ = {rho_ratio:.3f}  →  ~{(1-rho_ratio)*100:.0f}% density loss"
        )
        T1, P1 = T_isa, P_isa
    else:
        T1 = st.slider("T₁ Inlet Temp (K)", 200, 400, 288, 5)
        P1 = st.slider("P₁ Inlet Press (kPa)", 10.0, 200.0, 101.3, 1.0)

    gamma = st.slider("γ (Cp/Cv)", 1.30, 1.67, 1.40, 0.01)

    st.divider()
    st.markdown("**Peak Temperature Limit (T₃)**")
    if cycle == 'Brayton':
        T3_max = st.slider("T₃ max — turbine inlet limit (K)", 800, 2000, 1600, 50)
        st.caption("Represents turbine blade material limit. T₃ is fixed; Q_in varies with altitude.")
    else:
        T3_max = st.slider("T₃ max — knock/materials limit (K)", 1500, 3500, 2500, 50)
        st.caption("Represents knock or materials limit. T₃ is fixed; Q_in varies with altitude.")
    st.divider()
    st.markdown("*Sea-level reference traces appear on P-V and T-S diagrams when altitude > 500 ft.*")

# ── Compute ───────────────────────────────────────────────────────────────────
show_ref = alt > 500
SL_T, SL_P = isa_atmosphere(0)
res = compute_cycle(cycle, r, rc, pr, gamma, T1, P1, T3_max)
sl  = compute_cycle(cycle, r, rc, pr, gamma, SL_T, SL_P, T3_max)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Engine Cycle Thermodynamic Simulator")
st.markdown(
    f"**Cycle:** {cycle} &nbsp;|&nbsp; "
    f"**Alt:** {alt:,} ft &nbsp;|&nbsp; "
    f"**T₁:** {T1:.0f} K &nbsp;|&nbsp; "
    f"**P₁:** {P1:.1f} kPa"
)

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Thermal Efficiency", f"{res['eta']*100:.1f}%")
m2.metric("Net Work", f"{res['wnet']:.0f} kJ/kg")
m3.metric("Peak Temp T₃", f"{res['T3']:.0f} K")
m4.metric("Peak Press P₃", f"{res['P3']:.0f} kPa")
m5.metric("MEP", f"{res['mep']:.0f} kPa" if res['mep'] else "N/A")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(pv_figure(res, sl, show_ref), width="stretch")
with col2:
    st.plotly_chart(ts_figure(res, sl, gamma, show_ref), width="stretch")

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(eta_figure(gamma, rc, cycle), width="stretch")
with col4:
    st.plotly_chart(energy_figure(res, sl, show_ref), width="stretch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Ideal cycle analysis · ISA atmosphere model · "
    "Fixed peak temperature T₃ (materials/knock limit) — Q_in varies with altitude · "
    "Built with Streamlit + Plotly · "
    "© 2026 Anthony LoCurto.  All Rights Reserved.</small>",
    unsafe_allow_html=True
)
