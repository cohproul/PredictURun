# PredictURun - CS/D' sur GPX (sécurisé LaTeX)
import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ================= Utilitaires temps & vitesses =================
def parse_time_to_seconds(txt: str):
    if txt is None:
        return None
    s = str(txt).strip().lower().replace("’","'").replace(",",".").replace("/km","")
    try:
        return float(s)
    except Exception:
        pass
    s = (s.replace("min.","min").replace("sec.","s").replace("secs","s").replace("sec","s")
           .replace("mn","min").replace("mns","min")
           .replace("mins","min").replace("minutes","min").replace("minute","min")
           .replace("seconds","s").replace("secondes","s").replace("seconde","s"))
    if ":" in s and "h" not in s:
        parts = s.split(":")
        if len(parts) == 2:
            m, ss = parts
            return int(m)*60 + float(ss)
        if len(parts) == 3:
            h, m, ss = parts
            return int(h)*3600 + int(m)*60 + float(ss)
    nums = [x for x in s.replace("min"," ").replace("s"," ").split() if x.replace('.','',1).isdigit()]
    nums = list(map(float, nums))
    if len(nums) == 1: return nums[0]
    if len(nums) == 2: return nums[0]*60 + nums[1]
    if len(nums) >= 3: return nums[0]*3600 + nums[1]*60 + nums[2]
    return None

def pace_from_v(v: float) -> str:
    if v is None or v <= 0 or np.isnan(v): return "-"
    sec_per_km = 1000.0 / v
    m = int(sec_per_km // 60)
    s = int(round(sec_per_km % 60))
    if s == 60: m += 1; s = 0
    return f"{m}:{s:02d}/km"

def format_hms(seconds: float) -> str:
    if seconds is None or np.isnan(seconds): return "-"
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(round(seconds % 60))
    if s == 60: m += 1; s = 0
    return f"{h}h{m:02d}m{s:02d}s" if h>0 else f"{m}m{s:02d}s"

# ================= Distances, GPX, pente =================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def load_gpx(file_bytes: bytes):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(file_bytes)
    pts = root.findall(".//{*}trkpt")
    lat = []; lon = []; ele = []
    for p in pts:
        la = float(p.attrib["lat"]); lo = float(p.attrib["lon"])
        e = p.find("{*}ele"); z = float(e.text) if e is not None else 0.0
        lat.append(la); lon.append(lo); ele.append(z)
    lat = np.array(lat); lon = np.array(lon); ele = np.array(ele)
    if len(lat) < 2: return None
    dist = [0.0]
    for i in range(1, len(lat)):
        dist.append(dist[-1] + haversine(lat[i-1], lon[i-1], lat[i], lon[i]))
    return np.array(lat), np.array(lon), np.array(ele), np.array(dist)

def resample_track(lat, lon, ele, dist, step_m=25.0):
    total = float(dist[-1]); s = np.arange(0.0, total + step_m, step_m)
    lat_i = np.interp(s, dist, lat)
    lon_i = np.interp(s, dist, lon)
    ele_i = np.interp(s, dist, ele)
    return lat_i, lon_i, ele_i, s

def moving_average(a, n=5):
    if n <= 1: return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    head = ret[n-1:] / n
    pad = np.concatenate([np.full(n-1, head[0]), head])
    return pad[:len(a)]

# ================= Modèles: CS/D', Température, Pente, Surface =================
def compute_cs_dprime_from_tests(t1000, t2000, t3200) -> Tuple[float, float]:
    D = np.array([1000.0, 2000.0, 3200.0])
    T = np.array([t1000, t2000, t3200], dtype=float)
    A = np.vstack([T, np.ones_like(T)]).T
    CS, Dprime = np.linalg.lstsq(A, D, rcond=None)[0]
    return float(CS), float(Dprime)

HOT_PENALTY_PER_DEG  = 0.0020  # -0.20%/°C au-dessus de 15
COLD_PENALTY_PER_DEG = 0.0015  # -0.15%/°C au-dessous de 15
def temperature_speed_factor(tempC: float) -> float:
    d = tempC - 15.0
    penalty = HOT_PENALTY_PER_DEG * d if d >= 0 else COLD_PENALTY_PER_DEG * (-d)
    return float(np.clip(1.0 - penalty, 0.70, 1.0))

def minetti_cost_factor(grade: float) -> float:
    i = max(-0.35, min(0.35, float(grade)))
    C = (155.4*(i**5) - 30.4*(i**4) - 43.3*(i**3) + 46.3*(i**2) + 19.5*i + 3.6)
    if abs(i) < 0.015:
        C = 3.6 + (C - 3.6) * 0.5
    return max(1e-3, C / 3.6)

def linear_grade_cost(grade, kup=6.0, kdown=2.0):
    return max(1e-3, 1.0 + (kup if grade>=0 else kdown) * grade)

def apply_dprime_weight_to_cost(base_cost: float, grade: float, dprime: float, weight: float) -> float:
    dp = min(1.0, max(0.0, dprime / 500.0)) * weight
    delta = base_cost - 1.0
    if grade >= 0: delta *= (1.0 - 0.6*dp)
    else:          delta *= (1.0 - 0.2*dp)
    return max(0.5, 1.0 + delta)

# ================= Simulation =================
@dataclass
class SimConfig:
    use_minetti: bool = True
    kup: float = 6.0
    kdown: float = 2.0
    surface_factor: float = 1.0
    tempC: float = 15.0
    dprime_relief_weight: float = 0.5
    target_arrival_reserve_pct: float = 10.0
    # (optionnel) si tu veux forcer une valeur précise :
    v1500_cap_mps: float | None = None   # None => on calcule depuis CS/D'/V0

@dataclass
class SimResult:
    total_time_s: float
    v_eq_target: float
    df: pd.DataFrame  # s_m, km, grade, cost, CS_i, V0_i, v, dt, ds, Dprime_bal

def simulate_course(lat_i, lon_i, ele_i, s_i,
                    CS, Dprime, V0,
                    cfg: SimConfig) -> SimResult:
    """
    - CS en m/s ; Dprime en m ; V0 en m/s (optionnel, peut être None)
    - Cap vitesse : v(t) <= v1500 (vitesse moyenne max prédite sur 1500 m)
    """
    # --- Ajustement température  ---
    fT = temperature_speed_factor(cfg.tempC)
    CS_adj = CS * fT
    V0_adj = (V0 * fT) if (V0 is not None) else None

    # ---------- Calcul du plafond v1500 à partir de CS/D' (+ V0 si fourni) ----------
    # Temps "modèle" sur 1500 m (sans autre plafond) : t_model = (D - D')/CS
    D_1500 = 1500.0
    t_model_1500 = max(0.0, (D_1500 - Dprime) / max(1e-6, CS_adj))
    # Si V0 existe, le temps moyen sur 1500 m ne peut pas être < 1500/V0
    if V0_adj is not None and V0_adj > 0:
        t1500 = max(t_model_1500, D_1500 / V0_adj)
    else:
        t1500 = t_model_1500 if t_model_1500 > 0 else float("inf")
    v1500_cap = (cfg.v1500_cap_mps if cfg.v1500_cap_mps is not None
                 else (D_1500 / t1500 if np.isfinite(t1500) and t1500 > 0 else float("inf")))

    # --- Géométrie parcours / pentes  ---
    ds = np.diff(s_i); de = np.diff(ele_i)
    valid = ds > 0.05
    ds = ds[valid]; de = de[valid]
    grades = moving_average(de / ds, 5)

    # --- Coûts (Minetti vs linéaire) + pondération D′ relief  ---
    base_cost = (np.array([minetti_cost_factor(g) for g in grades]) if cfg.use_minetti
                 else np.array([linear_grade_cost(g, cfg.kup, cfg.kdown) for g in grades]))
    cost = np.array([
        apply_dprime_weight_to_cost(c, g, Dprime, cfg.dprime_relief_weight)
        for c, g in zip(base_cost, grades)
    ])
    # Surface globale (facteur multiplicatif sur le coût)
    cost *= cfg.surface_factor

    # --- CS/V0 locaux  ---
    CS_i = CS_adj / cost
    V0_i = (V0_adj / cost) if (V0_adj is not None) else None

    # --- Cible de D' utilisé  ---
    target_used = Dprime * (1.0 - cfg.target_arrival_reserve_pct / 100.0)

    def simulate_for_c(c):
        # Vitesse cible locale avant plafonds
        v_raw = CS_i + c
        # Plafond V0 local si fourni
        if V0_i is not None:
            v_raw = np.minimum(v_raw, V0_i)
        # Plafond "vitesse max 1500 m" global
        v = np.minimum(v_raw, v1500_cap)

        # Sécurité
        v = np.maximum(v, 0.1)
        dt = ds / v

        # D' consommé (au-dessus de CS local)
        above = np.maximum(0.0, v - CS_i)
        used = float(np.sum(above * dt))

        T = float(np.sum(dt))
        return used, T, v, dt

    # Recherche de c pour atteindre la cible de D' utilisé 
    c_max = float(np.max(np.maximum(0.0, (V0_i if V0_i is not None else (CS_i + 5.0)) - CS_i)))
    low, high = 0.0, c_max
    used_high, _, _, _ = simulate_for_c(high)
    if used_high < target_used:
        used, T, v, dt = simulate_for_c(high)
        d_cum = np.cumsum(np.maximum(0.0, v - CS_i) * dt)
        Dbal = Dprime - d_cum
        s_mid = (s_i[:-1] + s_i[1:])[valid] / 2
        df = pd.DataFrame({
            "s_m": s_mid, "km": s_mid / 1000.0, "grade": grades, "cost": cost,
            "CS_i": CS_i, "V0_i": (V0_i if V0_i is not None else np.nan),
            "v": v, "dt": dt, "ds": ds, "Dprime_bal": Dbal
        })
        return SimResult(T, high, df)

    for _ in range(60):
        mid = 0.5 * (low + high)
        used_mid, _, _, _ = simulate_for_c(mid)
        if used_mid > target_used:
            high = mid
        else:
            low = mid
    c_star = 0.5 * (low + high)
    used, T, v, dt = simulate_for_c(c_star)

    d_cum = np.cumsum(np.maximum(0.0, v - CS_i) * dt)
    Dbal = Dprime - d_cum
    s_mid = (s_i[:-1] + s_i[1:])[valid] / 2
    df = pd.DataFrame({
        "s_m": s_mid, "km": s_mid / 1000.0, "grade": grades, "cost": cost,
        "CS_i": CS_i, "V0_i": (V0_i if V0_i is not None else np.nan),
        "v": v, "dt": dt, "ds": ds, "Dprime_bal": Dbal
    })
    return SimResult(T, c_star, df)


# ================= UI Streamlit =================
st.set_page_config(page_title="PredictURun - CS/D' GPX", layout="centered")
st.title("PredictURun - Optimisation de stratégies de course")

with st.sidebar:
    st.header("Paramètres")
    slope_model = st.radio("Modèle de pente", ["Minetti", "Linéaire"], index=0)
    if slope_model == "Linéaire":
        kup = st.slider("k_up (montée)", 0.0, 12.0, 6.0, 0.1)
        kdown = st.slider("k_down (descente)", 0.0, 12.0, 2.0, 0.1)
    else:
        kup, kdown = 6.0, 2.0

    tempC = st.number_input("Température course (°C)", value=15.0, step=0.5)
    st.markdown("### Surface (facteur global)")
    surface_options = {
        "Piste tartan": 1.00,
        "Route / Asphalte": 1.01,
        "Béton": 1.015,
        "Gravier / Chemin dur": 1.025,
        "Gazon": 1.06,
        "Trail roulant": 1.08,
        "Trail technique": 1.15,
        "Sable tassé": 1.20,
        "Sable meuble / Neige molle": 1.35,
    }
    surf_name = st.selectbox("Surface dominante", list(surface_options.keys()), index=1)
    surface_factor = surface_options[surf_name]

    dprime_relief_weight = st.slider("Impact D' sur pénalité de pente", 0.0, 1.0, 0.5, 0.05)
    reserve_arrivee = st.slider("D' restant à l’arrivée (%)", 0, 50, 10)

    st.subheader("Chronos de test")    
    c1, c2, c3, c4 = st.columns(4)
    with c1: t20 = st.text_input("20 m (s)", value=" ")
    with c2: t1000 = st.text_input("1000 m", value=" ")
    with c3: t2000 = st.text_input("2000 m", value=" ")
    with c4: t3200 = st.text_input("3200 m", value=" ")

    btn_calc = st.button("Calculer V0 / CS / D'")
    V0 = CS = Dprime = None
    if btn_calc:
        t20s = parse_time_to_seconds(t20)
        t1000s = parse_time_to_seconds(t1000)
        t2000s = parse_time_to_seconds(t2000)
        t3200s = parse_time_to_seconds(t3200)
        if None in (t20s, t1000s, t2000s, t3200s) or min(t20s, t1000s, t2000s, t3200s) <= 0:
            st.warning("Temps invalides.")
        else:
            V0 = 20.0 / t20s
            CS, Dprime = compute_cs_dprime_from_tests(t1000s, t2000s, t3200s)
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("V0 (m/s)", f"{V0:.3f}"); st.caption(pace_from_v(V0))
            with m2: st.metric("CS (m/s)", f"{CS:.3f}"); st.caption(pace_from_v(CS))
            with m3: st.metric("D′ (m)", f"{Dprime:.1f}")

st.markdown("---")
st.subheader("Fichier GPX (analyse au 20 m)")
gpx_file = st.file_uploader("Choisir un .gpx", type=["gpx"])

if gpx_file is not None:
    try:
        data = load_gpx(gpx_file.read())
    except Exception as e:
        data = None
        st.error(f"Lecture GPX impossible: {e}")
    if data is None:
        st.stop()

    lat, lon, ele, dist = data
    if len(lat) < 3 or dist[-1] < 20:
        st.error("GPX trop court/invalide.")
        st.stop()

    st.success(f"GPX chargé : {dist[-1]/1000:.2f} km - points bruts: {len(lat)}")
    lat_i, lon_i, ele_i, s_i = resample_track(lat, lon, ele, dist, 10.0)
    st.caption(f"Resample: {len(lat_i)} points (par 10 m)")

    # ---------------- Carte (couleurs version 10) ----------------
    path = [[float(lon_i[k]), float(lat_i[k])] for k in range(len(lon_i))]
    df_path = pd.DataFrame([{"path": path}])

    km_marks = np.arange(0.0, s_i[-1] + 1.0, 1000.0)
    idx = np.searchsorted(s_i, km_marks, side="left"); idx = np.clip(idx, 0, len(s_i)-1)
    df_km = pd.DataFrame({"lon": lon_i[idx], "lat": lat_i[idx], "km": [int(km/1000) for km in km_marks]})

    layers = [
        pdk.Layer("PathLayer", data=df_path, get_path="path", get_color=[121, 156, 19], width_min_pixels=4),   # vert
        pdk.Layer("ScatterplotLayer", data=df_km, get_position="[lon, lat]", get_radius=10, get_color=[0, 0, 255]),  # bleu
        pdk.Layer("TextLayer",
                  data=[{"position":[float(r["lon"]), float(r["lat"])], "text": str(int(r["km"]))} for _, r in df_km.iterrows()],
                  get_position="position", get_text="text", get_size=20, get_color=[255, 255, 255], get_alignment_baseline="'bottom'")  # blanc
    ]
    view_state = pdk.ViewState(latitude=float(np.mean(lat_i)), longitude=float(np.mean(lon_i)), zoom=12, pitch=45)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state))

    # ---------------- Profil altimétrique : X en mètres, ticks 500 m ----------------
    fig, ax = plt.subplots()
    x_m = s_i  # distance en mètres
    ax.plot(x_m, ele_i, linewidth=1.5, color="green")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Profil altimétrique")
    ax.xaxis.set_major_locator(MultipleLocator(1000))  # repères tous les 1000 m
    st.pyplot(fig, use_container_width=True)

    # ---------------- Simulation ----------------
    if V0 is None or CS is None or Dprime is None:
        st.info("Calculez d'abord V0 / CS / D'.")
    else:
        cfg = SimConfig(
            use_minetti=(slope_model=="Minetti"),
            kup=kup, kdown=kdown,
            surface_factor=float(surface_factor),
            tempC=float(tempC),
            dprime_relief_weight=float(dprime_relief_weight),
            target_arrival_reserve_pct=float(reserve_arrivee)
        )
        sim = simulate_course(lat_i, lon_i, ele_i, s_i, CS, Dprime, V0, cfg)
        st.subheader("Résultat")
        st.write(f"**Temps prédit** : {format_hms(sim.total_time_s)}  •  **c*** ≈ {sim.v_eq_target:.3f} m/s")

        cA, cB = st.columns(2)
        with cA:
            fig2, ax2 = plt.subplots()
            ax2.plot(sim.df["km"]*1000.0, sim.df["Dprime_bal"], color="green")  # X en mètres
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("D'bal (m)")
            ax2.set_title("Évolution de D'bal")
            ax2.xaxis.set_major_locator(MultipleLocator(500))  # repères 500 m
            st.pyplot(fig2, use_container_width=True)
        with cB:
            # --- Vitesse simulée en min:ss/km, X en mètres + ticks 500 m ---
            fig3, ax3 = plt.subplots()
            pace_sec = 1000.0 / np.maximum(1e-6, sim.df["v"].to_numpy())  # sec/km
            x_m_sim = (sim.df["km"].to_numpy() * 1000.0)  # mètres
            ax3.plot(x_m_sim, pace_sec, linewidth=1.5, color="green")
            ax3.set_xlabel("Distance (m)")
            ax3.set_ylabel("Allure (min:s / km)")
            ax3.set_title("Vitesse simulée")
            ax3.xaxis.set_major_locator(MultipleLocator(500))  # repères 500 m

            def _fmt_pace(sec_per_km, _pos):
                sec = float(sec_per_km)
                m = int(sec // 60)
                s = int(round(sec % 60))
                if s == 60:
                    m += 1
                    s = 0
                return f"{m}:{s:02d}"
                

            ax3.yaxis.set_major_formatter(FuncFormatter(_fmt_pace))
            st.pyplot(fig3, use_container_width=True)

        # ---------------- Tableau récapitulatif : tous les 400 m ----------------
        st.markdown("### Tableau récapitulatif (tous les 400 m)")
        df_seg = sim.df.copy()
        df_seg["cum_dist"] = df_seg["ds"].cumsum()
        df_seg["cum_time"] = df_seg["dt"].cumsum()

        rows = []
        target = 400.0
        total_dist = float(df_seg["cum_dist"].iloc[-1])
        while target <= total_dist + 1e-6:
            start_d = max(0.0, target - 400.0)
            win = df_seg[(df_seg["cum_dist"] > start_d + 1e-9) & (df_seg["cum_dist"] <= target + 1e-9)]
            if not win.empty:
                dist_m = float(win["ds"].sum())
                time_s = float(win["dt"].sum())
                v_bin = dist_m / max(1e-6, time_s)
                pace_sec = 1000.0 / max(1e-6, v_bin)
                mP = int(pace_sec // 60); sP = int(round(pace_sec % 60))
                if sP == 60: mP += 1; sP = 0
                t_elapsed = float(win["cum_time"].iloc[-1])
                h = int(t_elapsed // 3600); mE = int((t_elapsed % 3600)//60); sE = int(round(t_elapsed % 60))
                if sE == 60: mE += 1; sE = 0
                rows.append({
                    "Distance (m)": int(round(target)),
                    "Allure (min:s/km)": f"{mP}:{sP:02d}",
                    "Temps écoulé": (f"{h:d}h{mE:02d}m{sE:02d}s" if h>0 else f"{mE:02d}m{sE:02d}s")
                })
            target += 400.0
        table100 = pd.DataFrame(rows)
        st.dataframe(table100, use_container_width=True)

        # ---------------- Explications complètes + exemples numériques ----------------
        with st.expander("📘 Explications des calculs (formules + exemples avec vos paramètres)"):
            st.markdown("**Paramètres issus de vos tests**")
            st.markdown(f"- V0 = **{V0:.3f} m/s**")
            st.markdown(f"- CS = **{CS:.3f} m/s**")
            st.markdown(f"- D' = **{Dprime:.1f} m**")
            st.markdown(f"- Température course T = **{tempC:.1f}°C** ⇒ facteur température $f_T = {temperature_speed_factor(tempC):.3f}$")
            st.markdown(f"- Surface « {surf_name} » ⇒ facteur coût **{surface_factor:.3f}**")
            st.markdown(f"- Pondération D′ (curseur) = **{dprime_relief_weight:.2f}**")
            st.markdown(f"- Réserve visée à l’arrivée = **{reserve_arrivee}%**")

            st.markdown("### Régression CS/D' à partir des tests (1000/2000/3200)")
            st.latex(r"d = CS \cdot t + D'")

            st.markdown("### Température (optimum 15 °C)")
            st.latex(r"""
f_T = 1 - p \cdot |T-15|,\quad
p = \begin{cases}
0.0020 & T \ge 15\\
0.0015 & T < 15
\end{cases}
""")
            st.markdown(f"Donc $CS_T = CS \\cdot f_T = {CS*temperature_speed_factor(tempC):.3f}$ m/s et $V0_T = V0 \\cdot f_T = {V0*temperature_speed_factor(tempC):.3f}$ m/s.")

            st.markdown("### Relief (coût de pente)")
            st.latex(r"C(i) = 155.4\,i^5 - 30.4\,i^4 - 43.3\,i^3 + 46.3\,i^2 + 19.5\,i + 3.6,\quad \text{facteur} = \frac{C(i)}{3.6}")
            st.markdown("Modèle linéaire optionnel :")
            st.latex(r"\text{facteur} = \begin{cases} 1 + k_\uparrow\,i & (i\ge0) \\ 1 + k_\downarrow\,i & (i<0) \end{cases}")

            st.markdown("### Pondération par D' sur la pénalité de pente")
            st.latex(r"""
\Delta = (\text{facteur}-1),\quad
\Delta \leftarrow
\begin{cases}
\Delta\,(1-0.6\,w) & \text{montée}\\
\Delta\,(1-0.2\,w) & \text{descente}
\end{cases},\quad
w = \min(1, D'/500)\cdot \text{poids}
""")
            st.markdown(f"Facteur final $C_f = 1 + \Delta$, puis *surface* = {surface_factor:.3f}.")
            st.latex(r"CS_i = \frac{CS_T}{C_f},\quad V0_i = \frac{V0_T}{C_f}")

            st.markdown("### Surplus constant et simulation D'bal")
            st.latex(r"v = \min(V0_i,\; CS_i + c^*),\qquad dt = \frac{ds}{v}")
            st.latex(r"\text{usage D'} = \sum (v - CS_i)\,dt,\qquad D'_{\text{bal}} = D' - \text{usage cumulé}")

            st.markdown("### Tableau récapitulatif (tous les 100 m)")
            st.latex(r"v_{400} = \frac{400}{\sum dt_{(400m)}},\qquad \text{allure} = \frac{1000}{v_{400}}\; (\text{min:s/km})")
            st.markdown("On affiche aussi le **temps écoulé** à la fin de chaque 400 m.")

st.caption("Modèle: CS/D' + température (optimum 15 °C) + surface + pente (Minetti/linéaire) pondéré par D', par 20 m.")

st.caption("Créé par Charles livier Huapaya Proulx")
