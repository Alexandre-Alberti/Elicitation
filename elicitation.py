# -*- coding: utf-8 -*-
"""
Elicitation for delay-time model (Streamlit)

Correções principais:
- Evita zeros em tempos padrão (log(0)).
- 'Clampe' de probabilidades para (0,1) antes de log/log.
- Checagens de finitude e cardinalidade antes do polyfit.
- Monotonicidade e limites válidos para s_prob.
- Mensagens de erro/aviso mais claras no Streamlit.

Autor original: alexa
Ajustes: ChatGPT (2025-10-28)
"""

import streamlit as st
import numpy as np
import random as rd
from scipy.integrate import dblquad

# =========================
# Utilidades numéricas
# =========================
EPS = 1e-9

def safe_clip01(p, eps=EPS):
    """Clampa probabilidades para (eps, 1-eps)."""
    return float(np.clip(p, eps, 1 - eps))

def safe_log_weibull_transform_probs(p_failure_array):
    """
    Recebe array de probabilidades de falha (0,1).
    Retorna y = log(-log(1 - p)), com clamp e filtro de finitos.
    """
    p = np.asarray(p_failure_array, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)
    y = np.log(-np.log(1.0 - p))
    mask = np.isfinite(y)
    return y, mask

def ensure_finite_pair(x, y):
    """Aplica máscara de finitude a x e y e garante pelo menos 2 pontos."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], mask.sum() >= 2

# =========================
# Densidades (Z ~ Weibull(forma, escala); H ~ Exp(1/d_medio))
# =========================
def fz(z, forma, escala):
    if z < 0 or forma <= 0 or escala <= 0:
        return 0.0
    return (forma / escala) * ((z / escala) ** (forma - 1)) * np.exp(-((z / escala) ** forma))

def fh(h, d_medio):
    if h < 0 or d_medio <= 0:
        return 0.0
    return (1.0 / d_medio) * np.exp(- (1.0 / d_medio) * h)

# =========================
# UI
# =========================
st.title("Elicitation for delay-time model")

unit = st.text_input("Time unit (e.g., hours, days, weeks):", key="unit")
TM = st.number_input(f"Average time to system failure ({unit})", min_value=0.0, format="%.2f", key="TM")
DM = st.number_input(f"Average delay-time (defect detection window) ({unit}):", min_value=0.0, format="%.2f", key="DM")
ID = st.number_input("Imprecision in delay-time estimation (%)", min_value=0.0, format="%.2f", key="ID")

qualitative_options = {
    "Very High (almost certain) – 90 to 100%": (90.0, 99.99),
    "High – 75 to 90%": (75.0, 90.0),
    "Medium – High – 60 to 75%": (60.0, 75.0),
    "Medium – 40 to 60%": (40.0, 60.0),
    "Medium – Low – 25 to 40%": (25.0, 40.0),
    "Low – 10 to 25%": (10.0, 25.0),
    "Very Low (unlikely) – 0 to 10%": (0.01, 10.0),
}

def get_time_points(TM_value):
    """Gera 5 pontos de tempo positivos, sem cast para int, com piso EPS."""
    raw_points = [0.1 * TM_value, 0.5 * TM_value, TM_value, 1.5 * TM_value, 2.0 * TM_value]
    return [max(float(p), EPS) for p in raw_points]

def get_matrix_from_state(time_points):
    """
    Lê as respostas do usuário em st.session_state (resp_0..resp_4) e monta:
    - matrix_t: array dos tempos (float)
    - matrix_p: matriz 5x2 com limites percentuais (min, max) para sobrevivência
    Retorna (matrix_t, matrix_p) ou levanta ValueError se algo faltar.
    """
    matrix_t = np.zeros(5, dtype=float)
    matrix_p = np.zeros((5, 2), dtype=float)
    for i in range(5):
        key = f"resp_{i}"
        if key not in st.session_state:
            raise ValueError("Você precisa selecionar todas as probabilidades qualitativas antes de estimar.")
        response = st.session_state[key]
        if response not in qualitative_options:
            raise ValueError("Seleção de probabilidade inválida ou ausente.")
        matrix_t[i] = time_points[i]
        lo, hi = qualitative_options[response]
        matrix_p[i, 0] = float(lo)
        matrix_p[i, 1] = float(hi)
    return matrix_t, matrix_p

# Coleta das probabilidades (mostra as caixas quando TM > 0)
if TM > 0:
    tp = get_time_points(TM)
    st.markdown("### Estimated survival probabilities:")
    for i in range(5):
        label_t = f"{tp[i]:.2f}"
        st.selectbox(
            f"Chance that the system remains operational after {label_t} {unit}",
            list(qualitative_options.keys()),
            key=f"resp_{i}"
        )

# =========================
# Botão de cálculo
# =========================
if st.button("Estimate parameters"):
    # -------------------------
    # Validações iniciais
    # -------------------------
    if TM <= 0:
        st.error("Informe um TM > 0.")
        st.stop()
    if DM <= 0:
        st.error("Informe um DM > 0.")
        st.stop()
    if ID < 0:
        st.error("A imprecisão (ID) deve ser ≥ 0%.")
        st.stop()

    # Reconstrói pontos de tempo e coleta respostas
    time_points = get_time_points(TM)
    try:
        matrix_t, matrix_p = get_matrix_from_state(time_points)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # -------------------------
    # Monte Carlo para Z ~ Weibull (tempo até falha)
    # -------------------------
    n_mc = 1000
    sample_eta = []
    sample_beta = []

    for i in range(n_mc):
        # Sobrevivência percentual (s_prob, decrescente/monótona)
        s_prob = np.zeros(5, dtype=float)
        f_prob = np.zeros(5, dtype=float)

        # Primeiro ponto
        lo0, hi0 = matrix_p[0, 0], matrix_p[0, 1]
        if hi0 <= lo0:
            s_prob[0] = lo0
        else:
            s_prob[0] = rd.uniform(lo0, hi0)

        f_prob[0] = (100.0 - s_prob[0]) / 100.0

        # Demais pontos com limite superior = min(s_anterior, hi_j)
        ok_sequence = True
        for j in range(1, 5):
            lo_j, hi_j = matrix_p[j, 0], matrix_p[j, 1]
            lim_sup = min(s_prob[j - 1], hi_j)
            lim_inf = lo_j
            if lim_sup <= lim_inf:
                # Se invertido, fixe no limite inferior (mantém monotonicidade)
                s_prob[j] = lim_inf
            else:
                s_prob[j] = rd.uniform(lim_inf, lim_sup)

            f_prob[j] = (100.0 - s_prob[j]) / 100.0
            # Se toda sobrevivência for 100% (falha 0%) ou 0% (falha 100%), ajuste depois
            if not (0.0 <= f_prob[j] <= 1.0):
                ok_sequence = False
                break

        if not ok_sequence:
            continue

        # Clamp e transformação
        f_prob = np.clip(f_prob, EPS, 1 - EPS)
        x_input = np.log(matrix_t.astype(float))
        y_input = np.log(-np.log(1.0 - f_prob))

        x_fit, y_fit, ok = ensure_finite_pair(x_input, y_input)
        if not ok:
            continue

        # Ajuste linear: y = A*x + B
        try:
            A, B = np.polyfit(x_fit, y_fit, 1)
        except Exception:
            continue

        beta_hat = A
        if not np.isfinite(beta_hat) or beta_hat <= 0:
            continue

        eta_hat = float(np.exp(-B / beta_hat))
        if not np.isfinite(eta_hat) or eta_hat <= 0:
            continue

        sample_beta.append(beta_hat)
        sample_eta.append(eta_hat)

    if len(sample_beta) < 10:
        st.error("Não foi possível ajustar a Weibull de Z com estabilidade suficiente. Tente respostas menos extremas nas probabilidades qualitativas.")
        st.stop()

    sample_beta = np.asarray(sample_beta)
    sample_eta = np.asarray(sample_eta)

    beta_25 = np.percentile(sample_beta, 25)
    beta_75 = np.percentile(sample_beta, 75)
    beta_central = 0.5 * (beta_25 + beta_75)
    beta_imprecisao = 100.0 * (beta_75 - beta_central) / max(beta_central, EPS)

    eta_25 = np.percentile(sample_eta, 25)
    eta_75 = np.percentile(sample_eta, 75)
    eta_central = 0.5 * (eta_25 + eta_75)
    eta_imprecisao = 100.0 * (eta_75 - eta_central) / max(eta_central, EPS)

    st.markdown("### Time to Failure – Weibull Parameters")
    st.write(f"**Shape (beta):** {beta_central:.2f} (IQR: {beta_25:.2f} – {beta_75:.2f})")
    st.write(f"**Scale (eta):** {eta_central:.2f} {unit} (IQR: {eta_25:.2f} – {eta_75:.2f})")
    st.write(f"**Beta relative imprecision:** {beta_imprecisao:.2f}%")
    st.write(f"**Eta relative imprecision:** {eta_imprecisao:.2f}%")

    # -------------------------
    # Estimativa para X = Z - H (tempo até defeito)
    # -------------------------
    # Se o intervalo interquartil de Z ficar degenerado, não dá para amostrar X
    if not (np.isfinite(beta_25) and np.isfinite(beta_75) and beta_75 > beta_25 and
            np.isfinite(eta_25) and np.isfinite(eta_75) and eta_75 > eta_25):
        st.warning("Intervalo interquartil de Z degenerado. Não é possível estimar X (= Z − H) com estabilidade.")
        st.stop()

    n_k = 150  # número de amostras para X (pode ajustar)
    sample_eta_x = []
    sample_beta_x = []

    for k in range(n_k):
        escala = rd.uniform(float(eta_25), float(eta_75))
        forma = rd.uniform(float(beta_25), float(beta_75))

        # DM com imprecisão relativa ID (%)
        lo_dm = max((1.0 - ID / 100.0) * DM, EPS)
        hi_dm = (1.0 + ID / 100.0) * DM
        if hi_dm <= lo_dm:
            d_medio = lo_dm
        else:
            d_medio = rd.uniform(lo_dm, hi_dm)

        # Construção de pontos (t) para ajuste (evitar 0)
        x_input = np.zeros(30, dtype=float)
        y_input = np.zeros(30, dtype=float)
        ini = 0.0

        valid_curve = True
        for i in range(30):
            ini += 0.1  # 0.1, 0.2, ..., 3.0
            t = ini * escala
            x_input[i] = np.log(max(t, EPS))

            # Probabilidade acumulada "observada" para (Z - H) <= t
            try:
                area = dblquad(
                    lambda h, z: fz(z, forma, escala) * fh(h, d_medio),
                    t, 10.0 * escala,
                    lambda z: 0.0, lambda z: z - t
                )[0]
            except Exception:
                valid_curve = False
                break

            prob = 1.0 - float(area)
            prob = safe_clip01(prob)

            y_input[i] = np.log(-np.log(1.0 - prob))

        if not valid_curve:
            continue

        x_fit, y_fit, ok = ensure_finite_pair(x_input, y_input)
        if not ok:
            continue

        try:
            A, B = np.polyfit(x_fit, y_fit, 1)
        except Exception:
            continue

        beta_x = A
        if not np.isfinite(beta_x) or beta_x <= 0:
            continue

        eta_x = float(np.exp(-B / beta_x))
        if not np.isfinite(eta_x) or eta_x <= 0:
            continue

        sample_beta_x.append(beta_x)
        sample_eta_x.append(eta_x)

    if len(sample_beta_x) < 10:
        st.warning("Não foi possível estimar os parâmetros de X (= Z − H) com estabilidade suficiente. Ajuste TM/DM/ID ou use respostas menos extremas.")
        st.stop()

    sample_beta_x = np.asarray(sample_beta_x)
    sample_eta_x = np.asarray(sample_eta_x)

    beta_x_25 = np.percentile(sample_beta_x, 25)
    beta_x_75 = np.percentile(sample_beta_x, 75)
    beta_x_central = 0.5 * (beta_x_25 + beta_x_75)
    beta_x_imprecisao = 100.0 * (beta_x_75 - beta_x_central) / max(beta_x_central, EPS)

    eta_x_25 = np.percentile(sample_eta_x, 25)
    eta_x_75 = np.percentile(sample_eta_x, 75)
    eta_x_central = 0.5 * (eta_x_25 + eta_x_75)
    eta_x_imprecisao = 100.0 * (eta_x_75 - eta_x_central) / max(eta_x_central, EPS)

    st.markdown("### Time to Defect – Weibull Parameters")
    st.write(f"**Shape (beta):** {beta_x_central:.2f} (IQR: {beta_x_25:.2f} – {beta_x_75:.2f})")
    st.write(f"**Scale (eta):** {eta_x_central:.2f} {unit} (IQR: {eta_x_25:.2f} – {eta_x_75:.2f})")
    st.write(f"**Beta relative imprecision:** {beta_x_imprecisao:.2f}%")
    st.write(f"**Eta relative imprecision:** {eta_x_imprecisao:.2f}%")
