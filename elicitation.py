# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 01:53:56 2025

@author: alexa
"""

import streamlit as st
import numpy as np
import random as rd
from scipy.stats import weibull_min
from scipy.integrate import dblquad

# Funções de densidade
def fz(z, forma, escala):
    return (forma / escala) * ((z / escala) ** (forma - 1)) * np.exp(-((z / escala) ** forma))

def fh(h, d_medio):
    return (1 / d_medio) * np.exp(- (1 / d_medio) * h)

# Interface Streamlit
st.title("Elicitation for delay-time model")

unit = st.text_input("Time unit (e.g., hours, days, weeks):", key="unit")
TM = st.number_input(f"Average time to system failure ({unit})", min_value=0.0, format="%.2f", key="TM")
DM = st.number_input(f"Average delay-time (defect detection window) ({unit}):", min_value=0.0, format="%.2f", key="DM")
ID = st.number_input("Imprecision in delay-time estimation (%)", min_value=0.0, format="%.2f", key="ID")

qualitative_options = {
    "Very High (almost certain) – 90 to 100%": (90, 99.99),
    "High – 75 to 90%": (75, 90),
    "Medium – High – 60 to 75%": (60, 75),
    "Medium – 40 to 60%": (40, 60),
    "Medium – Low – 25 to 40%": (25, 40),
    "Low – 10 to 25%": (10, 25),
    "Very Low (unlikely) – 0 to 10%": (0.01, 10),
}

# Coleta das probabilidades
if TM > 0:
    time_points = [int(x) for x in [0.1 * TM, 0.5 * TM, TM, 1.5 * TM, 2 * TM]]
    matrix_t = np.zeros(5)
    matrix_p = np.zeros([5, 2])

    st.markdown("### Estimated survival probabilities:")
    for i in range(5):
        t = time_points[i]
        response = st.selectbox(f"Chance system remains functional after {t} {unit}:", list(qualitative_options.keys()), key=f"resp_{i}")
        matrix_t[i] = t
        matrix_p[i][0] = qualitative_options[response][0]
        matrix_p[i][1] = qualitative_options[response][1]

# Cálculo com botão
if st.button("Estimate parameters"):
    # Monte Carlo para Weibull de tempo até falha (Z)
    sample_eta = np.zeros(1000)
    sample_beta = np.zeros(1000)

    for i in range(1000):
        f_prob = np.zeros(5)
        s_prob = np.zeros(5)
        s_prob[0] = rd.uniform(matrix_p[0][0], matrix_p[0][1])
        f_prob[0] = (100 - s_prob[0]) / 100
        for j in range(1, 5):
            lim_sup = min(s_prob[j - 1], matrix_p[j][1])
            s_prob[j] = rd.uniform(matrix_p[j][0], lim_sup)
            f_prob[j] = (100 - s_prob[j]) / 100

        x_input = np.log(matrix_t)
        y_input = np.log(-np.log(1 - f_prob))
        A, B = np.polyfit(x_input, y_input, 1)
        beta = A
        eta = np.exp(-B / beta)
        sample_beta[i] = beta
        sample_eta[i] = eta

    beta_25 = np.percentile(sample_beta, 25)
    beta_75 = np.percentile(sample_beta, 75)
    beta_central = (beta_25 + beta_75) / 2
    beta_imprecisao = 100 * (beta_75 - beta_central) / beta_central
    eta_25 = np.percentile(sample_eta, 25)
    eta_75 = np.percentile(sample_eta, 75)
    eta_central = (eta_25 + eta_75) / 2
    eta_imprecisao = 100 * (eta_75 - eta_central) / eta_central

    st.markdown("### Time to Failure – Weibull Parameters")
    st.write(f"**Shape (beta):** {beta_central:.2f} (IQR: {beta_25:.2f} – {beta_75:.2f})")
    st.write(f"**Scale (eta):** {eta_central:.2f} {unit} (IQR: {eta_25:.2f} – {eta_75:.2f})")
    st.write(f"**Beta relative imprecision:** {beta_imprecisao:.2f}%")
    st.write(f"**Eta relative imprecision:** {eta_imprecisao:.2f}%")

    # Estimativa para X = Z - H via convolução invertida
    sample_eta_x = np.zeros(100)
    sample_beta_x = np.zeros(100)

    for k in range(100):
        escala = rd.uniform(eta_25, eta_75)
        forma = rd.uniform(beta_25, beta_75)
        d_medio = rd.uniform((1 - ID / 100) * DM, (1 + ID / 100) * DM)

        x_input = np.zeros(30)
        y_input = np.zeros(30)
        ini = 0
        for i in range(30):
            ini += 0.1
            t = ini * escala
            x_input[i] = np.log(t)
            prob = 1 - dblquad(lambda h, z: fz(z, forma, escala) * fh(h, d_medio),
                               t, 10 * escala,
                               lambda z: 0, lambda z: z - t)[0]
            y_input[i] = np.log(np.log(1 / (1 - prob)))

        A, B = np.polyfit(x_input, y_input, 1)
        beta_x = A
        eta_x = np.exp(-B / beta_x)
        sample_eta_x[k] = eta_x
        sample_beta_x[k] = beta_x

    beta_x_25 = np.percentile(sample_beta_x, 25)
    beta_x_75 = np.percentile(sample_beta_x, 75)
    beta_x_central = (beta_x_25 + beta_x_75) / 2
    beta_x_imprecisao = 100 * (beta_x_75 - beta_x_central) / beta_x_central
    eta_x_25 = np.percentile(sample_eta_x, 25)
    eta_x_75 = np.percentile(sample_eta_x, 75)
    eta_x_central = (eta_x_25 + eta_x_75) / 2
    eta_x_imprecisao = 100 * (eta_x_75 - eta_x_central) / eta_x_central

    st.markdown("### Time to Defect – Weibull Parameters")
    st.write(f"**Shape (beta):** {beta_x_central:.2f} (IQR: {beta_x_25:.2f} – {beta_x_75:.2f})")
    st.write(f"**Scale (eta):** {eta_x_central:.2f} {unit} (IQR: {eta_x_25:.2f} – {eta_x_75:.2f})")
    st.write(f"**Beta relative imprecision:** {beta_x_imprecisao:.2f}%")
    st.write(f"**Eta relative imprecision:** {eta_x_imprecisao:.2f}%")


 






