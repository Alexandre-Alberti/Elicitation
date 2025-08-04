# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 01:53:56 2025

@author: alexa
"""

import streamlit as st
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt

# Título
st.title("Elicitation for delay-time model")

# Medida de tempo
unit = st.text_input("Time unit (e.g., hours, days, weeks):", key="unit")

# Tempo médio até a falha
TM = st.number_input(f"Average time to system failure ({unit})", min_value=0.0, format="%.2f", key="TM")

# Delay-time médio
DM = st.number_input(
    f"Average delay-time (interval in which a defect can be detected by inspection before failure) ({unit})",
    min_value=0.0, format="%.2f", key="DM"
)

# Definição das faixas qualitativas
qualitative_options = {
    "Very High (almost certain) – 90 to 100%": (90, 99.99),
    "High – 75 to 90%": (75, 90),
    "Medium – High – 60 to 75%": (60, 75),
    "Medium – 40 to 60%": (40, 60),
    "Medium – Low – 25 to 40%": (25, 40),
    "Low – 10 to 25%": (10, 25),
    "Very Low (unlikely) – 0 to 10%": (0.01, 10),
}

matrix_t = np.zeros(5)
matrix_p = np.zeros([5, 2])

if TM > 0:
    time_points = [int(x) for x in [0.1 * TM, 0.5 * TM, TM, 1.5 * TM, 2 * TM]]

    st.markdown("### Estimated survival probabilities:")

    for i in range(5):
        t = time_points[i]
        question = f"What is the chance the system remains functional after {t} {unit} of operation?"
        response = st.selectbox(question, list(qualitative_options.keys()), key=f"resp_{i}")
        
        matrix_t[i] = t
        matrix_p[i][0] = qualitative_options[response][0]
        matrix_p[i][1] = qualitative_options[response][1]

# Botão para calcular parâmetros
if st.button("Estimate Weibull Parameters") and TM > 0:
    sample_eta = np.zeros(1000)
    sample_beta = np.zeros(1000)

    for i in range(1000):
        f_prob = np.zeros(5)
        s_prob = np.zeros(5)
        s_prob[0] = rd.uniform(matrix_p[0][0], matrix_p[0][1])
        f_prob[0] = (100 - s_prob[0]) / 100

        for j in range(1, 5):
            lim_sup = min(s_prob[j-1], matrix_p[j][1])
            s_prob[j] = rd.uniform(matrix_p[j][0], lim_sup)
            f_prob[j] = (100 - s_prob[j]) / 100

        x_input = np.log(matrix_t)
        y_input = np.log(1 / (1 - f_prob))

        A, B = np.polyfit(x_input, y_input, 1)
        beta = A
        eta = np.exp(-B / beta)

        sample_beta[i] = beta
        sample_eta[i] = eta

    # Estatísticas de beta
    beta_25 = np.percentile(sample_beta, 25)
    beta_75 = np.percentile(sample_beta, 75)
    beta_central = (beta_25 + beta_75) / 2
    beta_imprecisao = 100 * (beta_75 - beta_central) / beta_central

    # Estatísticas de eta
    eta_25 = np.percentile(sample_eta, 25)
    eta_75 = np.percentile(sample_eta, 75)
    eta_central = (eta_25 + eta_75) / 2
    eta_imprecisao = 100 * (eta_75 - eta_central) / eta_central

    # Exibir resultados
    st.markdown("### Weibull Parameter Estimates (from Monte Carlo Sampling)")

    st.markdown("**Beta (shape parameter):**")
    st.write(f"25th percentile: {beta_25:.4f}")
    st.write(f"75th percentile: {beta_75:.4f}")
    st.write(f"Central value (Q2): {beta_central:.4f}")
    st.write(f"Relative imprecision: {beta_imprecisao:.2f}%")

    st.markdown("**Eta (scale parameter):**")
    st.write(f"25th percentile: {eta_25:.4f} {unit}")
    st.write(f"75th percentile: {eta_75:.4f} {unit}")
    st.write(f"Central value (Q2): {eta_central:.4f} {unit}")
    st.write(f"Relative imprecision: {eta_imprecisao:.2f}%")

 
