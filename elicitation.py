# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 01:53:56 2025

@author: alexa
"""

import streamlit as st

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

# Geração dos tempos para as perguntas
if TM > 0:
    time_points = [int(x) for x in [0.1 * TM, 0.5 * TM, TM, 1.5 * TM, 2 * TM]]
    survival_estimates = {}

    st.markdown("### Estimated survival probabilities:")

    for idx, t in enumerate(time_points):
        question = f"What is the chance the system remains functional after {t} {unit} of operation?"
        response = st.selectbox(question, list(qualitative_options.keys()), key=f"q{idx+1}")
        survival_estimates[t] = {
            "response": response,
            "min_percent": qualitative_options[response][0],
            "max_percent": qualitative_options[response][1]
        }

    # Exibir resultados finais (para debug ou confirmação)
    st.markdown("### Summary of Inputs")
    st.write("Time unit:", unit)
    st.write("Average time to failure (TM):", TM, unit)
    st.write("Average delay-time (DM):", DM, unit)
    st.write("Survival estimates:")
    st.json(survival_estimates)
