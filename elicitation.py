# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 01:53:56 2025

@author: alexa
"""

import streamlit as st
import numpy as np

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
    matrix_t = np.zeros([5, 1])
    matrix_p = np.zeros([5, 2])

    st.markdown("### Estimated survival probabilities:")

    for i in range(5):
        t = time_points[i]
        question = f"What is the chance the system remains functional after {t} {unit} of operation?"
        response = st.selectbox(question, list(qualitative_options.keys()), key=f"resp_{i}")
        
        matrix_t[i][0] = t
        matrix_p[i][0] = qualitative_options[response][0]
        matrix_p[i][1] = qualitative_options[response][1]

    # Mostra os resultados
    st.markdown("### Summary of inputs:")
    st.write(f"Time unit: {unit}")
    st.write(f"Average time to failure (TM): {TM} {unit}")
    st.write(f"Average delay-time (DM): {DM} {unit}")
    
    st.markdown("#### Time points:")
    st.dataframe(matrix_t, use_container_width=True)

    st.markdown("#### Probability intervals (min %, max %):")
    st.dataframe(matrix_p, use_container_width=True)


