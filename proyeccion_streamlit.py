import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ----------- Configuración inicial ----------
st.set_page_config(page_title="Exportaciones Bolivia", layout="wide")
st.markdown("<h1 style='text-align: center;'>Dashboard de Exportaciones - Grupo 3</h1>", unsafe_allow_html=True)


# --- Cargar modelos ---
modelo_clas_peso = load_model("datalake/3_CONSUMPTION_ZONE/modelo_clasificacion_peso.h5")
modelo_clas_valor = load_model("datalake/3_CONSUMPTION_ZONE/modelo_clasificacion_valor.h5")

# --- Cargar datasets ---
df_clas_peso = pd.read_csv("datalake/2_CLEAN_ZONE/expactprodmes_92_25_peso.csv")
df_clas_valor = pd.read_csv("datalake/2_CLEAN_ZONE/expactprodmes_92_25_valor.csv")
df_reg_peso = pd.read_csv("datalake/2_CLEAN_ZONE/exppaisesmes_92_25_peso.csv")
df_reg_valor = pd.read_csv("datalake/2_CLEAN_ZONE/exppaisesmes_92_25_valor.csv")
df_reg_peso.fillna(0, inplace=True)
df_reg_valor.fillna(0, inplace=True)

# --- Pestañas principales ---
tab1, tab2 = st.tabs(["📊 Clasificación por Producto", "📈 Proyección por País"])

with tab1:
    st.header("Tendencia de Productos por Actividad Económica")

    # Cargar datos
    df_peso = pd.read_csv("datalake/2_CLEAN_ZONE/expactprodmes_92_25_peso.csv")
    df_valor = pd.read_csv("datalake/2_CLEAN_ZONE/expactprodmes_92_25_valor.csv")

    # Filtro por Actividad Económica
    actividades = sorted(df_peso['Actividad Económica'].unique())
    actividad_seleccionada = st.selectbox("Selecciona una Actividad Económica", actividades)

    df_peso_act = df_peso[df_peso['Actividad Económica'] == actividad_seleccionada].copy()
    df_valor_act = df_valor[df_valor['Actividad Económica'] == actividad_seleccionada].copy()

    # Cargar modelos
    modelo_peso = load_model("datalake/3_CONSUMPTION_ZONE/modelo_clasificacion_peso.h5")
    modelo_valor = load_model("datalake/3_CONSUMPTION_ZONE/modelo_clasificacion_valor.h5")

    # Columnas de tiempo
    columnas_tiempo = df_peso.columns[2:]

    def predecir_tendencias(df, modelo):
        tendencias = []
        for _, fila in df.iterrows():
            serie = fila[columnas_tiempo].fillna(0).values.astype(np.float32)
            scaler = MinMaxScaler()
            serie_norm = scaler.fit_transform(serie.reshape(-1, 1)).flatten()
            tendencia = modelo.predict(np.array([serie_norm]), verbose=0)
            clase = np.argmax(tendencia)
            if clase == 0:
                label = "Descenso"
            elif clase == 1:
                label = "Estable"
            else:
                label = "Crecimiento"
            tendencias.append(label)
        return tendencias

    # Obtener predicciones
    tendencias_peso = predecir_tendencias(df_peso_act, modelo_peso)
    tendencias_valor = predecir_tendencias(df_valor_act, modelo_valor)

    # Construir tabla
    tabla_resultados = pd.DataFrame({
        "Producto": df_peso_act["Producto"].values,
        "Tendencia Peso": tendencias_peso,
        "Tendencia Valor": tendencias_valor
    })

    # Estilo condicional
    def color_tendencia(val):
        if val == "Crecimiento":
            return "background-color: #d4edda; color: green;"
        elif val == "Estable":
            return "background-color: #dbeafe; color: #1d4ed8;"
        elif val == "Descenso":
            return "background-color: #f8d7da; color: red;"
        return ""

    styled_df = tabla_resultados.style.applymap(color_tendencia, subset=["Tendencia Peso", "Tendencia Valor"])

    st.dataframe(styled_df, use_container_width=True)


# ======================================================
# ================ TAB 2 - REGRESIÓN ===================
# ======================================================
with tab2:
    st.header("Proyección de Exportaciones por País (Peso y Valor)")

    col1, col2 = st.columns(2)
    with col1:
        pais_seleccionado = st.selectbox("Selecciona un país", df_reg_peso['País'].unique())
    with col2:
        anios = st.number_input("¿Cuántos años deseas proyectar?", min_value=1, max_value=20, value=10)

    meses_a_predecir = anios * 12
    ventana = 12

    def predecir_serie(df, modelo_path, pais):
        serie = df[df["País"] == pais].drop(columns=["País"]).values.flatten()
        scaler = MinMaxScaler()
        serie_escalada = scaler.fit_transform(serie.reshape(-1, 1)).flatten()
        entrada = serie_escalada[-ventana:].tolist()
        modelo = load_model(modelo_path)
        predicciones = []
        for _ in range(meses_a_predecir):
            entrada_array = np.array(entrada[-ventana:]).reshape(1, -1)
            pred = modelo.predict(entrada_array, verbose=0)[0][0]
            predicciones.append(pred)
            entrada.append(pred)
        predicciones_reales = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1)).flatten()
        fechas_existentes = pd.date_range(start="1992-01-01", periods=len(serie), freq="MS")
        fechas_futuras = pd.date_range(start=fechas_existentes[-1] + pd.offsets.MonthBegin(1), periods=meses_a_predecir, freq="MS")
        return fechas_existentes, serie, fechas_futuras, predicciones_reales

    # Peso
    fechas_peso, reales_peso, fechas_pred_peso, pred_peso = predecir_serie(
        df_reg_peso, "datalake/3_CONSUMPTION_ZONE/modelo_regresion_peso.h5", pais_seleccionado)

    fig_peso = go.Figure()
    fig_peso.add_trace(go.Scatter(x=fechas_peso, y=reales_peso, mode='lines+markers', name='Reales (Peso)', line=dict(color='blue')))
    fig_peso.add_trace(go.Scatter(x=fechas_pred_peso, y=pred_peso, mode='lines+markers', name='Proyección (Peso)', line=dict(color='green')))
    fig_peso.update_layout(
        title=f"Proyección de Exportaciones por Peso - {pais_seleccionado}",
        xaxis_title="Fecha", yaxis_title="Peso (kg)",
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=5, label="5A", step="year", stepmode="backward"),
            dict(count=10, label="10A", step="year", stepmode="backward"),
            dict(step="all")
        ]), rangeslider=dict(visible=True), type="date"),
        template="plotly_white"
    )

    # Valor
    fechas_valor, reales_valor, fechas_pred_valor, pred_valor = predecir_serie(
        df_reg_valor, "datalake/3_CONSUMPTION_ZONE/modelo_regresion_valor.h5", pais_seleccionado)

    fig_valor = go.Figure()
    fig_valor.add_trace(go.Scatter(x=fechas_valor, y=reales_valor, mode='lines+markers', name='Reales (Valor)', line=dict(color='orange')))
    fig_valor.add_trace(go.Scatter(x=fechas_pred_valor, y=pred_valor, mode='lines+markers', name='Proyección (Valor)', line=dict(color='red')))
    fig_valor.update_layout(
        title=f"Proyección de Exportaciones por Valor - {pais_seleccionado}",
        xaxis_title="Fecha", yaxis_title="Valor (millones USD)",
        xaxis=dict(rangeselector=dict(buttons=[
            dict(count=5, label="5A", step="year", stepmode="backward"),
            dict(count=10, label="10A", step="year", stepmode="backward"),
            dict(step="all")
        ]), rangeslider=dict(visible=True), type="date"),
        template="plotly_white"
    )

    st.subheader("Exportaciones por Peso")
    st.plotly_chart(fig_peso, use_container_width=True)

    st.subheader("Exportaciones por Valor")
    st.plotly_chart(fig_valor, use_container_width=True)
