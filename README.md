# Dashboard de Exportaciones Bolivianas

## Origen de los datos

Los datos provienen del Instituto Nacional de Estadística de Bolivia (INE):  
🔗 https://www.ine.gob.bo/index.php/estadisticas-economicas/comercio-exterior/cuadros-estadisticos-exportaciones/

## Estructura del proyecto (DataLake)

```bash
📁 LANDING_ZONE/          # Datos originales sin procesar
📁 CLEAN_ZONE/            # Datos limpios y transformados
📁 CONSUMPTION_ZONE/      # Datos finales utilizados por el dashboard
```

## Bibliotecas utilizadas
- streamlit – para el dashboard interactivo
- pandas, numpy – manipulación de datos
- tensorflow – entrenamiento y carga de modelos .h5
- scikit-learn – preprocesamiento y evaluación
- plotly, matplotlib – visualización de datos y métricas
- openpyxl – lectura de archivos Excel

## Cómo ejecutar el dashboard
Utilizar Python 3.10
```bash
streamlit run proyeccion_streamlit.py
```
