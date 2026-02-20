import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Detector de Tumores Cerebrales", page_icon="🧠", layout="centered")

st.title("🧠 Clasificador de Resonancias Magnéticas")
st.write("Sube una imagen de resonancia magnética para obtener un diagnóstico asistido por IA.")

@st.cache_resource
def cargar_modelo():
    return keras.models.load_model('modelos/mejor_modelo_fase2.keras')

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error("No se ha encontrado el archivo del modelo.")
    st.stop()

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Subida de archivo
archivo = st.file_uploader("Seleccionar imagen (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if archivo is not None:
    # Mostrar la imagen
    imagen_original = Image.open(archivo)
    st.image(imagen_original, caption="Imagen subida", width=350)
    
    if st.button('Analizar Imagen', type="primary"):
        with st.spinner('Analizando...'):
            # Preprocesamiento

            # Pasamos a escala de grises
            imagen_gris = imagen_original.convert('L')
            # Redimensionamos a 128x128
            img_resized = imagen_gris.resize((128, 128))
            # Pasamos a array y normalizamos (0 a 1)
            img_array = np.array(img_resized) / 255.0
            # Añadimos la dimensión del canal (128, 128, 1)
            img_array = np.expand_dims(img_array, axis=-1)
            # Añadimos la dimensión del batch (1, 128, 128, 1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            probabilidades = modelo.predict(img_array)[0]
            idx_predicho = np.argmax(probabilidades)
            clase_predicha = CLASS_NAMES[idx_predicho]
            confianza = probabilidades[idx_predicho] * 100
            
            st.divider()
            st.subheader("Resultado del Análisis")
            
            # Mensaje de éxito
            if confianza > 80:
                st.success(f"**Diagnóstico sugerido:** {clase_predicha.upper()} ({confianza:.1f}%)")
            else:
                st.warning(f"**Diagnóstico sugerido:** {clase_predicha.upper()} ({confianza:.1f}%) - *Baja confianza*")
            
            # Gráfico de barras con las probabilidades
            st.write("**Desglose de probabilidades:**")
            df_probs = pd.DataFrame({
                'Clase': [c.capitalize() for c in CLASS_NAMES],
                'Probabilidad (%)': probabilidades * 100
            })
            st.bar_chart(df_probs.set_index('Clase'))

st.markdown("---")
st.caption(
    "⚠️ **Aviso Clínico:** "
    "Esta herramienta es un prototipo académico. "
    "El diagnóstico final debe ser realizado siempre por un médico especialista."
)