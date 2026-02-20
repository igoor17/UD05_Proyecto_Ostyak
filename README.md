# UD05 Proyecto Final - Ostyak

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completado-success?style=for-the-badge)

## Descripción
Este proyecto implementa y compara diferentes arquitecturas de Redes Neuronales Convolucionales (CNN) para la clasificación de tumores cerebrales a partir de imágenes de Resonancia Magnética (MRI). Incluye además el despliegue del mejor modelo en una aplicación web interactiva.

## Dataset utilizado
**Brain Tumor MRI Dataset**
- 4 clases disponibles: `glioma`, `meningioma`, `notumor`, `pituitary`.
- Imágenes en escala de grises redimensionadas a 128x128 píxeles.

## Fases completadas
- [x] Fase 1 - Fundamentos (Modelo base `CNN_3B`)
- [x] Fase 2 - Experimentación (Comparativa de arquitecturas `CNN_4B` y `CNN_3B_BN_GAP`)
- [ ] Fase 3 - Excelencia
- [x] Bonus: Opción B (Mini-aplicación de inferencia con Streamlit)

## Instrucciones de ejecución

### 1. Entorno y Requisitos
Proyecto realizado con Python 3.12. Para instalar todas las dependencias necesarias en un entorno virtual local, ejecuta:

    pip install -r requirements.txt

### 2. Ejecución de Notebooks
El proceso de entrenamiento completo se encuentra unificado y extendido en el notebook de la Fase 2.

Para reproducir el entrenamiento y la evaluación de los modelos:
1. Abrir y ejecutar `UD05_Proyecto_Ostyak_Fase2.ipynb` (contiene todo el preprocesamiento inicial de la Fase 1 más la experimentación de la Fase 2).
2. El archivo `UD05_Proyecto_Ostyak_Fase1.ipynb` se incluye para mantener el historial de la entrega, pero no es necesario ejecutarlo previamente para que funcione la Fase 2.

### 3. Aplicación Web (Bonus)
Para probar la aplicación interactiva de clasificación, abre una terminal en la raíz del proyecto y ejecuta:

    streamlit run bonus/app.py

Esto abrirá una interfaz en tu navegador (por defecto en `http://localhost:8501`) donde podrás subir imágenes y obtener diagnósticos asistidos por IA.

### 4. Script de inferencia individual
Para clasificar una imagen directamente por consola sin levantar el entorno gráfico:

    python inferencia.py --imagen ruta/a/tu/imagen.jpg

    ## Resultados principales
- **Dataset:** Brain Tumor MRI Dataset
- **Accuracy objetivo:** >=78%
- **Accuracy inicial (Fase 1, `CNN_3B`):** 90.87%
- **Mejor accuracy (Fase 2, `CNN_4B`):** 93.19%
