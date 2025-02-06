import streamlit as st
import numpy as np
import gzip
import pickle

# Cargar el modelo entrenado
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, "rb") as f:
        model = pickle.load(f)
    return model

# Interfaz en Streamlit
def main():
    st.title("Predicción del precio de casas en Boston")
    st.write("El modelo final utiliza el método de clasificación basado en el voto de k-vecinos más cercanos con los hiperparámetros **n_neighbors: 4, p: 3**.")
    st.write("---")
    st.subheader("Ingrese las características de la casa para obtener la predicción")
    
    # Entrada de datos por el usuario
    criminalidad = st.number_input("Tasa de criminalidad per cápita", min_value=0.0, format="%.5f")
    residencias = st.number_input("Proporción de terrenos residenciales", min_value=0.0, format="%.2f")
    acres = st.number_input("Proporción de acres comerciales", min_value=0.0, format="%.2f")
    riocharles = st.selectbox("Cercanía al río Charles", [0, 1])
    nox = st.number_input("Concentración de óxidos de nitrógeno (NOX)", min_value=0.0, format="%.3f")
    habitaciones = st.number_input("Número medio de habitaciones por vivienda", min_value=1.0, format="%.2f")
    edad = st.number_input("Proporción de casas construidas antes de 1940", min_value=0.0, format="%.1f")
    empleo = st.number_input("Distancia a centros de empleo", min_value=0.0, format="%.2f")
    calles = st.number_input("Índice de accesibilidad a carreteras", min_value=1, max_value=24)
    impuestos = st.number_input("Tasa de impuestos a la propiedad", min_value=0, format="%d")
    ptratio = st.number_input("Ratio de alumnos por profesor", min_value=0.0, format="%.1f")
    afroa = st.number_input("Proporción de residentes afroamericanos", min_value=0.0, format="%.2f")
    estrato = st.number_input("Porcentaje de población con bajo nivel socioeconómico", min_value=0.0, format="%.2f")
    
    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        model = load_model()
        input_data = np.array([[criminalidad, residencias, acres, riocharles, nox, habitaciones, edad, empleo, calles, impuestos, ptratio, afroa, estrato]])
        prediction = model.predict(input_data)[0]
        st.write(f"El valor estimado de la casa es: **${prediction * 1000:.2f} USD**")

if __name__ == '__main__':
    main()
