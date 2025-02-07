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
    st.title("Housing Boston")
    st.write("""
    ### Conoce un poco sobre la base de datos.
    La base de datos **Boston Housing** proviene de información recopilada por el Servicio del Censo de EE.UU. sobre viviendas en la zona de Boston, MA. 
    Contiene información detallada sobre diferentes características de los vecindarios que pueden influir en el valor de las viviendas.""")
    st.image('housingboston.jpg', caption="Boston")
    st.write("""
    #### Descripción de las variables del conjunto de datos:
    - **CRIM**: Tasa de criminalidad per cápita por ciudad.
    - **ZN**: Proporción de terrenos residenciales para lotes de más de 25,000 pies cuadrados.
    - **INDUS**: Proporción de acres comerciales no minoristas por ciudad.
    - **CHAS**: Variable ficticia del río Charles (1 si la zona limita con el río; 0 en caso contrario).
    - **NOX**: Concentración de óxidos de nitrógeno (partes por 10 millones).
    - **RM**: Número promedio de habitaciones por vivienda.
    - **AGE**: Proporción de unidades ocupadas por propietarios construidas antes de 1940.
    - **DIS**: Distancia ponderada a cinco centros de empleo de Boston.
    - **RAD**: Índice de accesibilidad a carreteras radiales.
    - **TAX**: Tasa del impuesto a la propiedad por cada $10,000.
    - **PTRATIO**: Ratio de alumnos por profesor por ciudad.
    - **B**: 1000(Bk - 0.63)^2 donde Bk es la proporción de residentes afroamericanos por ciudad.
    - **LSTAT**: Porcentaje de población con bajo nivel socioeconómico.
    - **MEDV**: Valor medio de las viviendas ocupadas por sus propietarios en miles de dólares.
    
    Este conjunto de datos es ampliamente utilizado en estudios sobre la valorización inmobiliaria y el desarrollo de modelos de regresión.
    """)
    st.title("Predicción del precio de casas en Boston")
    st.subheader("Modelo utilizado para la predicción")
 
    
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
    ptratio = st.number_input("Ratio de alumnos por profesor", min_value=0.0, format="%.01f")
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
