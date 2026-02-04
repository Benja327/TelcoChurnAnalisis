import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Creación de clase
class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def classify_variables(self):
        #Para clasisficar el tipo de variable
        col_numerico = self.df.select_dtypes(include=[np.number]).columns.tolist()
        col_categorico = self.df.select_dtypes(include=['object']).columns.tolist()
        return col_numerico, col_categorico

    def get_stats(self):
        #Estadística descriptiva
        return self.df.describe()


# Estableciendo nombre de la pestaña
st.set_page_config(page_title="Isaac Hernández - Telco Churn Análisis", layout="wide")

#Menú tipo radio paralos modulos 1y 2
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Ir a:", ["Módulo 1: Inicio", "Módulo 2: Carga de Datos"])

# --- Modulo 1, inicio o home
if menu == "Módulo 1: Inicio":
    st.title("Proyecto: Análisis de Fuga de Clientes (Telco Churn)")
    st.write("""
    ### Descripción del Objetivo
    Este proyecto analiza los patrones asociados a la fuga de clientes en una empresa de telecomunicaciones, 
    buscando identificar factores clave que influyen en la retención.
    """)
    
    st.info(f"**Autor:** Isaac Benjamín Hernánde Suárez\n\n**Especialización:** Python for Analytics\n\n**Tecnologías utilizadas:** Python, Pandas, Streamlit, Matplotlib, Seaborn")

# --- Modulo 2, carga de datos y análisis
elif menu == "Módulo 2: Carga de Datos":
    
    st.sidebar.title("Parámetros")

    archivo = st.sidebar.file_uploader("Sube tu archivo", type=["csv","xlsx"])

    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel("archivo")

        #Limíeza de valores con valor espacio en blanco
        df = df.replace(r'^\s*$', np.nan, regex=True)

        st.title("Carga del Dataset")
        st.subheader("Estado de carga")
        st.success("Archivo cargado correctamente")
        ##st.write(df)

        st.subheader("Vista previa del Dataset")
        st.dataframe(df.head())

        st.warning("**Seleccione el item a visualizar en la parte izquierda**")

        #Creación de menú seleccionable para los items
        item_seleccionado = st.sidebar.selectbox("Seleccionar Item:", 
                                                 ["Ítem 1: Información general del dataset", "Ítem 2: Clasificación de variables",
                                                  "Ítem 3: Estadísticas descriptivas","Ítem 4: Análisis de valores faltantes",
                                                  "Ítem 5: Distribución de variables numéricas","Ítem 6: Análisis de variables categóricas",
                                                  "Ítem 7: Análisis bivariado (numérico vs categórico)","Ítem 8: Análisis bivariado (categórico vs categórico)",
                                                  "Ítem 9: Análisis basado en parámetros seleccionados","Ítem 10: Hallazgos clave","Conclusiones Finales"], index=None)
        
        if item_seleccionado == "Ítem 1: Información general del dataset":
            st.title("Ítem 1: Información general del dataset")
            st.write("""En esta sección, se mostrará la cantidad de filas, columnas y valores nulos que presenta el dataset""")

            #Hallando las dimensiones del dataset
            st.subheader("Dimensiones del Dataset")
            st.info(f"El dataset contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.")

            #Hallando el tipo de datos
            st.markdown("---")
            st.subheader("Tipos de Datos")
            st.write(df.dtypes)

            #Hallando la frecuencia de valores nulos
            st.markdown("---")
            st.subheader("Conteo de Valores nulo por columna")
            st.write(df.isnull().sum())
            

        elif item_seleccionado == "Ítem 2: Clasificación de variables":

            analyzer = DataAnalyzer(df)
            st.title("Ítem 2: Clasificación de variables")
            st.write("""En esta sección, se mostrará el tipo y cantidad de variables""")
            numerico_col, categorico_col = analyzer.classify_variables()
            
            st.markdown("---")
            st.subheader("**Tipo de Datos**")
            st.info(f"**Variables Numéricas:** {len(numerico_col)}\n\n**Variables Categóricas:** {len(categorico_col)}")


        elif item_seleccionado == "Ítem 3: Estadísticas descriptivas":

            st.title("Ítem 3: Estadísticas descriptivas")
            st.write("""En esta sección, se mostrará características estadísticas presentes en el dataset (cantidad, media , mediana, desviación ,etc)""")
            st.subheader("Descripción del Dataset")
            st.write(df.describe())

            media_permencia = round(df['tenure'].mean(),2)
            media_recarga_mes = round(df['MonthlyCharges'].mean(),2)

            st.markdown("---")
            st.subheader("Medias del Dataset")
            st.info(f"**Mes de permanencia promedio:** {media_permencia}\n\n**Recarga mensual promedio:** {media_recarga_mes}")



        elif item_seleccionado == "Ítem 4: Análisis de valores faltantes":
            
            st.header("Ítem 4: Análisis de valores faltantes")
            st.write("""En esta sección, se mostrarán la cantidad de valores nulos en el dataset""")
            #Conteo de Nulos
            nulos_conteo = df.isnull().sum()

            #Consideramos los que tengan conteo de nulos mayor a 0
            faltante_nulos = nulos_conteo[nulos_conteo > 0]

            #Visualización por columnas
            if not faltante_nulos.empty:
                col1, col2 = st.columns(2)
                    
                with col1:
                    st.write("Conteo de Campos Nulos")
                    st.dataframe(faltante_nulos)

                with col2:
                    st.write("Visualización")
                    fig, ax = plt.subplots()
                    sns.barplot(x=faltante_nulos.index, y=faltante_nulos.values, ax=ax)
                    
                    #Etiqueta de datos
                    ax.bar_label(ax.containers[0], padding=3)
                    
                    #Ajustando Eje
                    ax.set_ylim(0, faltante_nulos.max() * 1.2)

                    ax.set_ylabel("Cantidad de datos Nulos")
                    st.pyplot(fig)

                st.markdown("---")
                st.write("### Hallazgo")
                st.info(f"""
                Se encontraron **{faltante_nulos.sum()}** valores faltantes en la columna **TotalCharges**. 
                Al buscar la permanencia de estos (Tenure), se encontraron que tienen permanencia de 0 meses.  
                Significa que no puedieron realizar el primer pago, por eso figura el valor de pago nulo.
                """)
            else:
                st.success("No se encontraron valores faltantes en el dataset después de la limpieza.")


        elif item_seleccionado == "Ítem 5: Distribución de variables numéricas":
            st.header("Ítem 5: Distribución de variables numéricas")
            st.write("""En esta sección, se mostrarán gráficos relacionados a las frecuencias y la media de datos""")

            # Selección de variable
            variable_analizar_grafico = st.selectbox("Seleccione la variable a analizar:", ["tenure", "MonthlyCharges", "TotalCharges"])

            if variable_analizar_grafico:
                #Realizamos una limpieza ya que pueden figurar valores nulos que afecten al cálculo
                datos_limpios = pd.to_numeric(df[variable_analizar_grafico], errors='coerce').dropna()
                
                if not datos_limpios.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    #Realizamos el gráfico de barras para la frecuencia de datos
                    sns.histplot(datos_limpios, kde=True, ax=ax, color="#3DA8E6", bins=30)
                    
                    #Realizamos una línea vertical para poder ubicar la media
                    media_val = datos_limpios.mean()
                    ax.axvline(media_val, color="#F7C334", linestyle='--', label=f'Media: {media_val:.2f}')
                    ax.legend()

                    #Asignamos los valores a los ejes del gráfico
                    ax.set_title(f"Distribución de {variable_analizar_grafico}", fontsize=14)
                    ax.set_xlabel(variable_analizar_grafico)
                    ax.set_ylabel("Frecuencia de Clientes")
                    
                    st.pyplot(fig)

                    #Interpretación de los datos según la variable seleccionada
                    st.markdown("---")
                    st.markdown("### Interpretación Visual")
                    if variable_analizar_grafico == "tenure":
                        st.info("""
                        * **Observación:** Se observa mayor frecuencia en los extremos del gráfico. 
                        * **Análisis:** La mayor parte de clientes se concentran en clientes nuevos (0 meses) y clientes antiguos (entre los 70 meses).
                        """)
                    elif variable_analizar_grafico == "MonthlyCharges":
                        st.info("""
                        * **Observación:** Hay un pico respecto a la cantidad de clientes cerca del plan de 20. 
                        * **Análisis:** La mayor parte de clientes, prefieren tener un servicio básico con un monto menor.
                        """)
                    elif variable_analizar_grafico == "TotalCharges":
                        st.info("""
                        * **Observación:** Se observa un pico al extremo izquierdo.
                        * **Análisis:** La mayoría de los clientes tienen cargos totales acumulados bajos, ya que una gran parte de ellos son clientes nuevos (antiguedad 0).
                        """)
                else:
                     st.error("Error, selecconar otra variable para poder analizar.")
        elif item_seleccionado == "Ítem 6: Análisis de variables categóricas":
            st.header("Ítem 6: Análisis de variables categóricas")
            st.write("""En esta sección, se mostrará el comparativo de los valores de las variables categóricas""")

            #Variables categóricas para poder realizar graficos comparativos de particiapción
            var_categoricas = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                'Contract', 'PaymentMethod', 'PaperlessBilling']
            
            col_analizar = st.selectbox("Seleccione la variable categórica:", var_categoricas)

            if col_analizar:
                #Definimos tanto el Q (cantidad) y % (Proporción)
                cantidad = df[col_analizar].value_counts()
                proporcion = df[col_analizar].value_counts(normalize=True) * 100

                #Vista en tabla para complementar el gráfico utilizando las variables definidas anteriormente
                df_resumen = pd.DataFrame({
                    'Frecuencia (Cantidad)': cantidad,
                    'Participación (%)': proporcion.map("{:.2f}%".format)
                })

                #Colocaremos la tabla anterior, dentro de otra junto al gráfico para verlo de forma horizontal
                col1, col2 = st.columns([1, 2])

                #La primera columna para el cuadro
                with col1:
                    st.write(f"### Tabla de Frecuencias: {col_analizar}")
                    st.dataframe(df_resumen)

                #La segunda columna para el gráfico
                with col2:
                    st.write(f"### Distribución Visual")
                    fig, ax = plt.subplots()
                    sns.barplot(x=cantidad.index, y=cantidad.values, ax=ax)
                    
                #Agregar etiquetas de datos sobre las barras
                    ax.bar_label(ax.containers[0], padding=3)
                    
                    ax.set_ylabel("Cantidad de Clientes")
                    ax.set_title(f"Conteo por {col_analizar}")
                    st.pyplot(fig)

                #Interpretación de datos
                st.markdown("---")
                st.write("### Lectura de las Proporciones")
                
                cant_max = cantidad.idxmax()
                prop_max = proporcion.max()
                
                st.info(f"""
                * La categoría con mayor participación en el campo **{col_analizar}** es **{cant_max}**, 
                representando un **{prop_max:.2f}%** del total de la data. 
                * Este análisis permite identificar la mayor participación por perfil de cliente para la elaboración de estratégias.
                """)
        elif item_seleccionado == "Ítem 7: Análisis bivariado (numérico vs categórico)":
            st.write("""En esta sección, se mostrará la relación entre las variables numéricas y las variables categóricas para identificar patrones relacionados al Churn""")
            st.header("Ítem 7: Análisis bivariado (Numérico vs Categórico)")

            #Variables numéricas para poder desarrollar distintos gráficos
            var_num = st.selectbox("Seleccione la variable numérica para comparar con el Churn:", 
                                    ["tenure", "MonthlyCharges", "TotalCharges"])

            if var_num:
                #Asignamos que todo sea numérico para evitar errores al realizar el cálculo
                if var_num == "TotalCharges":
                    df[var_num] = pd.to_numeric(df[var_num], errors='coerce')

                fig, ax = plt.subplots(figsize=(10, 6))
                
                #Primer gráfico, gráfico de caja para identificar el comportamiento por mediana
                sns.boxplot(x='Churn', y=var_num, data=df, ax=ax, palette="Set2")
                
                #Agregamos títulos al eje
                ax.set_title(f"Relación entre {var_num} y la baja (Churn)", fontsize=14)
                ax.set_xlabel("¿Se va el cliente? (Churn)")
                ax.set_ylabel(var_num)
                
                st.pyplot(fig)

                #Lectura de datos
                st.markdown("---")
                st.markdown("### Interpretación del Análisis")
                
                if var_num == "tenure":
                    st.info("""
                    * **Observación:** La mediana de permanencia de los clientes que se van (Yes) es mucho menor que la de los que se quedan (No).
                    * **Conclusión:** Los clientes nuevos tienen mayor propensión a Churn, como estatégia se deben realizar campañas de fidelización para que decidan quedarse
                    """)
                elif var_num == "MonthlyCharges":
                    st.info("""
                    * **Observación:** Los clientes con mayor cargo fijo o monto de pago, son los más propensos a generar Churn.
                    * **Conclusión:** El precio elevado es un factor determinante en la fuga de clientes, se pueden realizar campañas Upgrade o Downgrade de planes con descuento.
                    """)
                elif var_num == "TotalCharges":
                    st.info("""
                    * **Observación:** Se observa los clientes con mayor cargo fijo y tiempo en la operadora, son los más fieles y no generan Churn.
                    """)

        elif item_seleccionado == "Ítem 8: Análisis bivariado (categórico vs categórico)":
            st.write("""En esta sección, se mostrará el comportamiento de los datos categóricos en valor porcentual""")
            st.header("Ítem 8: Relación entre Categorías y Churn")

            #Variable categórica para cruzar con Churn
            var_cat = st.selectbox("Seleccione una variable para ver su impacto en el Churn:", 
                                    ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"])

            if var_cat:
                #Creacion de frecuencias porcentaje
                tabla_cruzada = pd.crosstab(df[var_cat], df['Churn'], normalize='index') * 100

                #Tabla apilada
                fig, ax = plt.subplots(figsize=(10, 6))
                tabla_cruzada.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
                
                #Ejes
                ax.set_title(f"Impacto de {var_cat} en el Churn de Clientes", fontsize=14)
                ax.set_ylabel("Porcentaje de Clientes (%)")
                ax.set_xlabel(var_cat)
                plt.xticks(rotation=45)
                ax.legend(title="Churn", labels=["No (Se queda)", "Yes (Se va)"])
                
                #Etiqueta de datos
                for p in ax.patches:
                    width, height = p.get_width(), p.get_height()
                    if height > 0: 
                        ax.annotate(f'{height:.1f}%', 
                                    (p.get_x() + width/2, p.get_y() + height/2), 
                                    ha='center', va='center', color='white', fontweight='bold')

                st.pyplot(fig)

                #Lectura de datos
                st.markdown("---")
                st.markdown("### Hallazgos del Análisis")
                
                if var_cat == "Contract":
                    st.info("""
                    * **Observación:** Los clientes con contrato **Month-to-month** (Mes a mes) tienen un Churn superior a los contratos de uno o dos años.
                    * **Estrategia:** Se recomienda aplicarles un cambio de contrato + descuentos para que no efecte en su pago.
                    """)
                elif var_cat == "InternetService":
                    st.info("""
                    * **Observación:** Los clientes con servicio de **Fiber optic** (Fibra óptica) presentan mayor Churn que los de DSL. 
                    * **Análisis:** Los motivos de baja podrian relacionarse con el precio o problemas técnicos en la red de fibra.
                    """)

        elif item_seleccionado == "Ítem 9: Análisis basado en parámetros seleccionados":
            st.header("Ítem 9: Análisis basado en parámetros seleccionados")
            st.write("En esta sección, puedes seleccionar las variables y segmentos para explorar hallazgos específicos.")

            #Selección múltiple para cabeceras
            st.subheader("Configuración de Columnas")
            
            #Se define que columnas utilizar
            columnas_usuario = st.multiselect(
                "Seleccione las variables que desea incluir en el análisis:",
                options=df.columns.tolist(),
                default=["tenure", "MonthlyCharges", "Churn", "Contract"]
            )

            if columnas_usuario:
                #Filtros
                segmento = st.selectbox(
                    "Filtrar análisis por tipo de Contrato:",
                    options=["Todos"] + df['Contract'].unique().tolist()
                )

                # Aplicar filtro al dataframe según la elección
                if segmento == "Todos":
                    df_final = df[columnas_usuario]
                else:
                    df_final = df[df['Contract'] == segmento][columnas_usuario]

                #Resultados
                st.write(f"### Vista previa de datos ({segmento})")
                st.dataframe(df_final.head(10))

                #Análiis según selección
                cols_num_elegidas = df_final.select_dtypes(include=['number']).columns.tolist()

                if cols_num_elegidas:
                    st.subheader("Resumen Estadístico de Variables Seleccionadas")
                    
                    col_a_promediar = st.selectbox("Ver media de:", cols_num_elegidas)
                    
                    media_dinamica = df_final[col_a_promediar].mean()
                    
                    st.metric(
                        label=f"Media de {col_a_promediar} para el segmento {segmento}", 
                        value=f"{media_dinamica:.2f}"
                    )
                else:
                    st.warning("Seleccione al menos una columna numérica para poder ver sus estadísticas.")
                    
            else:
                st.error("Por favor, seleccione al menos una columna para comenzar el análisis.")

        elif item_seleccionado == "Ítem 10: Hallazgos clave":
            st.header("Ítem 10: Conclusiones y Hallazgos Clave (Insights)")
            st.write("""En esta sección, encontraremos los principales focos o motivos de la baja de clientes""")

            st.subheader("Mapa de Relaciones (Correlación)")
            st.write(""" realizaremos un cuadro con la depencia entre variables según la tonalidad y el dato ponderado""")

            df_num = df.select_dtypes(include=['number'])
            corr = df_num.corr()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.markdown("---")

            st.subheader("💡 Insights Estratégicos")

            col1, col2 = st.columns(2)

            with col1:
                st.info("### 1. Churn por Contrato")
                st.write("""
                Los clientes con contratos **mes a mes** representan el mayor ratio a Churn. 
                **Recomendación:** Campañas a cambios de tipo de contrato y aplicación de descuentos.
                """)

                st.success("### 2. Tiempo de Permanencia")
                st.write(f"""
                La media de permanencia (tenure) es de **{df['tenure'].mean():.1f} meses**. 
                Sin embargo, el riesgo de abandono cae drásticamente después del primer año. 
                **Recomendación:** Realizar campañas de fidelización recordándole sus beneficios como clientes o por sus pagos.
                """)

            with col2:
                st.warning("### 3. Churn por Cargo")
                st.write("""
                Los clientes que se van tienen, en promedio, **MonthlyCharges** más altos. 
                Uno de los motivos de Churn puede referise al precio que no puede pagar el cliente.
                """)

                st.error("### 4. Servicio de Fibra Óptica")
                st.write("""
                Se detectó una alta tasa de Churn en usuarios de Fibra Óptica. 
                **Hallazgo:** Problemas con el servicio o pagos para este tipo de fibra.
                """)
        elif item_seleccionado == "Conclusiones Finales":
            st.header("Conclusiones Finales")
            st.write(
               """
                * **1. Contratos Mensuales:** Los clientes con contratos *Month-to-month* presentan el mayor ratio de Churn. Se recomienda incentivar migraciones a planes mediante descuentos progresivos.
                * **2. Riesgo en la Permanencia:** La probabilidad de Churn es mayor durante los primeros **12 meses** de permanencia (*tenure*). Es vital priorizar campañas de babysitting (bienvenida) para asegurar permanencia en el cliente.
                * **3. Sensibilidad al Precio:** Clientes que poseen cargos mensuales (*MonthlyCharges*) por encima del promedio generan mayor Churn. Se sugiere mejorar los beneficios de planes para mantener retenido al cliente.
                * **4. Calidad en Fibra Óptica:** Existe una fuga atípica en usuarios de Fibra Óptica. Se requiere una revisión técnica de las instalaciones para validar y solucionar los problemas con este tipo de fibra.
                * **5. Agrupación de Servicios:** Se puede generar un segmento de cliente que posea servicios móviles y fijos a la vez y brindarles descuentos para poder blindarlos de la competencia.
                """)
        