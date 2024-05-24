#importamos las librerías
import pandas as pd

#importamos el archivo de funciones
from funciones import data_import_and_cleaning, graph_distribution_exited_customers, graph_numeric_var_density, graph_numeric_var_distribution, graph_gender_distribution_exited_customers,\
    graph_age_distribution_exited_customers, graph_geographic_distribution_exited_customers, graph_balance_distribution_exited_customers, graph_creditscore_distribution_exited_customers,\
    graph_tenure_distribution_exited_customers, graph_salary_distribution_exited_customers, graph_num_products_exited_customers, graph_has_crcard_exited_customers,\
    graph_activity_exited_customers, column_to_boolean_numeric, pairplot, correlation_heatmap, chi_squared_heatmap, abt_creation, dummies_correlation_heatmap, prediction_model

#llamamos al archivo desde el archivo yalm e importamos los dataframes
yalm_path = "../config.yaml"

#importamos y limpiamos el dataset
df = data_import_and_cleaning(yalm_path)

#imprimimos el gráfico circular de distribución de clientes que han dejado de serlo
graph_distribution_exited_customers(df)

#imprimimos el gráfico de densidad de todas las variables numéricas del dataframe
graph_numeric_var_density(df)

#imprimimos el gráfico de distribución de todas las variables numéricas del datagframe
graph_numeric_var_distribution(df)

#imprimimos el gráfico de barras con la distribución de clientes por género
graph_gender_distribution_exited_customers(df)

#imprimimos el gráfico de dispersión con la distribución de clientes por edad
graph_age_distribution_exited_customers(df)

#imprimimos el gráfico de barras con la distribución geográfica de clientes
graph_geographic_distribution_exited_customers(df)

#imprimimos el gráfico de barras con la distribución de balance (saldo) de clientes
graph_balance_distribution_exited_customers(df)

#imprimimos el gráfico de barras con la distribución del creditscore de clientes
graph_creditscore_distribution_exited_customers(df)

#imprimimos el gráfico de barras agrupadas con la distribución del permanencia de clientes
graph_tenure_distribution_exited_customers(df)

#imprimimos el gráfico de barras con la distribución de salario estimado de clientes
graph_salary_distribution_exited_customers(df)

#imprimimos el gráfico de barras agrupadas con la distribución del num de productos de los clientes
graph_num_products_exited_customers(df)

#imprimimos el gráfico de barras con la distribución de los clientes que tienen tarjeta de crédito y los qu eno
graph_has_crcard_exited_customers(df)

#imprimimos el gráfico de barras con la distribución de los clientes según si son activos o no
graph_activity_exited_customers(df)

#empezamos con los gráficos y preparación de datos para el análisis logarítimico

#reconvertimos la columna 'Exited' a tipo booleano numérico para el análisis logarítmico
df = column_to_boolean_numeric(df, 'Exited')

#creamos un pairplot para observar la relación entre todas las variables numéricas con la variable a predecir: Exited
pairplot(df, ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited'], target_column='Exited')

#creamos un mapa de calor para observar la correlación entre las variables numéricas
correlation_heatmap(df, ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited'])

#creamos un mapa de calor a partir del chi cuadrado para cuantificar la relación entre las variables categóricas
chi_squared_heatmap(df, ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited'])

#creamos un nuevo dataframe analítico (ABT) a partir del datraframe original
df_dummies = abt_creation(df)

#creamos un mapa de calor entre todas las variables dummies
dummies_correlation_heatmap(df_dummies)

#creamos el modelo de predicción utilizando un clasificador Bagging
prediction_model(df_dummies)