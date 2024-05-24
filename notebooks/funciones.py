def data_import_and_cleaning(yalm_path):

    """
    Lee un archivo de configuración YAML, carga un dataset desde una ruta especificada en el archivo,
    realiza varias transformaciones en el dataset y devuelve el DataFrame resultante.

    Args:
        config_path (str): Ruta al archivo de configuración YAML.

    Returns:
        pandas.DataFrame: DataFrame procesado con las transformaciones aplicadas.
    """

    import pandas as pd
    import yaml

    #importamos el dataset desde el archivo yaml
    with open (yalm_path, 'r') as file:
        config = yaml.safe_load(file)

    # Leemos el dataset
    df = pd.read_csv(config['data']['df'])

    # Eliminamos las columnas RowNumber y Surname del dataframe
    columns_to_drop = ['RowNumber', 'Surname']
    df = df.drop(columns=columns_to_drop)

    # Agrupamos la edad en grupos hasta 30 adultos jóvenes, hasta 55 adultos de mediana edad y a partir de 55 adultos mayores
    age_bins = [18, 30, 55, df['Age'].max()]
    age_labels = ['Young adults', 'Middle-aged adults', 'Older adults']
    df['Age_grouped'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)


    # Agrupamos el Balance en grupos de cero, bajo, medio, medio alto y alto
    balance_bins = [0, 1, 50000, 100000, 150000, df['Balance'].max()]
    balance_labels = ['Zero', 'Low', 'Medium', 'Medium high', 'High']
    df['Balance_grouped'] = pd.cut(df['Balance'], bins=balance_bins, labels=balance_labels, include_lowest=True)

    # Agrupamos el CreditScore en alto, medio y bajo riesgo
    credit_bins = [300, 620, 700, df['CreditScore'].max()]
    credit_labels = ['High risk', 'Medium risk', 'Low risk']
    df['CreditScore_grouped'] = pd.cut(df['CreditScore'], bins=credit_bins, labels=credit_labels, include_lowest=True)

    # Agrupamos el EstimatedSalary en alto, medio y bajo
    salary_bins = [0, 50000, 100000, df['EstimatedSalary'].max()]
    salary_labels = ['Low', 'Medium', 'High']
    df['EstimatedSalary_grouped'] = pd.cut(df['EstimatedSalary'], bins=salary_bins, labels=salary_labels, include_lowest=True)

    # Convertimos la columna Exited a tipo objeto para el análisis EDA
    df['Exited'] = df['Exited'].map({0: 'No', 1: 'Yes'})

    return df

def graph_distribution_exited_customers(df):
    
    """
    Calcula el porcentaje de clientes que han dejado de serlo y los que no,
    y crea un gráfico circular para visualizar esta distribución.

    Args:
        df (pandas.DataFrame): DataFrame que contiene la columna 'Exited' con la información de los clientes que han dejado de serlo.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico circular mostrando la distribución de clientes que han dejado de serlo y los que no.
    """
    import pandas as pd
    import plotly.express as px
    
    # Calculamos el porcentaje de clientes que siguen siéndolo y los que no
    Exited_percent = (df['Exited'].value_counts(normalize=True) * 100).reset_index()
    Exited_percent.columns = ['Exited', 'proportion']

    # Creamos el gráfico circular para ver la distribución de clientes que han dejado de serlo y los que no
    fig = px.pie(Exited_percent, values='proportion', names='Exited', title='Distribution of exited customers')
    
    # Mostramos el gráfico
    fig.show()

def graph_numeric_var_density(df):

    """
    Crea una figura con subgráficos para mostrar la densidad de las variables numéricas en el DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las variables numéricas a analizar.

    Returns:
        matplotlib.figure.Figure: Figura que contiene los subgráficos con la distribución de densidad de las variables numéricas.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Comprobamos la densidad de todas las variables numéricas
    numeric_var = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

    # Creamos una figura con un arreglo de 3 filas y 2 columnas de subgráficos
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Iteramos sobre las variables y los ejes correspondientes
    for i, var in enumerate(numeric_var):
        row, col = divmod(i, 2)
        sns.kdeplot(df[var], shade=True, ax=axes[row, col])
        axes[row, col].set_title(f'{var} density distribution')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Density')

    # Ocultamos el último subgráfico vacío
    axes[2, 1].axis('off')

    # Ajustamos el espacio entre los subgráficos
    plt.tight_layout()

    # Mostramos la figura con todos los subgráficos
    plt.show()

    return fig

def graph_numeric_var_distribution(df):

    """
    Crea histogramas para mostrar la distribución de las variables numéricas en el DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las variables numéricas a analizar.

    Returns:
        matplotlib.figure.Figure: Figura que contiene los histogramas de las variables numéricas.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

        # Comprobamos la distribución de todas las variables numéricas
    numeric_var = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']

    # Creamos histogramas para las variables numéricas
    ax = df[numeric_var].hist(figsize=(15, 20), bins=60, xlabelsize=10, ylabelsize=10)
    for axis in ax.flatten():
        axis.ticklabel_format(style='plain', axis='x')

    # Mostramos la figura con todos los histogramas
    plt.tight_layout()
    plt.show()

def graph_gender_distribution_exited_customers(df):

    """
    Agrupa los datos por Género y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de género entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'Gender', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de género entre los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por Género y Exited y contamos la cantidad de clientes en cada grupo
    Gender_df = df.groupby(['Gender', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos un gráfico para ver la distribución de género entre los clientes que se fueron
    fig = px.bar(Gender_df, x='Gender', y='CustomerId', color='Exited', title='Gender distribution among exited customers')
    
    # Mostramos el gráfico
    fig.show()

def graph_age_distribution_exited_customers(df):

    """
    Agrupa los datos por Edad y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de dispersión para mostrar la relación entre la edad y los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'Age', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de dispersión mostrando la relación entre la edad y los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por Edad y Exited y contamos la cantidad de clientes en cada grupo
    Age_df = df.groupby(['Age', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos un gráfico para ver la relación entre la edad y los clientes que se fueron
    fig = px.scatter(Age_df, x='Age', y='CustomerId', color='Exited', title='Age distribution among exited customers')
    
    # Mostramos el gráfico
    fig.show()

def graph_geographic_distribution_exited_customers(df):

    """
    Agrupa los datos por Geografía y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la relación entre la geografía y los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'Geography', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la relación entre la geografía y los clientes que se fueron.
    """    

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por Geography y Exited y contamos la cantidad de clientes en cada grupo
    Geography_df = df.groupby(['Geography', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos un gráfico para ver la distribución de geografía entre los clientes que se fueron
    fig = px.bar(Geography_df, x='Geography', y='CustomerId', color='Exited', title='Geographic distribution among exited customers')
    
    # Mostramos el gráfico
    fig.show()

def graph_balance_distribution_exited_customers(df):

    """
    Agrupa los datos por Balance_grouped y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la relación entre el Balance y los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'Balance_grouped', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la relación entre el balance y los clientes que se fueron.
    """    

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por Balance y Exited y contamos la cantidad de clientes en cada grupo
    Balance_df = df.groupby(['Balance_grouped', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos un gráfico para ver la distribución de geografía entre los clientes que se fueron
    fig = px.bar(Balance_df.rename(columns={'CustomerId':'Num_clients', 'Balance_grouped':'Balance'}), x='Balance', y='Num_clients', color='Exited', title='Balance distribution among exited customers',\
             hover_data=['Balance'])
    
    #Mostramos el gráfico
    fig.show()

def graph_creditscore_distribution_exited_customers(df):

    """
    Agrupa los datos por CreditScore agrupado y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de dispersión para mostrar la relación entre el Balance y los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'CreditScore_grouped', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de dispersión mostrando la relación entre el creditscore y los clientes que se fueron.
    """    

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por CreditScore y Exited y contamos la cantidad de clientes en cada grupo
    CreditScore_df = df.groupby(['CreditScore_grouped', 'Exited'])['CustomerId'].count().reset_index()
    
    # Creamos un gráfico para ver la distribución de geografía entre los clientes que se fueron
    fig = px.bar(CreditScore_df, x='CreditScore_grouped', y='CustomerId', color='Exited', title='CreditScore distribution among exited customers')
    fig.show()
    
    #Mostramos el gráfico
    fig.show()

def graph_tenure_distribution_exited_customers(df):

    """
    Agrupa los datos por Tenure y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de Tenure entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'Tenure', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de Tenure entre los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por Tenure y Exited y contamos la cantidad de clientes en cada grupo
    Tenure_df = df.groupby(['Tenure', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos el gráfico
    fig = px.bar(Tenure_df, x="Tenure", y="CustomerId", color="Exited", barmode='group')

    # Modificamos el título y etiquetas del gráfico
    fig.update_layout(title="Tenure distribution among exited customers", xaxis_title='Tenure', yaxis_title='NumClients')

    # Mostramos el gráfico
    fig.show()

def graph_salary_distribution_exited_customers(df):

    """
    Agrupa los datos por EstimatedSalary_grouped y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de EstimatedSalary entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'EstimatedSalary_grouped', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de EstimatedSalary entre los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por EstimatedSalary y Exited y contamos la cantidad de clientes en cada grupo
    EstimatedSalary_df = df.groupby(['EstimatedSalary_grouped', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos un gráfico para ver la distribución de EstimatedSalary entre los clientes que se fueron
    fig = px.bar(EstimatedSalary_df, x='EstimatedSalary_grouped', y='CustomerId', color='Exited', title='EstimatedSalary distribution among exited customers')
    
    # Mostramos el gráfico
    fig.show()

def graph_num_products_exited_customers(df):
    
    """
    Agrupa los datos por NumOfProducts y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de NumOfProducts entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'NumOfProducts', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de NumOfProducts entre los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por NumOfProducts y Exited y contamos la cantidad de clientes en cada grupo
    NumOfProducts_df = df.groupby(['NumOfProducts', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos el gráfico
    fig = px.bar(NumOfProducts_df, x="NumOfProducts", y="CustomerId", color="Exited", barmode='group')

    # Modificamos el título y etiquetas del gráfico
    fig.update_layout(title="NumOfProducts distribution among exited customers", xaxis_title='NumOfProducts', yaxis_title='NumClients')

    # Mostramos el gráfico
    fig.show()

def graph_has_crcard_exited_customers(df):

    """
    Agrupa los datos por HasCrCard y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de HasCrCard entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'HasCrCard', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de HasCrCard entre los clientes que se fueron.
    """

    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por HasCrCard y Exited y contamos la cantidad de clientes en cada grupo
    HasCrCard_df = df.groupby(['HasCrCard', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos el gráfico
    fig = px.bar(HasCrCard_df, x="HasCrCard", y="CustomerId", color="Exited", barmode='group')

    # Modificamos el título y etiquetas del gráfico
    fig.update_layout(title="HasCrCard distribution among exited customers", xaxis_title='HasCrCard', yaxis_title='NumClients')

    # Mostramos el gráfico
    fig.show()

def graph_activity_exited_customers(df):

    """
    Agrupa los datos por IsActiveMember y Exited, cuenta la cantidad de clientes en cada grupo,
    y crea un gráfico de barras para mostrar la distribución de IsActiveMember entre los clientes que se fueron.

    Args:
        df (pandas.DataFrame): DataFrame que contiene las columnas 'IsActiveMember', 'Exited' y 'CustomerId'.

    Returns:
        plotly.graph_objects.Figure: Figura del gráfico de barras mostrando la distribución de IsActiveMember entre los clientes que se fueron.
    """
    
    import pandas as pd
    import plotly.express as px

    # Agrupamos los datos por IsActiveMember y Exited y contamos la cantidad de clientes en cada grupo
    IsActiveMember_df = df.groupby(['IsActiveMember', 'Exited'])['CustomerId'].count().reset_index()

    # Creamos el gráfico
    fig = px.bar(IsActiveMember_df, x="IsActiveMember", y="CustomerId", color="Exited", barmode='group')

    # Modificamos el título y etiquetas del gráfico
    fig.update_layout(title="IsActiveMember distribution among exited customers", xaxis_title='IsActiveMember', yaxis_title='NumClients')

    # Mostramos el gráfico
    fig.show()

def column_to_boolean_numeric(df, column):

    """
    Convierte una columna del dataframe de tipo categórico a tipo numérico booleano.

    Args:
    df (pandas.DataFrame): El dataframe que contiene la columna a convertir.
    columna (str): El nombre de la columna a convertir.

    Returns:
    pandas.DataFrame: El dataframe con la columna convertida a tipo numérico.
    """
    df[column] = df[column].map({'No': 0, 'Yes': 1})

    return df

def pairplot(df, numeric_var, target_column='Exited'):
    
    """
    Genera un pairplot para verificar la relación entre las variables numéricas en un dataframe.

    Args:
    df (pandas.DataFrame): El dataframe que contiene las variables.
    numeric_var (list): Lista de nombres de las variables numéricas a incluir en el pairplot.
    target_column (str, opcional): El nombre de la columna que se usará para diferenciar los datos en el pairplot. 
                                    Por defecto, 'Exited'.

    Returns:
    None
    """

    import seaborn as sns

    numeric_var = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']

    sns.pairplot(df[numeric_var], hue=target_column)

def correlation_heatmap(df, numeric_var):

    """
    Crea un gráfico de correlación de las variables numéricas en un dataframe.

    Args:
    df (pandas.DataFrame): El dataframe que contiene las variables numéricas.
    numeric_var (list): Lista de nombres de las variables numéricas a incluir en el gráfico de correlación.

    Returns:
    None
    """

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Calculamos la matriz de correlación
    corr = np.abs(df[numeric_var].corr())

    # Creamos la máscara para la representación triangular
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Configuramos el gráfico de matplotlib
    f, ax = plt.subplots(figsize=(10, 10))

    # Generamos un mapa de colores personalizado
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Dibujamos el heatmap con la máscara y la relación de aspecto correcta
    sns.heatmap(corr, mask=mask, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=corr, cmap=cmap)

    # Mostramos el mapa de calor
    plt.show()

def chi_squared_heatmap(df, categoric_var):

    """
    Calcula el chi cuadrado para cuantificar la relación entre las variables categóricas y
    crea un mapa de calor para visualizar los resultados.

    Args:
    df (pandas.DataFrame): El dataframe que contiene las variables categóricas.
    categoric_var (list): Lista de nombres de las variables categóricas.

    Returns:
    None
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import chi2_contingency

    results = {}
    for col1 in df[categoric_var]:
        for col2 in df[categoric_var]:
            if col1 != col2:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                results[(col1, col2)] = {'chi2': chi2, 'p': p}

    # Organizamos los datos en una matriz cuadrada
    chi_squared_matrix = pd.DataFrame.from_dict(results, orient='index').reset_index()

    # Creamos un mapa de calor con seaborn
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(chi_squared_matrix.pivot(index='level_0', columns='level_1', values='chi2'),
                          annot=True,
                          cmap='coolwarm',
                          linewidths=0.5,
                          fmt=".2f")
    heatmap.set_title('Chi-Squared correlation map among categoric vars')

    # Mostramos el mapa de calor
    plt.show()

def abt_creation(df):

    """
    Crea una tabla de datos analíticos (ABT) a partir de un dataframe dado.

    Args:
    df (pandas.DataFrame): El dataframe original.

    Returns:
    pandas.DataFrame: La tabla de datos analíticos (ABT) creada.
    """
    import pandas as pd

    # Eliminamos la columna CustomerId
    df = df.drop(columns=['CustomerId'])

    # Creamos una nueva columna con el promedio del número de productos por geografía y la desviación estándar
    NumOfProducts_per_geography_sum_df = df[['Geography', 'NumOfProducts']].groupby('Geography').mean().reset_index()
    NumOfProducts_per_geography_std_df = df[['Geography', 'NumOfProducts']].groupby('Geography').std().reset_index()

    # Merge left para añadir los nuevos dataframes al original
    df = pd.merge(df, NumOfProducts_per_geography_sum_df, on='Geography', how='left')
    df = pd.merge(df, NumOfProducts_per_geography_std_df, on='Geography', how='left')

    # Cambiamos el nombre de las columnas
    df = df.rename(columns={'NumOfProducts_y': 'Avg_NumOfProducts_per_Geography', 'NumOfProducts' : 'Std_NumOfProducts_per_Geography', 'NumOfProducts_x' : 'NumOfProducts'})

    # Convertimos las variables categóricas en booleanas
    df_dummies = pd.get_dummies(df, columns=['Geography', 'Gender'])

    # Creamos una nueva columna con el promedio por valor agrupado de Age
    age_df = df_dummies.groupby('Age_grouped').agg({'Age':'mean'}).reset_index()
    df_dummies = pd.merge(df_dummies, age_df, on='Age_grouped', how='left')

    # Creamos una nueva columna con el promedio por valor agrupado de Balance
    balance_df = df_dummies.groupby('Balance_grouped').agg({'Balance':'mean'}).reset_index()
    df_dummies = pd.merge(df_dummies, balance_df, on='Balance_grouped', how='left')

    # Creamos una nueva columna con el promedio por valor agrupado de CreditScore
    creditscore_df = df_dummies.groupby('CreditScore_grouped').agg({'CreditScore':'mean'}).reset_index()
    df_dummies = pd.merge(df_dummies, creditscore_df, on='CreditScore_grouped', how='left')

    # Creamos una nueva columna con el promedio por valor agrupado de EstimatedSalary
    estimatedsalary_df = df_dummies.groupby('EstimatedSalary_grouped').agg({'EstimatedSalary':'mean'}).reset_index()
    df_dummies = pd.merge(df_dummies, estimatedsalary_df, on='EstimatedSalary_grouped', how='left')

    # Cambiamos el nombre de las columnas
    df_dummies = df_dummies.rename(columns={'CreditScore_x': 'CreditScore', 'CreditScore_y' : 'Avg_grouped_CreditScore', 'Age_x' : 'Age', 'Age_y' : 'Avg_grouped_Age', 'Balance_x' : 'Balance', 'Balance_y' : 'Avg_grouped_Balance', 'EstimatedSalary_x' : 'EstimatedSalary',\
                    'EstimatedSalary_y' : 'Avg_grouped_EstimatedSalary'})

    # Eliminamos las columnas no necesarias
    columns_to_drop = ['Age_grouped', 'Balance_grouped', 'CreditScore_grouped', 'EstimatedSalary_grouped']
    df_dummies = df_dummies.drop(columns=columns_to_drop)

    return df_dummies

def dummies_correlation_heatmap(df_dummies):

    """
    Calcula el índice de correlación entre todas las variables de df_dummies y crea un mapa de calor para visualizar la correlación.

    Args:
    df_dummies (pandas.DataFrame): El dataframe que contiene las variables dummy.

    Returns:
    None
    """

    import numpy as np
    import pandas as pd
    import plotly.express as px

    # Calculamos el índice de correlación entre todas las variables
    corr = df_dummies.corr()

    # Creamos una máscara para la parte superior del triángulo
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Aplicamos la máscara a la matriz de correlación
    corr_masked = corr.mask(mask)

    # Convertimos la matriz de correlación en un DataFrame de larga distancia
    corr_long = corr.reset_index().melt(id_vars='index')
    corr_long.columns = ['Variable1', 'Variable2', 'Correlation']

    # Creamos el gráfico de calor
    fig = px.imshow(corr, 
                    labels=dict(x="Variables", y="Variables", color="Correlation"),
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1, zmax=1,
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    aspect="equal"  # Hace el gráfico cuadrado
                   )

    # Agregamos anotaciones
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            if not mask[i, j]:
                fig.add_annotation(dict(
                    x=corr.columns[j],
                    y=corr.index[i],
                    text=str(round(corr.values[i, j], 2)),
                    showarrow=False,
                    font=dict(size=10)
                ))

    fig.update_layout(
        title="Heatmap of Correlation Matrix",
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.index),
        autosize=False,
        width=1600,
        height=1000,
    )

    fig.show()

def prediction_model(df_dummies):

    """
    Entrena un modelo de clasificación utilizando un clasificador Bagging, lo evalúa y muestra las métricas de evaluación.

    Args:
    df_dummies (pandas.DataFrame): El dataframe que contiene las variables dummy.

    Returns:
    tuple: Una tupla que contiene las métricas de evaluación (precision, recall, f1_score).
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Seleccionamos las variables que emplearemos en el algoritmo y definimos nuestro target en 'Exited'
    features = df_dummies[['Age', 'Balance', 'IsActiveMember', 'NumOfProducts', 'Tenure', 'Gender_Female', 'Geography_France', 'Geography_Spain', 'Geography_Germany']]
    target = df_dummies['Exited']

    # Dividimos los datos en conjuntos de entrenamiento y prueba, donde el 20% de los datos se utiliza para la prueba
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)

    # Estandarizamos los datos
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # Probamos a balancear el modelo con SMOTE
    sm = SMOTE(random_state=1, sampling_strategy=1.0)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_standardized, y_train)

    # Probamos el modelo bagging
    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=20),
                                n_estimators=45,
                                max_samples=1000)

    # Entrenamos el modelo
    bagging.fit(X_train_sm, y_train_sm)

    # Evaluamos el modelo
    pred = bagging.predict(X_test_standardized)

    # Calculamos métricas de evaluación
    precision_bagging = precision_score(y_test, pred, average='macro')
    recall_bagging = recall_score(y_test, pred, average='macro')
    f1_bagging = f1_score(y_test, pred, average='macro')

    # Imprimimos las métricas
    print("La precisión del modelo predictivo para la variable 'Exited' es la siguiente:")
    print("Precision:", precision_bagging)
    print("Recall:", recall_bagging)
    print("F1 Score:", f1_bagging)

    # Devolvemos las métricas de evaluación
    return precision_bagging, recall_bagging, f1_bagging