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

def 