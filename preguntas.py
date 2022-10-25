"""
Regresi贸n Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construir谩 un modelo de regresi贸n lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el c贸digo presentado a continuaci贸n.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    # df = pd.read_csv('C:/Users/jlopezl/OneDrive - Renting Colombia S.A/Archivos/Personal/Especializaci髇/Ciencia de los datos/regresion-lineal-basica-gapminder-JuanesLopez/gm_2008_region.csv')
    df = pd.read_csv('gm_2008_region.csv')


    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df['life'].values
    X = df['fertility'].values

    # Imprima las dimensiones de `y`
    print(y.shape)

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.reshape(-1, 1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.reshape(-1, 1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresi贸n de algunas estad铆sticas b谩sicas
    Complete el c贸digo presentado a continuaci贸n.
    """

    # df = pd.read_csv('C:/Users/jlopezl/OneDrive - Renting Colombia S.A/Archivos/Personal/Especializaci髇/Ciencia de los datos/regresion-lineal-basica-gapminder-JuanesLopez/gm_2008_region.csv')
    df = pd.read_csv('gm_2008_region.csv')

    # Imprima las dimensiones del DataFrame
    print(df.shape)

    # Imprima la correlaci贸n entre las columnas `life` y `fertility` con 4 decimales.
    print(round(df.life.corr(df.fertility),4))

    # Imprima la media de la columna `life` con 4 decimales.
    print(round(df.life.mean(),4))

    # Imprima el tipo de dato de la columna `fertility`.
    print(type(df.fertility))

    # Imprima la correlaci贸n entre las columnas `GDP` y `life` con 4 decimales.
    print(round(df.GDP.corr(df.life),4))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el c贸digo presentado a continuaci贸n.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    # df = pd.read_csv('C:/Users/jlopezl/OneDrive - Renting Colombia S.A/Archivos/Personal/Especializaci髇/Ciencia de los datos/regresion-lineal-basica-gapminder-JuanesLopez/gm_2008_region.csv')
    df = pd.read_csv('gm_2008_region.csv')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df.fertility.values.reshape(-1,1)

    # Asigne a la variable los valores de la columna `life`
    y_life = df.life.values.reshape(-1,1)

    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresi贸n lineal
    reg = LinearRegression()

    # Cree El espacio de predicci贸n. Esto es, use linspace para crear
    # un vector con valores entre el m谩ximo y el m铆nimo de X_fertility
    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
    ).reshape(-1, 1)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_fertility, y_life)

    # Compute las predicciones para el espacio de predicci贸n
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(reg.score(X_fertility, y_life).round(4))


def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el c贸digo presentado a continuaci贸n.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    # df = pd.read_csv('C:/Users/jlopezl/OneDrive - Renting Colombia S.A/Archivos/Personal/Especializaci髇/Ciencia de los datos/regresion-lineal-basica-gapminder-JuanesLopez/gm_2008_region.csv')
    df = pd.read_csv('gm_2008_region.csv')

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df.fertility.values.reshape(-1,1)

    # Asigne a la variable los valores de la columna `life`
    y_life = df.life.values.reshape(-1,1)

    # Divida los datos de entrenamiento y prueba. La semilla del generador de n煤meros
    # aleatorios es 53. El tama帽o de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
    )

    # Cree una instancia del modelo de regresi贸n lineal
    linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    linearRegression.fit(X_train, y_train)

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
