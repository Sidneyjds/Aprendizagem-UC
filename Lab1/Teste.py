import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Carrega o conjunto de dados
data = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv", sep=";")

# Prepara os dados para avaliação
evaluation_data = data[1001:]
data_X = evaluation_data.iloc[:, 0:11]
data_Y = evaluation_data.iloc[:, 11:12]

# Imprime os tipos dos dados
print(type(evaluation_data))
print(type(data_X))

# Carrega o modelo treinado
try:
    loaded_model = p1.load(open('../white-wine_quality_predictor.pkl', 'rb'))  # Adicionei a extensão .pkl
    print("Model loaded successfully.")
except FileNotFoundError:
    print("The model file was not found. Please check the path.")
    exit()

# Imprime os coeficientes do modelo
print("Coefficients: \n", loaded_model.coef_)

# Faz previsões
y_pred = loaded_model.predict(data_X)

# Calcula a diferença entre as previsões e os valores reais
z_pred = y_pred - data_Y.values  # Corrigi aqui

# Inicializa as variáveis para calcular a precisão
right = 0
wrong = 0
total = 0

# Avalia a precisão
for x in z_pred:
    z = round(x[0])  # Corrigi aqui
    total += 1
    if z == 0:
        right += 1
    else:
        wrong += 1

# Imprime as precisões
print("Precisão de previsões corretas: ", right / total)
print("Precisão de previsões incorretas: ", wrong / total)
