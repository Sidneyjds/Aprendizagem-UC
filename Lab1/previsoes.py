import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Carrega o conjunto de dados
data = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv", sep=";")

# Solicita ao usuário que insira os valores para a previsão
data_x = input("Introduza valores do vinho separados por ponto e vírgula (ex: 7.4;0.7;0.0;1.9;0.076;11.0;34.0;0.9978;3.51;0.56;9.4):\n")
data = data_x.split(";")

# Converte os valores para float
fmap_data = map(float, data)
flist_data = list(fmap_data)

# Prepara os dados para a previsão
data1 = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv", sep=";")
data2 = data1.iloc[:0, :11]  # Pega apenas as colunas sem dados
data_preparation = pd.DataFrame([flist_data], columns=list(data2.columns))  # Corrigi aqui

# Carrega o modelo treinado
try:
    loaded_model = p1.load(open('../white-wine_quality_predictor.pkl', 'rb'))  # Adicionei a extensão .pkl
except FileNotFoundError:
    print("O arquivo do modelo não foi encontrado. Por favor, verifique o caminho.")
    exit()

# Faz a previsão
y_pred = loaded_model.predict(data_preparation)

# Imprime a qualidade do vinho
print("Qualidade do vinho:", round(y_pred[0]))  # Arredondei para o número inteiro mais próximo

# Imprime os valores inseridos pelo usuário
for col, value in zip(data_preparation.columns, flist_data):
    print(f"{col}: {value}")

