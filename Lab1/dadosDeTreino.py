from matplotlib import pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns


data = pd.read_csv('Dataset/abalone.data', header=None)

print(data)
# Defina os nomes das colunas
column_names = ['comprimento', 'diametro', 'altura', 'peso_inteiro', 'peso_sem_casca', 'peso_viceras', 'peso_concha', 'n_aneis']

# Certifique-se de que o arquivo 'abalone.data' está no mesmo diretório que o script
#try:
data = pd.read_csv('Dataset/abalone.data', header=None, names=column_names)
#except FileNotFoundError:
print("O arquivo 'abalone.data' não foi encontrado. Por favor, verifique o caminho.")
    #exit()

# Imprime as primeiras linhas do dataframe
print(data.head())

# Converte a coluna 'n_aneis' para int
data['n_aneis'] = data['n_aneis'].astype(int)

# Imprime informações sobre o dataframe
print(data.info())
print(data.describe())

# Calcula a matriz de correlação
correlation_matrix = data.corr()
print(correlation_matrix)

# Plota a matriz de correlação
"""plt.figure(figsize=(10, 8))  # Corrigi o tamanho da figura para que ela seja visível
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()"""

# Prepara os dados para treinamento
X = data.drop(columns=['n_aneis'])
y = data['n_aneis']

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Imprime o tamanho dos conjuntos
print(f'Tamanho do conjunto de treinamento: {X_train.shape[0]}')
print(f'Tamanho do conjunto de teste: {X_test.shape[0]}')

# Cria e treina o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Faz previsões
y_pred = model.predict(X_test)

# Calcula o erro quadrático médio e o R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprime os resultados
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Prepara os dados para salvar o modelo
train_data = data[:3133]
data_X = train_data.iloc[:, 1:8]
data_Y = train_data.iloc[:, 8:9]

# Imprime os dados preparados
print(data_X)
print(data_Y)

# Cria e treina um novo modelo para salvar
regr = LinearRegression()
preditor_linear_model = regr.fit(data_X, data_Y)

# Salva o modelo em um arquivo pickle
preditor_Pickle = open('white-wine_quality_predictor.pkl', 'wb')  # Corrigi o nome do arquivo
print("white-wine_quality_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)
preditor_Pickle.close()
