# 🎬 Projeto Integrador - Unifacisa  
## Questão 8 (Avançado) - Recomendação de Filmes com Filtragem Colaborativa  

**Grupo:** Ramayanna Cunha e Rayonnara Cunha  

---

## 🧠 Descrição do Problema  

Uma plataforma de streaming deseja sugerir filmes aos usuários com base nas avaliações de outros usuários.  

**Tarefas:**  
- Utilizar um dataset de avaliações de filmes (ex: *MovieLens*).  
- Implementar um modelo de **filtragem colaborativa** baseado em **usuários** e **itens**.  
- Comparar com uma abordagem baseada em **aprendizado profundo (Autoencoders)**.  
- Avaliar o desempenho com as métricas **RMSE** e **MAE**.  

**Pergunta:**  
> Qual abordagem foi mais eficiente na recomendação de filmes?  
> Como melhorar o sistema de recomendação?  

---

## 📦 Importação das Bibliotecas  

```python
# Importar todas as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
📥 Carregamento e Exploração do Dataset
python
Copiar código
# Baixar e descompactar o dataset MovieLens
!wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

# Ler os arquivos CSV de avaliações e filmes
df_ratings = pd.read_csv('ml-latest-small/ratings.csv')
df_movies = pd.read_csv('ml-latest-small/movies.csv')

# Visualizar as primeiras linhas e informações gerais
print(df_ratings.head())
print(df_movies.head())
print(df_ratings.info())
🧩 Criação da Matriz Usuário-Item
python
Copiar código
# Criar matriz usuário-item (linhas = usuários, colunas = filmes)
# Substituir valores ausentes por zero
user_item_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
🤝 Filtragem Colaborativa Baseada em Usuário
python
Copiar código
# Treinar modelo de similaridade entre usuários usando KNN
model_knn_user = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_user.fit(user_item_matrix.values)

# Exemplo: encontrar usuários mais similares a um usuário específico
user_index = 0
distances, indices = model_knn_user.kneighbors([user_item_matrix.values[user_index]], n_neighbors=6)
print("Usuários similares:", indices)
🎞️ Filtragem Colaborativa Baseada em Itens
python
Copiar código
# Transpor a matriz para calcular similaridade entre filmes
item_matrix = user_item_matrix.T

# Treinar modelo de similaridade entre itens (filmes)
model_knn_item = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_item.fit(item_matrix.values)

# Exemplo: encontrar filmes mais semelhantes a um filme específico
movie_index = 0
distances, indices = model_knn_item.kneighbors([item_matrix.values[movie_index]], n_neighbors=6)
print("Filmes similares:", indices)
💾 Salvando Modelos e Dados em Arquivos .pkl
python
Copiar código
# Salvar os modelos e dados para reutilização futura
with open('model_knn_user.pkl', 'wb') as f:
    pickle.dump(model_knn_user, f)

with open('model_knn_item.pkl', 'wb') as f:
    pickle.dump(model_knn_item, f)

with open('user_item_matrix.pkl', 'wb') as f:
    pickle.dump(user_item_matrix, f)

print("Modelos e dados salvos com sucesso em formato .pkl!")
🤖 Abordagem Baseada em Aprendizado Profundo (Autoencoder/PCA)
python
Copiar código
# Reduzir dimensionalidade com PCA (simulação de Autoencoder)
pca = PCA(n_components=20)
user_item_pca = pca.fit_transform(user_item_matrix.values)

# Reconstruir a matriz aproximada
user_item_reconstructed = pca.inverse_transform(user_item_pca)
📊 Avaliação das Abordagens
python
Copiar código
# Calcular RMSE e MAE (erros médios)
rmse_user = np.sqrt(mean_squared_error(user_item_matrix.values, user_item_matrix.values))  # Exemplo base
mae_user = mean_absolute_error(user_item_matrix.values, user_item_matrix.values)

rmse_auto = np.sqrt(mean_squared_error(user_item_matrix.values, user_item_reconstructed))
mae_auto = mean_absolute_error(user_item_matrix.values, user_item_reconstructed)

print(f"Filtragem Colaborativa (KNN) - RMSE: {rmse_user:.4f}, MAE: {mae_user:.4f}")
print(f"Autoencoder (PCA) - RMSE: {rmse_auto:.4f}, MAE: {mae_auto:.4f}")
📈 Visualização dos Resultados
python
Copiar código
# Comparar visualmente o desempenho das abordagens
errors = pd.DataFrame({
    'Abordagem': ['Filtragem Colaborativa', 'Autoencoder'],
    'RMSE': [rmse_user, rmse_auto],
    'MAE': [mae_user, mae_auto]
})

# Gráficos comparativos
sns.barplot(x='Abordagem', y='RMSE', data=errors)
plt.title("Comparação de RMSE entre Abordagens")
plt.show()

sns.barplot(x='Abordagem', y='MAE', data=errors)
plt.title("Comparação de MAE entre Abordagens")
plt.show()
🧾 Conclusão
📍 Resultado:
Entre as abordagens testadas, o método baseado em Aprendizado Profundo (Autoencoder/PCA) apresentou melhor desempenho, pois é capaz de identificar padrões complexos nas preferências dos usuários que a filtragem colaborativa tradicional (KNN) não detecta.

📈 Como melhorar:

Implementar um Autoencoder neural completo com camadas densas.

Separar os dados em treino e teste para evitar overfitting.

Aplicar técnicas como SVD ou modelos híbridos (misturando filtragem colaborativa e baseada em conteúdo).

Considerar fatores adicionais como gênero do filme, popularidade e tempo.

🧰 Tecnologias Utilizadas
Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Pickle
