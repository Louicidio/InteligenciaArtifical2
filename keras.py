import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita otimizações oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduz mensagens de log do TensorFlow

import pandas as pd
import numpy as np
import tensorflow as tf
import kagglehub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Exemplo genérico de carregamento do dataset
path = "archive/formula-1-races-between-2020-2025"
results = pd.read_csv(path + "/results.csv")
drivers = pd.read_csv(path + "/drivers.csv")
races = pd.read_csv(path + "/races.csv")

# Exemplo de merge simplificado
data = results.merge(drivers, on="driverId").merge(races, on="raceId")

# Variável alvo: piloto vencedor
data["winner"] = (data["positionOrder"] == 1).astype(int)

# Features (exemplo simples: grid, year, round)
X = data[["grid", "year", "round"]].values
y = data["winner"].values

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo MLP
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binário: venceu ou não
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinar
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# Avaliação
loss, acc = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {acc:.2f}")
