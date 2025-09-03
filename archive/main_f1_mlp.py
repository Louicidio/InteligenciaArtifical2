#!/usr/bin/env python3
# main_f1_mlp.py
# Pipeline para treinar uma MLP (scikit-learn) que prediz se um piloto vence uma corrida (is_winner)
# Requisitos: pandas, numpy, scikit-learn, joblib
# Ex.: pip install pandas numpy scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import joblib, os, warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def build_and_train(csv_path='F1 Races 2020-2024.csv', out_dir='models', test_year=2024):
    # 1) carregar dados
    df = pd.read_csv(csv_path)

    # 2) construir target: vencedor por raceId (quem tem pontos máximos naquela corrida)
    max_points = df.groupby('raceId')['points'].transform('max')
    df['is_winner'] = (df['points'] == max_points).astype(int)

    # 3) features selecionadas (adapte se quiser outras)
    features = [
        'grid', 'rainy', 'Turns', 'Length',
        'Driver Top 3 Finish Percentage (Last Year)',
        'Constructor Top 3 Finish Percentage (Last Year)',
        'Driver Top 3 Finish Percentage (This Year till last race)',
        'Constructor Top 3 Finish Percentage (This Year till last race)',
        'Driver Avg position (Last Year)',
        'Constructor Avg position (Last Year)',
        'Driver Average Position (This Year till last race)',
        'Constructor Average Position (This Year till last race)',
        'position_previous_race', 'nro_cond_escuderia',
        'prom_points_10', 'wins', 'wins_cons'
    ]
    # mantém apenas as que existem no CSV
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0).copy()
    y = df['is_winner'].values

    # 4) encoders de driver/constructor (úteis para generalizar por piloto/escuderia)
    if 'driverId' in df.columns and 'constructorId' in df.columns:
        le_driver = LabelEncoder()
        le_constructor = LabelEncoder()
        X['driver_enc'] = le_driver.fit_transform(df['driverId'])
        X['constructor_enc'] = le_constructor.fit_transform(df['constructorId'])
    else:
        le_driver = None
        le_constructor = None

    # 5) normalização
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 6) divisão treino/teste: por default treino em anos < test_year e testa em test_year
    if 'year' in df.columns:
        train_mask = df['year'] < test_year
        if train_mask.sum() < 10 or (~train_mask).sum() < 10:
            # fallback aleatório estratificado
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        else:
            X_train = X_scaled[train_mask.values]
            y_train = y[train_mask.values]
            X_test = X_scaled[~train_mask.values]
            y_test = y[~train_mask.values]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("Tamanho treino:", X_train.shape, "Tamanho teste:", X_test.shape, "Positivos treino:", int(y_train.sum()))

    # 7) MLPClassifier (sklearn) — implementa backprop internamente
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16),
                        activation='relu',
                        solver='adam',
                        learning_rate_init=0.001,
                        max_iter=300,
                        random_state=42,
                        early_stopping=True,
                        n_iter_no_change=10)

    mlp.fit(X_train, y_train)

    pred_prob = mlp.predict_proba(X_test)[:, 1]
    pred = mlp.predict(X_test)

    print('\nAcurácia:', accuracy_score(y_test, pred))
    try:
        print('ROC AUC:', roc_auc_score(y_test, pred_prob))
    except Exception:
        print('ROC AUC: calculo não disponível (possível falta de positivos/negativos no conjunto de teste)')

    print('\nRelatório:\n', classification_report(y_test, pred, digits=3))

    # 8) salvar artefatos
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(mlp, os.path.join(out_dir, 'f1_mlp_sklearn.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    if le_driver is not None:
        joblib.dump(le_driver, os.path.join(out_dir, 'le_driver.joblib'))
    if le_constructor is not None:
        joblib.dump(le_constructor, os.path.join(out_dir, 'le_constructor.joblib'))

    print('\nModelos e encoders salvos em', out_dir)
    return mlp, scaler, (le_driver, le_constructor)

if __name__ == '__main__':
    build_and_train()
