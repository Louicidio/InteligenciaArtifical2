import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita otimizações oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduz mensagens de log do TensorFlow

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🏎️ MLP PARA PREVISÃO CAMPEÃO F1 2025 - REDE NEURAL COM BACKPROPAGATION 🏆")
print("="*80)

# Função para criar e treinar o modelo MLP
def create_mlp_model(input_dim, hidden_layers, neurons_per_layer, learning_rate, dropout_rate=0.2):
    """
    Cria um modelo MLP personalizado
    
    Args:
        input_dim: Dimensão de entrada
        hidden_layers: Número de camadas ocultas
        neurons_per_layer: Lista com neurônios por camada ou int único
        learning_rate: Taxa de aprendizado
        dropout_rate: Taxa de dropout
    """
    model = Sequential()
    
    # Se neurons_per_layer é um inteiro, usa o mesmo para todas as camadas
    if isinstance(neurons_per_layer, int):
        neurons_list = [neurons_per_layer] * hidden_layers
    else:
        neurons_list = neurons_per_layer
    
    # Primeira camada oculta
    model.add(Dense(neurons_list[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    # Camadas ocultas adicionais
    for i in range(1, hidden_layers):
        model.add(Dense(neurons_list[i] if i < len(neurons_list) else neurons_list[-1], 
                       activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Camada de saída (classificação binária: campeão ou não)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def prepare_championship_data():
    """
    Prepara os dados para treinar o modelo de previsão de campeão
    """
    print("\n📊 Preparando dados para treinamento da MLP...")
    
    # Carregamento dos datasets
    results = pd.read_csv("results.csv")
    drivers = pd.read_csv("drivers.csv")
    races = pd.read_csv("races.csv")
    driver_standings = pd.read_csv("driver_standings.csv")
    
    # Merge dos dados
    data = results.merge(drivers, on="driverId").merge(races, on="raceId")
    standings_data = driver_standings.merge(races, on="raceId")
    
    # Focar nos anos 2015-2023 para treinamento (deixando 2024 para teste)
    training_years = list(range(2015, 2024))
    
    # Criar dataset de features por piloto por ano
    yearly_features = []
    
    for year in training_years:
        year_data = data[data['year'] == year]
        year_standings = standings_data[standings_data['year'] == year]
        
        if year_standings.empty:
            continue
            
        # Pegar posição final de cada piloto no ano
        final_standings = year_standings.loc[year_standings.groupby('driverId')['raceId'].idxmax()]
        
        # Identificar o campeão do ano
        champion_id = final_standings.loc[final_standings['points'].idxmax(), 'driverId']
        
        # Para cada piloto ativo no ano
        active_drivers = year_data['driverId'].unique()
        
        for driver_id in active_drivers:
            driver_year_data = year_data[year_data['driverId'] == driver_id]
            driver_info = drivers[drivers['driverId'] == driver_id]
            
            if driver_info.empty or len(driver_year_data) == 0:
                continue
                
            driver_info = driver_info.iloc[0]
            
            # Converter position para numérico
            driver_year_data = driver_year_data.copy()
            driver_year_data['position_num'] = pd.to_numeric(driver_year_data['position'], errors='coerce')
            
            # Calcular features
            total_races = len(driver_year_data)
            wins = len(driver_year_data[driver_year_data['position_num'] == 1])
            podiums = len(driver_year_data[driver_year_data['position_num'].isin([1, 2, 3])])
            points_total = driver_year_data['points'].sum()
            
            # Posições e DNFs
            finished_races = driver_year_data[driver_year_data['position_num'].notna()]
            avg_position = finished_races['position_num'].mean() if len(finished_races) > 0 else 20
            dnf_rate = len(driver_year_data[driver_year_data['position_num'].isna()]) / total_races
            
            # Taxas
            win_rate = wins / total_races if total_races > 0 else 0
            podium_rate = podiums / total_races if total_races > 0 else 0
            points_per_race = points_total / total_races if total_races > 0 else 0
            
            # Target: 1 se é campeão, 0 caso contrário
            is_champion = 1 if driver_id == champion_id else 0
            
            yearly_features.append({
                'year': year,
                'driverId': driver_id,
                'driver_name': f"{driver_info['forename']} {driver_info['surname']}",
                'total_races': total_races,
                'wins': wins,
                'podiums': podiums,
                'points_total': points_total,
                'win_rate': win_rate,
                'podium_rate': podium_rate,
                'avg_position': avg_position,
                'dnf_rate': dnf_rate,
                'points_per_race': points_per_race,
                'is_champion': is_champion
            })
    
    df_features = pd.DataFrame(yearly_features)
    
    print(f"✅ Dataset criado: {len(df_features)} registros de {len(training_years)} anos")
    print(f"📈 Campeões no dataset: {df_features['is_champion'].sum()}")
    
    return df_features

def prepare_2024_data():
    """
    Prepara dados de 2024 para previsão
    """
    print("\n📊 Preparando dados de 2024 para previsão...")
    
    results = pd.read_csv("results.csv")
    drivers = pd.read_csv("drivers.csv")
    races = pd.read_csv("races.csv")
    
    # Dados de 2024
    data_2024 = results.merge(drivers, on="driverId").merge(races, on="raceId")
    data_2024 = data_2024[data_2024['year'] == 2024]
    
    # Features para cada piloto de 2024
    features_2024 = []
    active_drivers_2024 = data_2024['driverId'].unique()
    
    for driver_id in active_drivers_2024:
        driver_data = data_2024[data_2024['driverId'] == driver_id]
        driver_info = drivers[drivers['driverId'] == driver_id]
        
        if driver_info.empty:
            continue
            
        driver_info = driver_info.iloc[0]
        driver_data = driver_data.copy()
        driver_data['position_num'] = pd.to_numeric(driver_data['position'], errors='coerce')
        
        # Calcular mesmas features do treino
        total_races = len(driver_data)
        wins = len(driver_data[driver_data['position_num'] == 1])
        podiums = len(driver_data[driver_data['position_num'].isin([1, 2, 3])])
        points_total = driver_data['points'].sum()
        
        finished_races = driver_data[driver_data['position_num'].notna()]
        avg_position = finished_races['position_num'].mean() if len(finished_races) > 0 else 20
        dnf_rate = len(driver_data[driver_data['position_num'].isna()]) / total_races
        
        win_rate = wins / total_races if total_races > 0 else 0
        podium_rate = podiums / total_races if total_races > 0 else 0
        points_per_race = points_total / total_races if total_races > 0 else 0
        
        features_2024.append({
            'driverId': driver_id,
            'driver_name': f"{driver_info['forename']} {driver_info['surname']}",
            'total_races': total_races,
            'wins': wins,
            'podiums': podiums,
            'points_total': points_total,
            'win_rate': win_rate,
            'podium_rate': podium_rate,
            'avg_position': avg_position,
            'dnf_rate': dnf_rate,
            'points_per_race': points_per_race
        })
    
    return pd.DataFrame(features_2024)

try:
    # Preparar dados de treinamento
    df_training = prepare_championship_data()
    
    # Preparar features (X) e target (y)
    feature_columns = ['win_rate', 'podium_rate', 'avg_position', 'dnf_rate', 
                      'points_per_race', 'total_races']
    
    X = df_training[feature_columns].values
    y = df_training['is_champion'].values
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
    
    print(f"\n� Dados preparados:")
    print(f"   • Treino: {len(X_train)} amostras")
    print(f"   • Validação: {len(X_val)} amostras")
    print(f"   • Features: {len(feature_columns)}")
    
    # EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES DE MLP
    print("\n🧪 EXECUTANDO EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES DE MLP")
    print("="*70)
    
    # Configurações para testar
    configurations = [
        {"hidden_layers": 1, "neurons": 16, "learning_rate": 0.001, "name": "MLP 1 camada (16 neurônios)"},
        {"hidden_layers": 1, "neurons": 32, "learning_rate": 0.001, "name": "MLP 1 camada (32 neurônios)"},
        {"hidden_layers": 1, "neurons": 64, "learning_rate": 0.001, "name": "MLP 1 camada (64 neurônios)"},
        {"hidden_layers": 2, "neurons": [32, 16], "learning_rate": 0.001, "name": "MLP 2 camadas (32-16)"},
        {"hidden_layers": 2, "neurons": [64, 32], "learning_rate": 0.001, "name": "MLP 2 camadas (64-32)"},
        {"hidden_layers": 3, "neurons": [64, 32, 16], "learning_rate": 0.001, "name": "MLP 3 camadas (64-32-16)"},
        {"hidden_layers": 2, "neurons": [32, 16], "learning_rate": 0.01, "name": "MLP 2 camadas LR=0.01"},
        {"hidden_layers": 2, "neurons": [32, 16], "learning_rate": 0.0001, "name": "MLP 2 camadas LR=0.0001"},
    ]
    
    results_experiments = []
    
    for i, config in enumerate(configurations):
        print(f"\n🔬 Experimento {i+1}/8: {config['name']}")
        print("-" * 50)
        
        # Criar modelo
        model = create_mlp_model(
            input_dim=X_train.shape[1],
            hidden_layers=config["hidden_layers"],
            neurons_per_layer=config["neurons"],
            learning_rate=config["learning_rate"]
        )
        
        print(f"   Arquitetura: {config['hidden_layers']} camadas ocultas")
        print(f"   Neurônios: {config['neurons']}")
        print(f"   Taxa de aprendizado: {config['learning_rate']}")
        
        # Treinar modelo
        history = model.fit(X_train, y_train, 
                          validation_data=(X_val, y_val),
                          epochs=100, 
                          batch_size=32, 
                          verbose=0,
                          class_weight={0: 1, 1: 10})  # Dar mais peso à classe minoritária
        
        # Avaliar modelo
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
        
        # Predições
        y_pred_train = (model.predict(X_train, verbose=0) > 0.5).astype(int)
        y_pred_val = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        
        print(f"   ✅ Acurácia Treino: {train_acc:.4f}")
        print(f"   ✅ Acurácia Validação: {val_acc:.4f}")
        
        # Guardar resultados
        results_experiments.append({
            'name': config['name'],
            'hidden_layers': config['hidden_layers'],
            'neurons': config['neurons'],
            'learning_rate': config['learning_rate'],
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'model': model
        })
    
    # Mostrar resumo dos experimentos
    print("\n📊 RESUMO DOS EXPERIMENTOS")
    print("="*70)
    
    results_df = pd.DataFrame(results_experiments)
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    
    for idx, row in results_df.iterrows():
        print(f"{row['name']:30} | Val Acc: {row['val_accuracy']:.4f} | Train Acc: {row['train_accuracy']:.4f}")
    
    # Melhor modelo
    best_model_config = results_df.iloc[0]
    best_model = best_model_config['model']
    
    print(f"\n🏆 MELHOR CONFIGURAÇÃO: {best_model_config['name']}")
    print(f"   Acurácia de Validação: {best_model_config['val_accuracy']:.4f}")
    
    # PREVISÃO PARA 2025
    print("\n🔮 PREVISÃO PARA CAMPEONATO 2025")
    print("="*50)
    
    # Preparar dados de 2024
    df_2024 = prepare_2024_data()
    X_2024 = df_2024[feature_columns].values
    X_2024_scaled = scaler.transform(X_2024)
    
    # Fazer previsões com o melhor modelo
    predictions_2025 = best_model.predict(X_2024_scaled, verbose=0)
    
    # Criar ranking
    df_2024['championship_probability'] = predictions_2025.flatten()
    df_2024 = df_2024.sort_values('championship_probability', ascending=False)
    
    print("\n🏁 RANKING DE CANDIDATOS PARA 2025 (baseado na MLP):")
    for idx, driver in df_2024.head(10).iterrows():
        prob_percent = driver['championship_probability'] * 100
        print(f"🏆 {driver['driver_name']:25} | {prob_percent:5.1f}% de chance")
    
    # Análise detalhada do top 3
    print("\n� ANÁLISE DETALHADA DOS TOP 3:")
    print("="*45)
    
    for i, (idx, driver) in enumerate(df_2024.head(3).iterrows()):
        print(f"\n{i+1}º {driver['driver_name']}")
        print(f"   🎯 Probabilidade de título: {driver['championship_probability']*100:.1f}%")
        print(f"   🏁 Performance 2024:")
        print(f"      • {driver['wins']} vitórias em {driver['total_races']} corridas")
        print(f"      • {driver['podiums']} pódios (taxa: {driver['podium_rate']*100:.1f}%)")
        print(f"      • {driver['points_total']} pontos totais")
        print(f"      • Posição média: {driver['avg_position']:.1f}")
        print(f"      • Taxa de abandono: {driver['dnf_rate']*100:.1f}%")
    
    favorite = df_2024.iloc[0]
    print(f"\n🎲 FAVORITO SEGUNDO A MLP: {favorite['driver_name']}")
    print(f"💪 {favorite['championship_probability']*100:.1f}% de chance de ser campeão!")
    
    print("\n" + "="*70)
    print("📋 INFORMAÇÕES TÉCNICAS DA MLP:")
    print(f"• Melhor arquitetura: {best_model_config['hidden_layers']} camadas ocultas")
    print(f"• Neurônios por camada: {best_model_config['neurons']}")
    print(f"• Taxa de aprendizado: {best_model_config['learning_rate']}")
    print(f"• Acurácia de validação: {best_model_config['val_accuracy']:.4f}")
    print(f"• Algoritmo: Backpropagation com Adam optimizer")
    print(f"• Função de ativação: ReLU (camadas ocultas), Sigmoid (saída)")
    print("="*70)

except Exception as e:
    print(f"❌ Erro durante a análise: {e}")
    import traceback
    traceback.print_exc()
