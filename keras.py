import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
import seaborn as sns

print("🚗 MLP PARA CLASSIFICAÇÃO DE COMBUSTÍVEL DE CARROS 🛠️")
print("="*65)

def create_car_dataset():
    """
    Cria um dataset simulado de carros com diferentes características
    para classificação do tipo de combustível
    """
    np.random.seed(42)
    
    # Definir tipos de combustível
    fuel_types = ['Gasolina', 'Diesel', 'Híbrido', 'Elétrico', 'Flex']
    
    # Gerar dados simulados
    n_samples = 2000
    data = []
    
    for i in range(n_samples):
        # Escolher tipo de combustível aleatoriamente
        fuel_type = np.random.choice(fuel_types)
        
        # Características baseadas no tipo de combustível
        if fuel_type == 'Gasolina':
            cilindrada = np.random.normal(1.6, 0.4)  # Motor menor
            potencia = np.random.normal(120, 30)     # Potência média
            consumo_cidade = np.random.normal(8.5, 1.5)  # km/l na cidade
            consumo_estrada = np.random.normal(12, 2)    # km/l na estrada
            peso = np.random.normal(1200, 200)       # kg
            preco = np.random.normal(45000, 10000)   # R$
            
        elif fuel_type == 'Diesel':
            cilindrada = np.random.normal(2.2, 0.5)  # Motor maior
            potencia = np.random.normal(140, 25)     # Mais potente
            consumo_cidade = np.random.normal(11, 2) # Mais econômico
            consumo_estrada = np.random.normal(16, 2.5)
            peso = np.random.normal(1400, 250)       # Mais pesado
            preco = np.random.normal(55000, 12000)
            
        elif fuel_type == 'Híbrido':
            cilindrada = np.random.normal(1.8, 0.3)  # Motor médio
            potencia = np.random.normal(100, 20)     # Potência combinada
            consumo_cidade = np.random.normal(18, 3) # Muito econômico
            consumo_estrada = np.random.normal(20, 3)
            peso = np.random.normal(1350, 180)       # Peso médio-alto (bateria)
            preco = np.random.normal(70000, 15000)   # Mais caro
            
        elif fuel_type == 'Elétrico':
            cilindrada = 0  # Sem motor a combustão
            potencia = np.random.normal(150, 40)     # Alta potência instantânea
            consumo_cidade = np.random.normal(25, 5) # km/kWh equivalente
            consumo_estrada = np.random.normal(22, 4)
            peso = np.random.normal(1500, 200)       # Pesado (bateria)
            preco = np.random.normal(85000, 20000)   # Mais caro
            
        else:  # Flex
            cilindrada = np.random.normal(1.4, 0.3)  # Motor flexível
            potencia = np.random.normal(110, 25)     # Potência média
            consumo_cidade = np.random.normal(7.5, 1.5)  # Menos econômico
            consumo_estrada = np.random.normal(11, 2)
            peso = np.random.normal(1150, 150)       # Mais leve
            preco = np.random.normal(40000, 8000)    # Mais barato
        
        # Garantir valores positivos
        cilindrada = max(0, cilindrada)
        potencia = max(50, potencia)
        consumo_cidade = max(3, consumo_cidade)
        consumo_estrada = max(5, consumo_estrada)
        peso = max(800, peso)
        preco = max(20000, preco)
        
        data.append({
            'cilindrada': round(cilindrada, 1),
            'potencia': round(potencia, 0),
            'consumo_cidade': round(consumo_cidade, 1),
            'consumo_estrada': round(consumo_estrada, 1),
            'peso': round(peso, 0),
            'preco': round(preco, 0),
            'combustivel': fuel_type
        })
    
    return pd.DataFrame(data)

def create_mlp_model(input_dim, hidden_layers, neurons_per_layer, learning_rate, 
                     num_classes, optimizer_type='adam', dropout_rate=0.2):
    """
    Cria um modelo MLP personalizado para classificação
    
    Args:
        input_dim: Dimensão de entrada
        hidden_layers: Número de camadas ocultas
        neurons_per_layer: Lista com neurônios por camada ou int único
        learning_rate: Taxa de aprendizado
        num_classes: Número de classes de saída
        optimizer_type: Tipo de otimizador ('adam' ou 'sgd')
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
    
    # Camada de saída
    model.add(Dense(num_classes, activation='softmax'))
    
    # Escolher otimizador
    if optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)
    
    # Compilar modelo
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def plot_training_history(history, title="Treinamento da MLP"):
    """
    Plotar histórico de treinamento
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 4))
    
    # Plot de perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Perda (treino)")
    plt.plot(history.history["val_loss"], label="Perda (validação)")
    plt.title("Perda durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()
    
    # Plot de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Acurácia (treino)")
    plt.plot(history.history["val_accuracy"], label="Acurácia (validação)")
    plt.title("Acurácia durante o Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plotar matriz de confusão
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predição')
    plt.ylabel('Valor Real')
    plt.show()

# EXECUÇÃO PRINCIPAL
try:
    # 1. Criar e carregar dataset
    print("\n📊 Criando dataset de carros...")
    df = create_car_dataset()
    
    print(f"✅ Dataset criado com {len(df)} carros")
    print(f"📈 Distribuição de combustíveis:")
    print(df['combustivel'].value_counts())
    print(f"\n📋 Primeiras 5 linhas do dataset:")
    print(df.head())
    
    # 2. Preparar dados para MLP
    print("\n🔧 Preparando dados para treinamento...")
    
    # Separação da classe alvo e atributos
    X = df.iloc[:, :-1].values  # Todas as colunas exceto a última
    y = df.iloc[:, -1].values   # Última coluna (combustível)
    
    # Transformação das classes para valores numéricos
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)   # transforma em [0,1,2,3,4]
    
    # One-hot encoding (necessário para softmax)
    y_categorical = to_categorical(y_encoded)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    
    print(f"✅ Dados preparados:")
    print(f"   • Treino: {len(X_train)} amostras")
    print(f"   • Teste: {len(X_test)} amostras")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Classes: {y_categorical.shape[1]}")
    
    # 3. EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES DE MLP
    print("\n🧪 EXECUTANDO EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES DE MLP")
    print("="*70)
    
    # Configurações para testar
    configurations = [
        {"hidden_layers": 1, "neurons": 10, "learning_rate": 0.01, "optimizer": "sgd", "name": "MLP 1 camada (10 neurônios) - SGD"},
        {"hidden_layers": 2, "neurons": [10, 8], "learning_rate": 0.01, "optimizer": "sgd", "name": "MLP 2 camadas (10-8) - SGD"},
        {"hidden_layers": 1, "neurons": 16, "learning_rate": 0.001, "optimizer": "adam", "name": "MLP 1 camada (16 neurônios) - Adam"},
        {"hidden_layers": 2, "neurons": [32, 16], "learning_rate": 0.001, "optimizer": "adam", "name": "MLP 2 camadas (32-16) - Adam"},
        {"hidden_layers": 3, "neurons": [64, 32, 16], "learning_rate": 0.001, "optimizer": "adam", "name": "MLP 3 camadas (64-32-16) - Adam"},
        {"hidden_layers": 2, "neurons": [16, 8], "learning_rate": 0.01, "optimizer": "adam", "name": "MLP 2 camadas LR=0.01 - Adam"},
        {"hidden_layers": 2, "neurons": [16, 8], "learning_rate": 0.0001, "optimizer": "adam", "name": "MLP 2 camadas LR=0.0001 - Adam"},
    ]
    
    results_experiments = []
    
    for i, config in enumerate(configurations):
        print(f"\n🔬 Experimento {i+1}/{len(configurations)}: {config['name']}")
        print("-" * 50)
        
        # Criar modelo
        model = create_mlp_model(
            input_dim=X_train.shape[1],
            hidden_layers=config["hidden_layers"],
            neurons_per_layer=config["neurons"],
            learning_rate=config["learning_rate"],
            num_classes=y_categorical.shape[1],
            optimizer_type=config["optimizer"]
        )
        
        print(f"   Arquitetura: {config['hidden_layers']} camadas ocultas")
        print(f"   Neurônios: {config['neurons']}")
        print(f"   Taxa de aprendizado: {config['learning_rate']}")
        print(f"   Otimizador: {config['optimizer'].upper()}")
        
        # Treinar modelo
        history = model.fit(X_train, y_train, 
                          validation_data=(X_test, y_test),
                          epochs=100, 
                          batch_size=16, 
                          verbose=0)
        
        # Avaliar modelo
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"   ✅ Acurácia Treino: {train_acc:.4f}")
        print(f"   ✅ Acurácia Teste: {test_acc:.4f}")
        print(f"   📉 Perda Teste: {test_loss:.4f}")
        
        # Guardar resultados
        results_experiments.append({
            'name': config['name'],
            'hidden_layers': config['hidden_layers'],
            'neurons': config['neurons'],
            'learning_rate': config['learning_rate'],
            'optimizer': config['optimizer'],
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'model': model,
            'history': history
        })
    
    # 4. ANÁLISE DOS RESULTADOS
    print("\n📊 RESUMO DOS EXPERIMENTOS")
    print("="*80)
    
    results_df = pd.DataFrame(results_experiments)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    
    print(f"{'Configuração':45} | {'Teste Acc':10} | {'Treino Acc':11} | {'Perda':8}")
    print("-" * 80)
    for idx, row in results_df.iterrows():
        print(f"{row['name']:45} | {row['test_accuracy']:8.4f} | {row['train_accuracy']:9.4f} | {row['test_loss']:6.4f}")
    
    # 5. MELHOR MODELO - ANÁLISE DETALHADA
    best_model_config = results_df.iloc[0]
    best_model = best_model_config['model']
    best_history = best_model_config['history']
    
    print(f"\n🏆 MELHOR CONFIGURAÇÃO: {best_model_config['name']}")
    print(f"   Acurácia de Teste: {best_model_config['test_accuracy']:.4f}")
    print(f"   Acurácia de Treino: {best_model_config['train_accuracy']:.4f}")
    print(f"   Perda de Teste: {best_model_config['test_loss']:.4f}")
    
    # 6. PREVISÕES E RELATÓRIO DETALHADO
    print("\n🔍 ANÁLISE DETALHADA DO MELHOR MODELO")
    print("="*50)
    
    # Fazer previsões
    predictions = best_model.predict(X_test, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Nomes das classes
    class_names = le.classes_
    
    print("\n[INFO] Relatório de Classificação:\n")
    print(classification_report(true_classes, pred_classes, target_names=class_names, digits=4))
    
    # Mostrar alguns exemplos de predição
    print("\n🎯 EXEMPLOS DE PREDIÇÕES:")
    print("-" * 40)
    for i in range(min(10, len(X_test))):
        true_label = class_names[true_classes[i]]
        pred_label = class_names[pred_classes[i]]
        confidence = predictions[i][pred_classes[i]] * 100
        
        # Dados originais (desnormalizar para mostrar)
        original_data = scaler.inverse_transform(X_test[i:i+1])[0]
        
        print(f"Carro {i+1}:")
        print(f"  Cilindrada: {original_data[0]:.1f}L, Potência: {original_data[1]:.0f}HP")
        print(f"  Consumo: {original_data[2]:.1f}/{original_data[3]:.1f} km/l")
        print(f"  Real: {true_label} | Predito: {pred_label} | Confiança: {confidence:.1f}%")
        print()
    
    # 7. PLOTAR GRÁFICOS
    print("\n📈 Gerando gráficos de análise...")
    
    # Gráfico do treinamento do melhor modelo
    plot_training_history(best_history, f"Melhor Modelo: {best_model_config['name']}")
    
    # Matriz de confusão
    plot_confusion_matrix(true_classes, pred_classes, class_names)
    
    print(f"\n✅ ANÁLISE COMPLETA FINALIZADA!")
    print(f"🎯 Melhor acurácia obtida: {best_model_config['test_accuracy']*100:.2f}%")
    print(f"🔧 Melhor configuração: {best_model_config['name']}")
    
    print("\n" + "="*70)
    print("📋 INFORMAÇÕES TÉCNICAS:")
    print(f"• Dataset: {len(df)} carros com {X.shape[1]} características")
    print(f"• Classes: {', '.join(class_names)}")
    print(f"• Algoritmo: MLP com Backpropagation")
    print(f"• Função de ativação: ReLU (ocultas), Softmax (saída)")
    print(f"• Função de perda: Categorical Crossentropy")
    print("="*70)

except Exception as e:
    print(f"❌ Erro durante a execução: {e}")
    import traceback
    traceback.print_exc()
