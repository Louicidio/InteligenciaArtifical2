import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

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
    n_samples = 1000
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

class SimpleMLPClassifier:
    """
    Implementação simples de MLP com Backpropagation
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []
        
        # Definir arquitetura
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Inicializar pesos e bias
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1,
                'bias': np.zeros((1, layer_sizes[i + 1])),
                'activation': None,
                'z': None
            }
            self.layers.append(layer)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        current_input = X
        
        for i, layer in enumerate(self.layers):
            # Cálculo linear
            layer['z'] = np.dot(current_input, layer['weights']) + layer['bias']
            
            # Aplicar função de ativação
            if i == len(self.layers) - 1:  # Última camada (saída)
                layer['activation'] = self.softmax(layer['z'])
            else:  # Camadas ocultas
                layer['activation'] = self.relu(layer['z'])
            
            current_input = layer['activation']
        
        return self.layers[-1]['activation']
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        # Calcular erro da camada de saída
        error = y_pred - y_true
        
        # Backpropagation
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if i == 0:
                prev_activation = X
            else:
                prev_activation = self.layers[i-1]['activation']
            
            # Calcular gradientes
            dW = np.dot(prev_activation.T, error) / m
            db = np.sum(error, axis=0, keepdims=True) / m
            
            # Atualizar pesos
            layer['weights'] -= self.learning_rate * dW
            layer['bias'] -= self.learning_rate * db
            
            # Calcular erro para a próxima camada
            if i > 0:
                error = np.dot(error, layer['weights'].T)
                if i > 0:  # Não é a camada de saída
                    error *= self.relu_derivative(self.layers[i-1]['activation'])
    
    def fit(self, X, y, epochs=100, verbose=True):
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Calcular perda (cross-entropy)
            loss = -np.mean(np.sum(y * np.log(y_pred + 1e-15), axis=1))
            losses.append(loss)
            
            # Calcular acurácia
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy = accuracy_score(true_classes, pred_classes)
            accuracies.append(accuracy)
            
            # Backward pass
            self.backward(X, y, y_pred)
            
            # Imprimir progresso
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Época {epoch + 1}/{epochs} - Perda: {loss:.4f} - Acurácia: {accuracy:.4f}")
        
        return losses, accuracies
    
    def predict(self, X):
        return self.forward(X)
    
    def predict_classes(self, X):
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)

def one_hot_encode(y, num_classes):
    """Converter labels para one-hot encoding"""
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot

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
    
    # 2. Preparar dados
    print("\n🔧 Preparando dados para treinamento...")
    
    # Separação da classe alvo e atributos
    X = df.iloc[:, :-1].values  # Todas as colunas exceto a última
    y = df.iloc[:, -1].values   # Última coluna (combustível)
    
    # Transformação das classes para valores numéricos
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # One-hot encoding manual
    num_classes = len(le.classes_)
    y_one_hot = one_hot_encode(y_encoded, num_classes)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )
    
    print(f"✅ Dados preparados:")
    print(f"   • Treino: {len(X_train)} amostras")
    print(f"   • Teste: {len(X_test)} amostras")
    print(f"   • Features: {X.shape[1]}")
    print(f"   • Classes: {num_classes}")
    
    # 3. EXPERIMENTOS COM DIFERENTES CONFIGURAÇÕES
    print("\n🧪 EXECUTANDO EXPERIMENTOS COM MLP")
    print("="*50)
    
    configurations = [
        {"hidden": [10], "lr": 0.01, "name": "MLP 1 camada (10 neurônios)"},
        {"hidden": [10, 8], "lr": 0.01, "name": "MLP 2 camadas (10-8)"},
        {"hidden": [16], "lr": 0.001, "name": "MLP 1 camada (16 neurônios)"},
        {"hidden": [32, 16], "lr": 0.001, "name": "MLP 2 camadas (32-16)"},
        {"hidden": [64, 32, 16], "lr": 0.001, "name": "MLP 3 camadas (64-32-16)"},
    ]
    
    results = []
    
    for i, config in enumerate(configurations):
        print(f"\n🔬 Experimento {i+1}/{len(configurations)}: {config['name']}")
        print("-" * 40)
        
        # Criar e treinar modelo
        model = SimpleMLPClassifier(
            input_size=X_train.shape[1],
            hidden_sizes=config["hidden"],
            output_size=num_classes,
            learning_rate=config["lr"]
        )
        
        print(f"   Arquitetura: {len(config['hidden'])} camadas ocultas")
        print(f"   Neurônios: {config['hidden']}")
        print(f"   Taxa de aprendizado: {config['lr']}")
        
        # Treinar
        losses, accuracies = model.fit(X_train, y_train, epochs=100, verbose=False)
        
        # Avaliar
        train_pred = model.predict_classes(X_train)
        test_pred = model.predict_classes(X_test)
        
        train_acc = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        test_acc = accuracy_score(np.argmax(y_test, axis=1), test_pred)
        
        print(f"   ✅ Acurácia Treino: {train_acc:.4f}")
        print(f"   ✅ Acurácia Teste: {test_acc:.4f}")
        
        results.append({
            'name': config['name'],
            'hidden_layers': len(config['hidden']),
            'neurons': config['hidden'],
            'learning_rate': config['lr'],
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'model': model
        })
    
    # 4. RESULTADOS
    print("\n📊 RESUMO DOS EXPERIMENTOS")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    
    print(f"{'Configuração':35} | {'Teste Acc':10} | {'Treino Acc':11}")
    print("-" * 60)
    for idx, row in results_df.iterrows():
        print(f"{row['name']:35} | {row['test_accuracy']:8.4f} | {row['train_accuracy']:9.4f}")
    
    # 5. MELHOR MODELO
    best_model = results_df.iloc[0]
    
    print(f"\n🏆 MELHOR CONFIGURAÇÃO: {best_model['name']}")
    print(f"   Acurácia de Teste: {best_model['test_accuracy']:.4f}")
    
    # 6. CLASSIFICAÇÃO DETALHADA
    print("\n🔍 RELATÓRIO DE CLASSIFICAÇÃO (MELHOR MODELO)")
    print("="*50)
    
    best_clf = best_model['model']
    test_pred = best_clf.predict_classes(X_test)
    test_true = np.argmax(y_test, axis=1)
    
    print(classification_report(test_true, test_pred, target_names=le.classes_, digits=4))
    
    # 7. EXEMPLOS DE PREDIÇÃO
    print("\n🎯 EXEMPLOS DE PREDIÇÕES:")
    print("-" * 40)
    
    test_probs = best_clf.predict(X_test)
    
    for i in range(min(10, len(X_test))):
        true_label = le.classes_[test_true[i]]
        pred_label = le.classes_[test_pred[i]]
        confidence = test_probs[i][test_pred[i]] * 100
        
        # Dados originais
        original_data = scaler.inverse_transform(X_test[i:i+1])[0]
        
        print(f"Carro {i+1}:")
        print(f"  Cilindrada: {original_data[0]:.1f}L, Potência: {original_data[1]:.0f}HP")
        print(f"  Consumo: {original_data[2]:.1f}/{original_data[3]:.1f} km/l")
        print(f"  Peso: {original_data[4]:.0f}kg, Preço: R${original_data[5]:.0f}")
        print(f"  Real: {true_label} | Predito: {pred_label} | Confiança: {confidence:.1f}%")
        print()
    
    print(f"\n✅ ANÁLISE COMPLETA FINALIZADA!")
    print(f"🎯 Melhor acurácia obtida: {best_model['test_accuracy']*100:.2f}%")
    
    print("\n" + "="*60)
    print("📋 INFORMAÇÕES TÉCNICAS:")
    print(f"• Dataset: {len(df)} carros com {X.shape[1]} características")
    print(f"• Classes: {', '.join(le.classes_)}")
    print(f"• Algoritmo: MLP com Backpropagation implementado manualmente")
    print(f"• Função de ativação: ReLU (ocultas), Softmax (saída)")
    print(f"• Função de perda: Categorical Cross-entropy")
    print("="*60)

except Exception as e:
    print(f"❌ Erro durante a execução: {e}")
    import traceback
    traceback.print_exc()
