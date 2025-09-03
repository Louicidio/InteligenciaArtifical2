
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

print("=== MLP PARA CLASSIFICACAO DE COMBUSTIVEL DE CARROS ===")
print("="*60)

def load_car_dataset():
    """Carrega o dataset real de carros"""
    try:
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv("Cars Datasets 2025.csv", encoding=encoding)
                print(f"Dataset carregado com encoding {encoding}")
                break
            except:
                continue
        
        if df is None:
            raise Exception("Erro ao carregar CSV")
        
        print(f"Dataset original: {len(df)} carros")
        print(f"Combustiveis encontrados: {df['Fuel Types'].value_counts()}")
        
        # Preprocessar dados
        def clean_numeric(value, default=0):
            if pd.isna(value):
                return default
            value_str = str(value).replace('$', '').replace(',', '')
            value_str = value_str.replace(' cc', '').replace(' hp', '')
            value_str = value_str.replace(' km/h', '').replace(' sec', '')
            value_str = value_str.replace(' Nm', '').replace('L', '')
            
            if '-' in value_str:
                try:
                    parts = value_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    pass
            
            try:
                return float(value_str)
            except:
                return default
        
        # Extrair features
        df_clean = df.copy()
        df_clean['engine_capacity'] = df['CC/Battery Capacity'].apply(clean_numeric)
        df_clean['horsepower'] = df['HorsePower'].apply(clean_numeric)
        df_clean['max_speed'] = df['Total Speed'].apply(clean_numeric)
        df_clean['acceleration'] = df['Performance(0 - 100 )KM/H'].apply(clean_numeric)
        df_clean['price'] = df['Cars Prices'].apply(clean_numeric)
        df_clean['torque'] = df['Torque'].apply(clean_numeric)
        df_clean['seats'] = df['Seats'].apply(lambda x: clean_numeric(x, 4))
        
        # Limpar combustiveis
        fuel_map = {
            'petrol': 'Petrol',
            'gasoline': 'Petrol',
            'diesel': 'Diesel', 
            'hybrid': 'Hybrid',
            'plug in hyrbrid': 'Hybrid',
            'electric': 'Electric'
        }
        
        df_clean['fuel_clean'] = df['Fuel Types'].str.lower().str.strip()
        df_clean['fuel_mapped'] = df_clean['fuel_clean'].map(fuel_map)
        
        # Filtrar dados validos
        df_final = df_clean.dropna(subset=['fuel_mapped'])
        df_final = df_final[
            (df_final['horsepower'] > 0) & 
            (df_final['engine_capacity'] > 0) &
            (df_final['max_speed'] > 0)
        ]
        
        # Manter apenas combustiveis com dados suficientes
        fuel_counts = df_final['fuel_mapped'].value_counts()
        valid_fuels = fuel_counts[fuel_counts >= 15].index
        df_final = df_final[df_final['fuel_mapped'].isin(valid_fuels)]
        
        features = ['engine_capacity', 'horsepower', 'max_speed', 'acceleration', 'price', 'torque', 'seats']
        
        print(f"Dataset processado: {len(df_final)} carros")
        print(f"Combustiveis finais: {df_final['fuel_mapped'].value_counts()}")
        
        return df_final, features
        
    except Exception as e:
        print(f"Erro: {e}")
        print("Usando dados simulados...")
        
        # Fallback: dados simulados
        np.random.seed(42)
        data = []
        fuels = ['Petrol', 'Diesel', 'Hybrid']
        
        for i in range(200):
            fuel = np.random.choice(fuels)
            if fuel == 'Petrol':
                cap = np.random.normal(2500, 500)
                hp = np.random.normal(200, 40)
                speed = np.random.normal(220, 25)
                acc = np.random.normal(6.5, 1.0)
                price = np.random.normal(50000, 15000)
                torque = np.random.normal(300, 80)
            elif fuel == 'Diesel':
                cap = np.random.normal(2200, 400)
                hp = np.random.normal(150, 25)
                speed = np.random.normal(200, 20)
                acc = np.random.normal(8.0, 1.5)
                price = np.random.normal(45000, 12000)
                torque = np.random.normal(400, 100)
            else:  # Hybrid
                cap = np.random.normal(1800, 300)
                hp = np.random.normal(120, 20)
                speed = np.random.normal(180, 15)
                acc = np.random.normal(9.0, 1.0)
                price = np.random.normal(60000, 18000)
                torque = np.random.normal(250, 60)
            
            data.append({
                'engine_capacity': max(1000, cap),
                'horsepower': max(80, hp),
                'max_speed': max(150, speed),
                'acceleration': max(3, acc),
                'price': max(20000, price),
                'torque': max(150, torque),
                'seats': 4,
                'fuel_mapped': fuel
            })
        
        df_sim = pd.DataFrame(data)
        features = ['engine_capacity', 'horsepower', 'max_speed', 'acceleration', 'price', 'torque', 'seats']
        return df_sim, features

class SimpleMLP:
    """MLP simples com backpropagation"""
    
    def __init__(self, input_size, hidden_sizes, output_size, lr=0.01):
        self.lr = lr
        self.layers = []
        
        # Criar camadas
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layer = {
                'W': np.random.randn(sizes[i], sizes[i+1]) * 0.1,
                'b': np.zeros((1, sizes[i+1])),
                'a': None,
                'z': None
            }
            self.layers.append(layer)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        current = X
        for i, layer in enumerate(self.layers):
            layer['z'] = np.dot(current, layer['W']) + layer['b']
            if i == len(self.layers) - 1:  # Output layer
                layer['a'] = self.softmax(layer['z'])
            else:  # Hidden layers
                layer['a'] = self.relu(layer['z'])
            current = layer['a']
        return self.layers[-1]['a']
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        error = y_pred - y_true
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if i == 0:
                prev_a = X
            else:
                prev_a = self.layers[i-1]['a']
            
            # Gradients
            dW = np.dot(prev_a.T, error) / m
            db = np.sum(error, axis=0, keepdims=True) / m
            
            # Update weights
            layer['W'] -= self.lr * dW
            layer['b'] -= self.lr * db
            
            # Error for next layer
            if i > 0:
                error = np.dot(error, layer['W'].T)
                error *= self.relu_derivative(self.layers[i-1]['a'])
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            
            if (epoch + 1) % 25 == 0:
                loss = -np.mean(np.sum(y * np.log(y_pred + 1e-15), axis=1))
                acc = accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
                print(f"Epoca {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    def predict(self, X):
        return self.forward(X)
    
    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=1)

def one_hot_encode(y, num_classes):
    """One-hot encoding manual"""
    encoded = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        encoded[i, label] = 1
    return encoded

# === EXECUCAO PRINCIPAL ===
try:
    
    df, feature_cols = load_car_dataset()

    
    X = df[feature_cols].values
    y = df['fuel_mapped'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_onehot = one_hot_encode(y_encoded, num_classes)
    
    # normaliza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split( #alteração para testes e treino
        X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
    )
    
    print(f"Treino: {len(X_train)}, Teste: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {list(le.classes_)}")
    
    # configuraçãodo mlp
    print(f"\n=== EXPERIMENTOS MLP ===")
    
    configs = [
        {"hidden": [10], "lr": 0.01, "name": "MLP 1 camada (10)"},
        {"hidden": [16, 8], "lr": 0.01, "name": "MLP 2 camadas (16-8)"},
        {"hidden": [20], "lr": 0.005, "name": "MLP 1 camada (20)"},
        {"hidden": [32, 16], "lr": 0.001, "name": "MLP 2 camadas (32-16)"},
        {"hidden": [24, 12, 6], "lr": 0.001, "name": "MLP 3 camadas (24-12-6)"},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\nExperimento {i+1}: {config['name']}")
        print("-" * 40)
        
        model = SimpleMLP(
            input_size=X_train.shape[1],
            hidden_sizes=config['hidden'],
            output_size=num_classes,
            lr=config['lr']
        )
        
        print(f"Arquitetura: {config['hidden']}")
        print(f"Learning rate: {config['lr']}")
        
        model.fit(X_train, y_train, epochs=100)
        
        train_pred = model.predict_classes(X_train)
        test_pred = model.predict_classes(X_test)
        
        train_acc = accuracy_score(np.argmax(y_train, axis=1), train_pred)
        test_acc = accuracy_score(np.argmax(y_test, axis=1), test_pred)
        
        print(f"Resultado: Train={train_acc:.4f}, Test={test_acc:.4f}")
        
        results.append({
            'name': config['name'],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': model
        })
    
    # resultados do mlp
    print(f"\n=== RESUMO DOS RESULTADOS ===")
    results_df = pd.DataFrame(results).sort_values('test_acc', ascending=False)
    
    print(f"{'Configuracao':30} | {'Treino':8} | {'Teste':8}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['name']:30} | {row['train_acc']:6.4f} | {row['test_acc']:6.4f}")
    
    # acha o melhor modelo
    best = results_df.iloc[0]
    print(f"\nMELHOR MODELO: {best['name']}")
    print(f"Acuracia de teste: {best['test_acc']:.4f}")
    

    best_model = best['model']
    test_pred = best_model.predict_classes(X_test)
    test_true = np.argmax(y_test, axis=1)
    
    print(f"\nRELATORIO DE CLASSIFICACAO:")
    print(classification_report(test_true, test_pred, target_names=le.classes_, zero_division=0))
    
    
    print(f"=== ANALISE CONCLUIDA ===")
    print(f"Melhor accuracy: {best['test_acc']*100:.2f}%")
    
    print(f"\n=== TESTE COM CARRO PERSONALIZADO ===")
    
    carro_personalizado = {
        'engine_capacity': 2000,   
        'horsepower': 180,          
        'max_speed': 210,          
        'acceleration': 7.5,        
        'price': 45000,             
        'torque': 320,              
        'seats': 5                  
    }
    
    carro_features = np.array([[
        carro_personalizado['engine_capacity'],
        carro_personalizado['horsepower'],
        carro_personalizado['max_speed'],
        carro_personalizado['acceleration'],
        carro_personalizado['price'],
        carro_personalizado['torque'],
        carro_personalizado['seats']
    ]])
    
    carro_normalizado = scaler.transform(carro_features)
    
    pred_probs = best_model.predict(carro_normalizado)
    pred_class = best_model.predict_classes(carro_normalizado)[0]
    pred_label = le.classes_[pred_class]
    confidence = pred_probs[0][pred_class] * 100
    
    print(f"ESPECIFICAÇÕES DO CARRO:")
    print(f"  Motor: {carro_personalizado['engine_capacity']}cc")
    print(f"  Potência: {carro_personalizado['horsepower']}HP")
    print(f"  Velocidade máxima: {carro_personalizado['max_speed']}km/h")
    print(f"  Aceleração 0-100: {carro_personalizado['acceleration']}s")
    print(f"  Preço: ${carro_personalizado['price']:,}")
    print(f"  Torque: {carro_personalizado['torque']}Nm")
    print(f"  Assentos: {carro_personalizado['seats']}")
    print(f"\nPREDIÇÃO: {pred_label} ({confidence:.1f}% de confiança)")
    

except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()