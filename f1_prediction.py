import pandas as pd
import numpy as np

print("🏎️ ANÁLISE E PREVISÃO F1 2025 - CAMPEONATO DE PILOTOS 🏆")
print("="*60)

# Carregamento dos datasets
print("\n📊 Carregando dados...")
results = pd.read_csv("results.csv")
drivers = pd.read_csv("drivers.csv") 
races = pd.read_csv("races.csv")
driver_standings = pd.read_csv("driver_standings.csv")

print(f"✅ Dados carregados:")
print(f"   • {len(results)} resultados de corridas")
print(f"   • {len(drivers)} pilotos")
print(f"   • {len(races)} corridas")
print(f"   • {len(driver_standings)} registros de standings")

# Análise dos dados de 2024
print("\n🔍 Analisando temporada 2024...")

# Pegando corridas de 2024
races_2024 = races[races['year'] == 2024]
race_ids_2024 = races_2024['raceId'].tolist()

# Resultados de 2024
results_2024 = results[results['raceId'].isin(race_ids_2024)]

# Standings finais de 2024
standings_2024 = driver_standings[driver_standings['raceId'].isin(race_ids_2024)]

# Pegando a última corrida de 2024 para ter os standings finais
if not standings_2024.empty:
    last_race_2024 = max(race_ids_2024)
    final_standings_2024 = standings_2024[standings_2024['raceId'] == last_race_2024]
    
    print(f"\n🏆 TOP 10 CAMPEONATO 2024:")
    if not final_standings_2024.empty:
        final_standings_2024 = final_standings_2024.sort_values('points', ascending=False)
        
        for idx, row in final_standings_2024.head(10).iterrows():
            driver_info = drivers[drivers['driverId'] == row['driverId']]
            if not driver_info.empty:
                driver_name = f"{driver_info.iloc[0]['forename']} {driver_info.iloc[0]['surname']}"
                print(f"   {row['position']:2}º {driver_name:25} - {row['points']} pts")

# Análise detalhada dos top pilotos
print("\n📈 Análise detalhada dos candidatos para 2025...")

# Pilotos que correram em 2024
active_drivers_2024 = results_2024['driverId'].unique()

driver_analysis = []

for driver_id in active_drivers_2024:
    driver_info = drivers[drivers['driverId'] == driver_id]
    if driver_info.empty:
        continue
    
    driver_info = driver_info.iloc[0]
    driver_name = f"{driver_info['forename']} {driver_info['surname']}"
    
    # Resultados do piloto em 2024
    driver_results_2024 = results_2024[results_2024['driverId'] == driver_id]
    
    # Calcular estatísticas
    total_races = len(driver_results_2024)
    
    # Vitórias (position = '1')
    wins = len(driver_results_2024[driver_results_2024['position'] == '1'])
    
    # Pódios (positions 1, 2, 3)
    podiums = len(driver_results_2024[driver_results_2024['position'].isin(['1', '2', '3'])])
    
    # Pontos totais
    total_points = driver_results_2024['points'].sum()
    
    # Posição média (apenas para quem terminou)
    finished_positions = driver_results_2024[driver_results_2024['position'].str.isdigit()]
    if not finished_positions.empty:
        avg_position = finished_positions['position'].astype(int).mean()
    else:
        avg_position = 20
    
    # Taxa de DNF
    dnf_count = len(driver_results_2024[~driver_results_2024['position'].str.isdigit()])
    dnf_rate = dnf_count / total_races if total_races > 0 else 1
    
    # Performance score baseado em múltiplos fatores
    win_rate = wins / total_races if total_races > 0 else 0
    podium_rate = podiums / total_races if total_races > 0 else 0
    points_per_race = total_points / total_races if total_races > 0 else 0
    
    # Score geral (quanto maior, melhor)
    performance_score = (
        win_rate * 40 +           # Vitórias são muito importantes
        podium_rate * 25 +        # Pódios são importantes
        points_per_race * 2 +     # Pontos por corrida
        (1 - dnf_rate) * 15 +     # Confiabilidade
        (21 - avg_position) * 0.5 # Consistência (quanto menor a posição, melhor)
    )
    
    driver_analysis.append({
        'driver_id': driver_id,
        'name': driver_name,
        'nationality': driver_info['nationality'],
        'total_races': total_races,
        'wins': wins,
        'podiums': podiums,
        'total_points': total_points,
        'win_rate': win_rate,
        'podium_rate': podium_rate,
        'avg_position': avg_position,
        'dnf_rate': dnf_rate,
        'performance_score': performance_score
    })

# Ordenar por performance score
driver_analysis.sort(key=lambda x: x['performance_score'], reverse=True)

print("\n🏁 PREVISÃO PARA CAMPEONATO 2025:")
print("="*60)

# Calcular probabilidades baseadas no score
total_score = sum(d['performance_score'] for d in driver_analysis)

for i, driver in enumerate(driver_analysis[:10]):
    probability = (driver['performance_score'] / total_score) * 100 if total_score > 0 else 0
    
    print(f"{i+1:2}º {driver['name']:25} | {probability:5.1f}% chance")
    print(f"    📊 2024: {driver['total_points']:3.0f} pts | {driver['wins']} vitórias | {driver['podiums']} pódios")
    print(f"    📈 Avg pos: {driver['avg_position']:.1f} | DNF: {driver['dnf_rate']*100:.1f}%")
    print()

# Top 3 análise detalhada
print("🥇 TOP 3 CANDIDATOS PARA 2025:")
print("="*40)

for i in range(min(3, len(driver_analysis))):
    driver = driver_analysis[i]
    probability = (driver['performance_score'] / total_score) * 100 if total_score > 0 else 0
    
    print(f"\n{i+1}º {driver['name']} ({driver['nationality']})")
    print(f"   🎯 Chance de título: {probability:.1f}%")
    print(f"   🏁 Performance 2024:")
    print(f"      • {driver['total_points']} pontos totais")
    print(f"      • {driver['wins']} vitórias em {driver['total_races']} corridas")
    print(f"      • {driver['podiums']} pódios (taxa: {driver['podium_rate']*100:.1f}%)")
    print(f"      • Posição média: {driver['avg_position']:.1f}")
    print(f"      • Taxa de abandono: {driver['dnf_rate']*100:.1f}%")

if driver_analysis:
    favorite = driver_analysis[0]
    favorite_prob = (favorite['performance_score'] / total_score) * 100 if total_score > 0 else 0
    
    print(f"\n🎲 GRANDE FAVORITO: {favorite['name']}")
    print(f"💪 {favorite_prob:.1f}% de chance de conquistar o título mundial!")

print("\n" + "="*60)
print("📋 METODOLOGIA DA ANÁLISE:")
print("• Baseada inteiramente nos dados de performance 2024")
print("• Fatores: vitórias (40%), pódios (25%), confiabilidade (15%)")
print("• Pontos por corrida (2x peso) e consistência (0.5x peso)")
print("• Probabilidades calculadas proporcionalmente aos scores")
print("="*60)
