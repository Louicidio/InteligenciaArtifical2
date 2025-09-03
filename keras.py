import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita otimizações oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduz mensagens de log do TensorFlow

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("🏎️ ANÁLISE E PREVISÃO F1 2025 - CAMPEONATO DE PILOTOS 🏆")
print("="*60)

try:
    # Carregamento dos datasets
    print("\n📊 Carregando dados...")
    results = pd.read_csv("results.csv")
    drivers = pd.read_csv("drivers.csv")
    races = pd.read_csv("races.csv")
    driver_standings = pd.read_csv("driver_standings.csv")
    
    print(f"✅ Dados carregados: {len(results)} resultados, {len(drivers)} pilotos, {len(races)} corridas")

    # Merge dos dados principais
    data = results.merge(drivers, on="driverId").merge(races, on="raceId")
    standings_data = driver_standings.merge(races, on="raceId")

    print("\n🔍 Analisando dados recentes (2020-2024)...")

    # Filtrar dados recentes para análise
    recent_years = [2020, 2021, 2022, 2023, 2024]
    recent_data = data[data['year'].isin(recent_years)].copy()

    # Converter position para numérico, tratando 'R' e '\\N' como DNF
    recent_data['position_num'] = pd.to_numeric(recent_data['position'], errors='coerce')

    # Análise dos campeões recentes
    champions_by_year = []
    for year in recent_years:
        year_standings = standings_data[standings_data['year'] == year]
        if not year_standings.empty:
            final_standings = year_standings.loc[year_standings.groupby('driverId')['raceId'].idxmax()]
            if not final_standings.empty:
                champion = final_standings.loc[final_standings['points'].idxmax()]
                champion_info = drivers[drivers['driverId'] == champion['driverId']]
                if not champion_info.empty:
                    champion_info = champion_info.iloc[0]
                    champions_by_year.append({
                        'year': year,
                        'driverId': champion['driverId'],
                        'name': f"{champion_info['forename']} {champion_info['surname']}",
                        'points': champion['points']
                    })

    print("\n🏆 CAMPEÕES RECENTES:")
    for champ in champions_by_year:
        print(f"  {champ['year']}: {champ['name']} ({champ['points']} pontos)")

    # Análise estatística dos pilotos ativos
    print("\n📈 Calculando estatísticas dos pilotos ativos...")

    # Identificar pilotos ativos (correram em 2024)
    active_drivers = recent_data[recent_data['year'] == 2024]['driverId'].unique()

    driver_stats = []
    for driver_id in active_drivers:
        driver_data = recent_data[recent_data['driverId'] == driver_id]
        driver_info = drivers[drivers['driverId'] == driver_id]
        
        if driver_info.empty:
            continue
            
        driver_info = driver_info.iloc[0]
        
        # Calcular estatísticas
        total_races = len(driver_data)
        wins = len(driver_data[driver_data['position_num'] == 1])
        podiums = len(driver_data[driver_data['position_num'].isin([1, 2, 3])])
        points_2024 = driver_data[driver_data['year'] == 2024]['points'].sum()
        
        # Posição média apenas para corridas terminadas
        finished_races = driver_data[driver_data['position_num'].notna()]
        avg_position = finished_races['position_num'].mean() if len(finished_races) > 0 else 20
        
        dnf_rate = len(driver_data[driver_data['position_num'].isna()]) / total_races if total_races > 0 else 1
        
        # Tendência de performance (melhoria ao longo dos anos)
        yearly_points = driver_data.groupby('year')['points'].sum()
        if len(yearly_points) > 1:
            years = list(yearly_points.index)
            points = list(yearly_points.values)
            trend = np.polyfit(years, points, 1)[0] if len(years) > 1 else 0
        else:
            trend = 0
        
        driver_stats.append({
            'driverId': driver_id,
            'name': f"{driver_info['forename']} {driver_info['surname']}",
            'nationality': driver_info['nationality'],
            'total_races': total_races,
            'wins': wins,
            'podiums': podiums,
            'points_2024': points_2024,
            'win_rate': wins / total_races if total_races > 0 else 0,
            'podium_rate': podiums / total_races if total_races > 0 else 0,
            'avg_position': avg_position,
            'dnf_rate': dnf_rate,
            'performance_trend': trend
        })

    driver_stats_df = pd.DataFrame(driver_stats)

    print("\n🎯 TOP 10 PILOTOS ATIVOS (por pontos em 2024):")
    top_drivers = driver_stats_df.sort_values('points_2024', ascending=False).head(10)
    for idx, driver in top_drivers.iterrows():
        print(f"  {driver['name']:25}: {driver['points_2024']:3.0f} pts | {driver['wins']:2.0f} vitórias | {driver['podiums']:2.0f} pódios")

    # Calcular probabilidades de campeonato
    print("\n🤖 Calculando probabilidades de campeonato...")

    # Normalizar métricas
    max_points = driver_stats_df['points_2024'].max()
    max_trend = abs(driver_stats_df['performance_trend']).max()

    # Criar score de campeonato baseado em múltiplos fatores
    championship_score = (
        driver_stats_df['win_rate'] * 0.25 +  # Taxa de vitórias
        driver_stats_df['podium_rate'] * 0.20 +  # Taxa de pódios
        (1 / (driver_stats_df['avg_position'] + 1)) * 0.15 +  # Consistência
        (1 - driver_stats_df['dnf_rate']) * 0.10 +  # Confiabilidade
        (driver_stats_df['points_2024'] / max_points) * 0.25 +  # Performance 2024
        (driver_stats_df['performance_trend'] / max_trend if max_trend > 0 else 0) * 0.05  # Tendência
    )

    # Converter para probabilidades
    championship_probabilities = championship_score / championship_score.sum()

    # Criar resultado final
    final_predictions = driver_stats_df.copy()
    final_predictions['championship_probability'] = championship_probabilities
    final_predictions['championship_score'] = championship_score

    print("\n🏁 PREVISÃO CAMPEONATO F1 2025:")
    print("="*50)

    top_contenders = final_predictions.sort_values('championship_probability', ascending=False).head(10)

    for idx, driver in top_contenders.iterrows():
        prob_percent = driver['championship_probability'] * 100
        print(f"🏆 {driver['name']:25} | {prob_percent:5.1f}% | Score: {driver['championship_score']:.3f}")

    print("\n📊 ANÁLISE DOS TOP 3 CANDIDATOS:")
    print("="*40)

    for i, (idx, driver) in enumerate(top_contenders.head(3).iterrows()):
        print(f"\n{i+1}º {driver['name']} ({driver['nationality']})")
        print(f"   🎯 Probabilidade: {driver['championship_probability']*100:.1f}%")
        print(f"   🏁 Taxa de vitórias: {driver['win_rate']*100:.1f}%")
        print(f"   🥇 Taxa de pódios: {driver['podium_rate']*100:.1f}%")
        print(f"   📍 Posição média: {driver['avg_position']:.1f}")
        print(f"   💯 Pontos 2024: {driver['points_2024']:.0f}")
        
        if driver['performance_trend'] > 0:
            print(f"   📈 Tendência: Melhorando (+{driver['performance_trend']:.1f} pts/ano)")
        else:
            print(f"   📉 Tendência: Declinando ({driver['performance_trend']:.1f} pts/ano)")

    print(f"\n🎲 FAVORITO PARA 2025: {top_contenders.iloc[0]['name']}")
    print(f"💪 Com {top_contenders.iloc[0]['championship_probability']*100:.1f}% de chance de conquistar o título!")

    print("\n" + "="*60)
    print("📋 FATORES CONSIDERADOS NA ANÁLISE:")
    print("• Performance em 2024 (pontos, vitórias, pódios) - 25%")
    print("• Taxa de vitórias histórica (2020-2024) - 25%")
    print("• Taxa de pódios - 20%")
    print("• Consistência (posição média) - 15%")
    print("• Confiabilidade (taxa de abandonos) - 10%")
    print("• Tendência de melhoria - 5%")
    print("="*60)

except Exception as e:
    print(f"❌ Erro durante a análise: {e}")
    import traceback
    traceback.print_exc()
