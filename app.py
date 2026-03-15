"""
AI-OSINT: Цифровой анализ информационного поля Казахстана
в глобальном медиапространстве

Альфа-версия | Март 2026
Конкурс «AI SANA — Digital Kazakhstan: Projects of the Future»
Секция: Социально-гуманитарные науки

Методология: NLP + ABM + Markov Chains + Monte Carlo + OSINT
Данные: GDELT Project (демо-режим: синтетические данные на основе реальных паттернов)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime, timedelta
import json

# ═══════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI-OSINT | Информационное поле Казахстана",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Тёмная тема (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
    
    .stApp { font-family: 'JetBrains Mono', monospace; }
    
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 50%, #0d1f3c 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        border: 1px solid rgba(74, 158, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #4a9eff; font-size: 1.8rem; margin: 0; font-weight: 700; letter-spacing: 1px; }
    .main-header p { color: #8899aa; font-size: 0.95rem; margin-top: 0.5rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #0f1923 0%, #162233 100%);
        border: 1px solid rgba(74, 158, 255, 0.12); border-radius: 12px;
        padding: 1.2rem; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #6a7a8a; margin-top: 0.3rem; }
    
    .traffic-green { color: #4ade80; }
    .traffic-yellow { color: #fbbf24; }
    .traffic-red { color: #f87171; }
    
    .indicator-card {
        background: rgba(15, 25, 35, 0.8); border-radius: 10px; padding: 1rem;
        border-left: 4px solid; margin-bottom: 0.8rem;
    }
    
    .source-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 2px;
    }
    
    div[data-testid="stSidebar"] { background: #0a1628; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background: rgba(74, 158, 255, 0.08); border-radius: 8px; 
        color: #8899aa; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { 
        background: rgba(74, 158, 255, 0.2) !important; color: #4a9eff !important; 
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ДАННЫХ (Синтетические на основе GDELT-паттернов)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def generate_gdelt_data():
    """Генерация реалистичных данных по паттернам GDELT для Казахстана"""
    np.random.seed(42)
    dates = pd.date_range('2021-06-01', '2025-12-31', freq='D')
    n = len(dates)
    
    # Базовая частота упоминаний (с трендом роста)
    base = 120 + np.arange(n) * 0.03 + np.random.normal(0, 15, n)
    
    # Реальные информационные всплески (исторические события КЗ)
    events = {
        '2022-01-05': {'spike': 800, 'duration': 14, 'label': 'Қаңтар оқиғалары / Январские события'},
        '2022-01-07': {'spike': 1200, 'duration': 10, 'label': 'Пик освещения Кантара'},
        '2022-06-14': {'spike': 350, 'duration': 5, 'label': 'ПМЭФ — выступление Токаева'},
        '2022-09-15': {'spike': 400, 'duration': 4, 'label': 'Саммит ШОС, Самарканд'},
        '2022-10-14': {'spike': 280, 'duration': 3, 'label': 'Саммит СНГ, Астана'},
        '2022-11-20': {'spike': 220, 'duration': 5, 'label': 'Референдум по Конституции'},
        '2023-05-19': {'spike': 300, 'duration': 3, 'label': 'Саммит Центральная Азия — ЕС'},
        '2023-10-17': {'spike': 280, 'duration': 3, 'label': 'Один пояс — один путь, Пекин'},
        '2024-07-03': {'spike': 350, 'duration': 4, 'label': 'Саммит ШОС, Астана'},
        '2024-11-07': {'spike': 200, 'duration': 5, 'label': 'COP29 — КЗ как наблюдатель'},
        '2025-03-22': {'spike': 180, 'duration': 3, 'label': 'Наурыз — глобальное освещение'},
    }
    
    for date_str, info in events.items():
        idx = (dates >= pd.Timestamp(date_str)).argmax()
        for d in range(info['duration']):
            if idx + d < n:
                decay = np.exp(-d * 0.3)
                base[idx + d] += info['spike'] * decay
    
    # Тональность (Average Tone по шкале GDELT: -10 до +10)
    tone = np.random.normal(0.5, 1.8, n)  # Казахстан — в среднем нейтральный
    # Негатив во время Кантара
    jan22_start = (dates >= pd.Timestamp('2022-01-04')).argmax()
    tone[jan22_start:jan22_start+20] -= np.linspace(5, 1, 20)
    # Позитив во время саммитов
    for date_str in ['2022-09-15', '2024-07-03']:
        idx = (dates >= pd.Timestamp(date_str)).argmax()
        tone[idx:idx+5] += np.linspace(2.5, 0.5, 5)
    
    # Источники по странам (доля упоминаний)
    sources = {
        'Россия': 0.28, 'Китай': 0.12, 'США': 0.10, 'Великобритания': 0.08,
        'Турция': 0.07, 'Казахстан (внутр.)': 0.15, 'Центр. Азия': 0.06,
        'ЕС (прочие)': 0.08, 'Прочие': 0.06
    }
    
    # Языковое распределение
    languages = {'Русский': 0.42, 'Английский': 0.31, 'Казахский': 0.12, 'Китайский': 0.08, 'Прочие': 0.07}
    
    # Нарративные кластеры (BERTopic-подобные)
    narratives = [
        {'id': 0, 'label': 'Геополитика и многовекторность', 'weight': 0.22, 'sentiment': 0.3},
        {'id': 1, 'label': 'Экономика и инвестиции', 'weight': 0.20, 'sentiment': 1.2},
        {'id': 2, 'label': 'Внутренняя политика и реформы', 'weight': 0.18, 'sentiment': -0.5},
        {'id': 3, 'label': 'Энергетика и ресурсы', 'weight': 0.15, 'sentiment': 0.8},
        {'id': 4, 'label': 'Безопасность и стабильность', 'weight': 0.13, 'sentiment': -1.1},
        {'id': 5, 'label': 'Культура и soft power', 'weight': 0.07, 'sentiment': 2.1},
        {'id': 6, 'label': 'Права человека и критика', 'weight': 0.05, 'sentiment': -3.2},
    ]
    
    df = pd.DataFrame({
        'date': dates,
        'mentions': np.maximum(base, 20).astype(int),
        'tone': np.clip(tone, -8, 8),
        'tone_ma30': pd.Series(tone).rolling(30, min_periods=1).mean().values,
    })
    
    return df, sources, languages, narratives, events

# ═══════════════════════════════════════════════════════════════
# ABM ДВИЖОК + МАРКОВСКИЕ ЦЕПИ
# ═══════════════════════════════════════════════════════════════
class InformationABM:
    """
    Агентно-ориентированная модель информационных кампаний
    с марковскими цепями для моделирования состояний нарратива.
    
    Типы агентов:
    - Инициатор (Initiator): генерирует оригинальный нарратив
    - Усилитель (Amplifier): бот-сети, координированные аккаунты
    - Медиатор (Mediator): СМИ, крупные каналы
    - Реципиент (Recipient): обычные пользователи
    
    Состояния нарратива (Markov Chain):
    LATENT → EMERGING → GROWING → VIRAL → DECLINING
    """
    
    # Состояния нарратива
    STATES = ['LATENT', 'EMERGING', 'GROWING', 'VIRAL', 'DECLINING']
    STATE_COLORS = {'LATENT': '#6b7280', 'EMERGING': '#3b82f6', 'GROWING': '#f59e0b', 'VIRAL': '#ef4444', 'DECLINING': '#8b5cf6'}
    
    # Базовая матрица переходов (модифицируется в зависимости от усилителей)
    BASE_TRANSITION = np.array([
        #  LAT   EME   GRO   VIR   DEC
        [0.70, 0.25, 0.05, 0.00, 0.00],  # LATENT
        [0.10, 0.45, 0.35, 0.10, 0.00],  # EMERGING
        [0.00, 0.05, 0.40, 0.40, 0.15],  # GROWING
        [0.00, 0.00, 0.05, 0.55, 0.40],  # VIRAL
        [0.05, 0.00, 0.00, 0.05, 0.90],  # DECLINING
    ])
    
    def __init__(self, n_agents=80, seed=42):
        np.random.seed(seed)
        self.n_agents = n_agents
        self.tick = 0
        self.narrative_state = 0  # LATENT
        self.state_history = [0]
        
        # Генерация агентов
        types = np.random.choice(
            ['initiator', 'amplifier', 'mediator', 'recipient'],
            size=n_agents,
            p=[0.05, 0.20, 0.10, 0.65]
        )
        
        self.agents = []
        for i, t in enumerate(types):
            agent = {
                'id': i, 'type': t,
                'active': t == 'initiator',
                'influence': np.random.uniform(0.3, 1.0) if t in ['initiator', 'mediator'] else np.random.uniform(0.05, 0.3),
                'susceptibility': np.random.uniform(0.4, 0.9) if t == 'recipient' else 0.1,
                'reach': np.random.randint(100, 50000) if t == 'mediator' else np.random.randint(10, 5000),
                'sync_score': 0.0,
                'exposure_count': 0,
                'language': np.random.choice(['ru', 'kz', 'en'], p=[0.55, 0.30, 0.15]),
            }
            if t == 'amplifier':
                agent['bot_probability'] = np.random.uniform(0.6, 0.95)
                agent['repost_rate'] = np.random.uniform(0.7, 0.98)
            self.agents.append(agent)
        
        # Граф связей
        self.G = nx.barabasi_albert_graph(n_agents, 3, seed=seed)
        # Назначаем типы узлам
        for i, agent in enumerate(self.agents):
            self.G.nodes[i]['type'] = agent['type']
            self.G.nodes[i]['active'] = agent['active']
        
        self.history = []
        self.anomaly_scores = []
    
    def get_transition_matrix(self):
        """Модифицированная матрица переходов на основе текущего состояния сети"""
        M = self.BASE_TRANSITION.copy()
        
        # Считаем активных усилителей
        active_amplifiers = sum(1 for a in self.agents if a['type'] == 'amplifier' and a['active'])
        total_amplifiers = sum(1 for a in self.agents if a['type'] == 'amplifier')
        amp_ratio = active_amplifiers / max(total_amplifiers, 1)
        
        # Активные медиаторы ускоряют переход в VIRAL
        active_mediators = sum(1 for a in self.agents if a['type'] == 'mediator' and a['active'])
        
        # Больше усилителей → быстрее переход вправо (к VIRAL)
        if amp_ratio > 0.3:
            boost = amp_ratio * 0.25
            for i in range(4):  # Для каждого состояния кроме DECLINING
                if i + 1 < 5:
                    shift = min(boost, M[i][i] * 0.5)
                    M[i][i] -= shift
                    M[i][min(i+1, 4)] += shift
        
        # Медиаторы: из GROWING → VIRAL быстрее
        if active_mediators > 2:
            med_boost = min(0.15, active_mediators * 0.03)
            M[2][3] += med_boost
            M[2][2] -= med_boost
        
        # Нормализация строк
        for i in range(5):
            M[i] = np.clip(M[i], 0, 1)
            M[i] /= M[i].sum()
        
        return M
    
    def step(self):
        """Один шаг симуляции"""
        self.tick += 1
        
        # 1. Марковский переход состояния нарратива
        M = self.get_transition_matrix()
        current = self.narrative_state
        self.narrative_state = np.random.choice(5, p=M[current])
        self.state_history.append(self.narrative_state)
        
        # 2. Активация агентов в зависимости от состояния
        state_activation = {0: 0.02, 1: 0.10, 2: 0.25, 3: 0.50, 4: 0.05}
        activation_prob = state_activation[self.narrative_state]
        
        for agent in self.agents:
            if not agent['active']:
                # Проверяем соседей
                neighbors = list(self.G.neighbors(agent['id']))
                active_neighbors = sum(1 for n in neighbors if self.agents[n]['active'])
                neighbor_influence = active_neighbors / max(len(neighbors), 1)
                
                # Вероятность активации
                p_activate = activation_prob * agent['susceptibility'] * (1 + neighbor_influence)
                if agent['type'] == 'amplifier':
                    p_activate *= 2.0  # Усилители активируются быстрее
                
                if np.random.random() < min(p_activate, 0.95):
                    agent['active'] = True
                    agent['exposure_count'] += 1
            
            # Синхронность (для детекции координации)
            if agent['active'] and agent['type'] == 'amplifier':
                agent['sync_score'] = min(1.0, agent['sync_score'] + 0.1)
        
        # 3. Расчёт метрик
        active_count = sum(1 for a in self.agents if a['active'])
        active_ratio = active_count / self.n_agents
        
        sync = np.mean([a['sync_score'] for a in self.agents if a['type'] == 'amplifier'])
        
        # Z-score аномальности
        if len(self.history) > 5:
            recent = [h['active_ratio'] for h in self.history[-30:]]
            mu, sigma = np.mean(recent), max(np.std(recent), 0.01)
            z_score = (active_ratio - mu) / sigma
        else:
            z_score = 0
        
        record = {
            'tick': self.tick,
            'state': self.STATES[self.narrative_state],
            'active_count': active_count,
            'active_ratio': active_ratio,
            'sync_score': sync,
            'z_score': z_score,
            'anomaly': abs(z_score) > 2.5,
        }
        self.history.append(record)
        return record
    
    def run_monte_carlo(self, n_simulations=1000, n_steps=50, scenario_amp=0.05):
        """Monte Carlo симуляция: N прогонов с вариацией параметров.
        scenario_amp: базовая доля усилителей (зависит от сценария)
        """
        results = []
        peak_states = []
        
        for sim in range(n_simulations):
            state = 0  # LATENT
            max_state_reached = 0
            steps_to_viral = None
            
            for step in range(n_steps):
                M = self.BASE_TRANSITION.copy()
                
                # Случайная вариация матрицы
                noise = np.random.normal(0, 0.03, M.shape)
                M += noise
                
                # КЛЮЧЕВОЕ: усиление зависит от сценария
                amp_factor = np.random.uniform(
                    max(0, scenario_amp - 0.1), 
                    min(1.0, scenario_amp + 0.15)
                )
                
                # Чем больше усилителей — тем быстрее переход вправо
                if amp_factor > 0.1:
                    for i in range(4):
                        boost = amp_factor * 0.2
                        M[i][min(i+1, 4)] += boost
                        M[i][i] -= boost * 0.8
                
                # Нормализация
                M = np.clip(M, 0.001, 1)
                for i in range(5):
                    M[i] /= M[i].sum()
                
                state = np.random.choice(5, p=M[state])
                max_state_reached = max(max_state_reached, state)
                
                if state == 3 and steps_to_viral is None:
                    steps_to_viral = step
            
            results.append({
                'max_state': max_state_reached,
                'final_state': state,
                'steps_to_viral': steps_to_viral,
                'reached_viral': max_state_reached >= 3,
            })
            peak_states.append(max_state_reached)
        
        # Статистика
        viral_prob = sum(1 for r in results if r['reached_viral']) / n_simulations
        steps_viral = [r['steps_to_viral'] for r in results if r['steps_to_viral'] is not None]
        
        return {
            'n': n_simulations,
            'n_steps': n_steps,
            'peak_distribution': np.bincount(peak_states, minlength=5) / n_simulations,
            'viral_probability': viral_prob,
            'mean_steps_to_viral': np.mean(steps_viral) if steps_viral else None,
            'std_steps_to_viral': np.std(steps_viral) if steps_viral else None,
            'final_state_dist': np.bincount([r['final_state'] for r in results], minlength=5) / n_simulations,
            'peak_states': peak_states,
        }

# ═══════════════════════════════════════════════════════════════
# ИНДИКАТОРЫ ОБНАРУЖЕНИЯ
# ═══════════════════════════════════════════════════════════════
INDICATORS = {
    'anomaly_index': {
        'name': 'Индекс аномальности',
        'desc': 'Z-score частоты/тональности относительно скользящего среднего за 30 дней',
        'method': 'Z = (x − μ₃₀) / σ₃₀',
        'thresholds': {'green': 1.5, 'yellow': 2.5, 'red': 3.5},
        'source': 'GDELT Timeline API'
    },
    'sync_coefficient': {
        'name': 'Коэффициент синхронности',
        'desc': 'Доля агентов, опубликовавших нарратив в одном временном окне (<30 мин)',
        'method': 'S = count(posts ∈ window T) / total_agents',
        'thresholds': {'green': 0.15, 'yellow': 0.35, 'red': 0.55},
        'source': 'ABM Engine'
    },
    'text_homogeneity': {
        'name': 'Текстовая гомогенность',
        'desc': 'Cosine similarity между постами из разных каналов',
        'method': 'cos_sim > 0.85 между ≥ 3 источниками (TF-IDF + sentence-transformers)',
        'thresholds': {'green': 0.4, 'yellow': 0.7, 'red': 0.85},
        'source': 'NLP Pipeline (XLM-RoBERTa)'
    },
    'sentiment_shift': {
        'name': 'Тональный сдвиг',
        'desc': 'Резкое отклонение sentiment от нормы для данного источника',
        'method': '|Δsentiment| > 0.4 за 24ч или отклонение > 2σ от исторической нормы',
        'thresholds': {'green': 0.2, 'yellow': 0.4, 'red': 0.7},
        'source': 'GDELT AvgTone + XLM-RoBERTa'
    },
    'propagation_speed': {
        'name': 'Скорость распространения',
        'desc': 'Время от вброса до охвата N% аудитории',
        'method': 'Шаги ABM до порога. S-кривая = органика, вертикаль = координация',
        'thresholds': {'green': 20, 'yellow': 10, 'red': 5},
        'source': 'ABM Simulation'
    },
    'bot_activity': {
        'name': 'Бот-активность',
        'desc': 'Нечеловеческая регулярность, TTR < 0.3, кластерная активация',
        'method': 'CV интервалов < 0.1, доля репостов > 90%, TTR < 0.3',
        'thresholds': {'green': 0.1, 'yellow': 0.3, 'red': 0.6},
        'source': 'NLP + Temporal Analysis'
    },
}

# ═══════════════════════════════════════════════════════════════
# СЦЕНАРИИ ИНФОРМАЦИОННЫХ КАМПАНИЙ
# ═══════════════════════════════════════════════════════════════
SCENARIOS = {
    'organic': {
        'name': 'Органическое распространение',
        'desc': 'Естественная динамика информации без координации',
        'color': '#4ade80',
        'amp_ratio': 0.05,
        'mediator_boost': 0.0,
    },
    'amplified': {
        'name': 'Усиленная кампания',
        'desc': 'Использование бот-сетей и координированных аккаунтов',
        'color': '#f59e0b',
        'amp_ratio': 0.35,
        'mediator_boost': 0.1,
    },
    'coordinated': {
        'name': 'Координированная операция',
        'desc': 'Полноценная информационная операция с внешним управлением',
        'color': '#ef4444',
        'amp_ratio': 0.60,
        'mediator_boost': 0.25,
    },
    'hybrid': {
        'name': 'Гибридная кампания',
        'desc': 'Смешение органики и координации — наиболее сложная для детекции',
        'color': '#a855f7',
        'amp_ratio': 0.25,
        'mediator_boost': 0.15,
    },
}

# ═══════════════════════════════════════════════════════════════
# STREAMLIT ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════
def main():
    # ═══ HEADER ═══
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI-OSINT</h1>
        <p>Цифровой анализ информационного поля Казахстана в глобальном медиапространстве</p>
        <p style="font-size: 0.8rem; color: #556677; margin-top: 0.8rem;">
            ABM + Markov Chains + Monte Carlo + NLP + GDELT | Конкурс «AI SANA — Digital Kazakhstan» | 2026
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ═══ SIDEBAR ═══
    with st.sidebar:
        st.markdown("### ⚙️ Параметры")
        
        scenario_key = st.selectbox(
            "Сценарий кампании",
            list(SCENARIOS.keys()),
            format_func=lambda x: SCENARIOS[x]['name']
        )
        scenario = SCENARIOS[scenario_key]
        
        st.markdown("---")
        st.markdown("### 🎲 Monte Carlo")
        mc_n = st.slider("Число итераций", 100, 5000, 1000, 100)
        mc_steps = st.slider("Шагов на итерацию", 10, 100, 50, 5)
        
        st.markdown("---")
        st.markdown("### 🤖 ABM")
        n_agents = st.slider("Число агентов", 30, 200, 80, 10)
        sim_steps = st.slider("Шагов симуляции", 10, 100, 40, 5)
        
        st.markdown("---")
        st.markdown("### 📊 Данные")
        st.markdown("""
        **Источники:**
        - [GDELT Project](https://www.gdeltproject.org/)
        - [GDELT DOC API](https://api.gdeltproject.org/api/v2/doc/doc?query=Kazakhstan&mode=artlist&format=json)
        - [GDELT Timeline API](https://api.gdeltproject.org/api/v2/doc/doc?query=Kazakhstan&mode=timelinetone)
        - [BigQuery GDELT](https://console.cloud.google.com/marketplace/product/the-gdelt-project/gdelt-2-events)
        - [Stanford IO Reports](https://cyber.fsi.stanford.edu/io/publications)
        - [Meta CIB Reports](https://transparency.meta.com/metasecurity/threat-reporting)
        - [IRA Troll Tweets](https://github.com/fivethirtyeight/russian-troll-tweets)
        """)
    
    # Загрузка данных
    df, sources, languages, narratives, events = generate_gdelt_data()
    
    # ═══ ТАБЫ ═══
    tabs = st.tabs([
        "📊 Dashboard", "🤖 ABM Симуляция", "⛓️ Марковские цепи", 
        "🎲 Monte Carlo", "🔍 Индикаторы", "📋 Методология"
    ])
    
    # ─────────── TAB 1: DASHBOARD ───────────
    with tabs[0]:
        # Ключевые метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_mentions = df['mentions'].mean()
            st.metric("Ср. упоминаний/день", f"{avg_mentions:.0f}", f"+{df['mentions'].iloc[-30:].mean() - avg_mentions:.0f} (30д)")
        with col2:
            avg_tone = df['tone'].mean()
            st.metric("Ср. тональность", f"{avg_tone:+.2f}", f"{df['tone'].iloc[-30:].mean() - avg_tone:+.2f}")
        with col3:
            max_spike = df['mentions'].max()
            st.metric("Макс. всплеск", f"{max_spike:.0f}", "Январь 2022")
        with col4:
            days_total = len(df)
            st.metric("Дней анализа", f"{days_total}", f"{len(events)} событий")
        
        st.markdown("---")
        
        # Временной ряд частоты упоминаний
        st.markdown("#### 📈 Частота упоминаний Казахстана в мировых СМИ (GDELT)")
        fig_mentions = go.Figure()
        fig_mentions.add_trace(go.Scatter(
            x=df['date'], y=df['mentions'], mode='lines',
            name='Упоминания/день', line=dict(color='#4a9eff', width=1),
            fill='tozeroy', fillcolor='rgba(74, 158, 255, 0.1)'
        ))
        # MA30
        df['mentions_ma30'] = df['mentions'].rolling(30, min_periods=1).mean()
        fig_mentions.add_trace(go.Scatter(
            x=df['date'], y=df['mentions_ma30'], mode='lines',
            name='MA30', line=dict(color='#f59e0b', width=2, dash='dash')
        ))
        # Аннотации событий
        for date_str, info in events.items():
            if info['spike'] > 250:
                fig_mentions.add_annotation(
                    x=date_str, y=info['spike'] + 100,
                    text=info['label'][:30], showarrow=True,
                    arrowhead=2, font=dict(size=10, color='#f87171'),
                    bgcolor='rgba(15,25,35,0.8)', bordercolor='#f87171'
                )
        fig_mentions.update_layout(
            template='plotly_dark', height=420,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig_mentions, use_container_width=True)
        
        # Тональность
        col_tone, col_src = st.columns([3, 2])
        
        with col_tone:
            st.markdown("#### 😊😐😠 Тональность (Average Tone)")
            fig_tone = go.Figure()
            fig_tone.add_trace(go.Scatter(
                x=df['date'], y=df['tone'], mode='lines',
                name='Tone (daily)', line=dict(color='rgba(74,158,255,0.3)', width=1)
            ))
            fig_tone.add_trace(go.Scatter(
                x=df['date'], y=df['tone_ma30'], mode='lines',
                name='MA30', line=dict(color='#4ade80', width=2.5)
            ))
            fig_tone.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.2)')
            fig_tone.update_layout(
                template='plotly_dark', height=300,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                margin=dict(l=40, r=20, t=20, b=40),
                yaxis_title='Tone (-10...+10)'
            )
            st.plotly_chart(fig_tone, use_container_width=True)
        
        with col_src:
            st.markdown("#### 🌍 Источники по странам")
            fig_src = go.Figure(data=[go.Pie(
                labels=list(sources.keys()),
                values=list(sources.values()),
                hole=0.5,
                marker_colors=['#ef4444', '#f59e0b', '#3b82f6', '#06b6d4', '#10b981', '#4a9eff', '#a855f7', '#6366f1', '#6b7280']
            )])
            fig_src.update_layout(
                template='plotly_dark', height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True, legend=dict(font=dict(size=11))
            )
            st.plotly_chart(fig_src, use_container_width=True)
        
        # Нарративные кластеры
        col_nar, col_lang = st.columns(2)
        with col_nar:
            st.markdown("#### 💬 Нарративные кластеры (BERTopic)")
            nar_df = pd.DataFrame(narratives)
            fig_nar = go.Figure(data=[go.Bar(
                y=nar_df['label'], x=nar_df['weight'],
                orientation='h',
                marker_color=[
                    '#3b82f6' if s > 0 else '#ef4444' 
                    for s in nar_df['sentiment']
                ],
                text=[f"{w:.0%}" for w in nar_df['weight']],
                textposition='auto'
            )])
            fig_nar.update_layout(
                template='plotly_dark', height=320,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title='Доля в потоке'
            )
            st.plotly_chart(fig_nar, use_container_width=True)
        
        with col_lang:
            st.markdown("#### 🗣️ Языковое распределение")
            fig_lang = go.Figure(data=[go.Pie(
                labels=list(languages.keys()),
                values=list(languages.values()),
                hole=0.6,
                marker_colors=['#ef4444', '#3b82f6', '#4ade80', '#f59e0b', '#6b7280']
            )])
            fig_lang.update_layout(
                template='plotly_dark', height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig_lang, use_container_width=True)
    
    # ─────────── TAB 2: ABM СИМУЛЯЦИЯ ───────────
    with tabs[1]:
        st.markdown(f"#### 🤖 Агентно-ориентированная модель | Сценарий: **{scenario['name']}**")
        st.markdown(f"*{scenario['desc']}*")
        
        # Инициализация / запуск
        if st.button("▶ Запуск симуляции", type="primary", key="run_abm"):
            abm = InformationABM(n_agents=n_agents)
            
            # Настройка под сценарий: активируем усилителей
            for agent in abm.agents:
                if agent['type'] == 'amplifier':
                    if np.random.random() < scenario['amp_ratio']:
                        agent['active'] = True
            
            progress = st.progress(0)
            status = st.empty()
            
            for step in range(sim_steps):
                record = abm.step()
                progress.progress((step + 1) / sim_steps)
                status.text(f"Шаг {step+1}/{sim_steps} | Состояние: {record['state']} | Активных: {record['active_count']}/{n_agents}")
            
            st.session_state['abm'] = abm
            st.session_state['abm_done'] = True
        
        if st.session_state.get('abm_done'):
            abm = st.session_state['abm']
            
            col_net, col_stats = st.columns([3, 2])
            
            with col_net:
                st.markdown("##### 🕸️ Граф информационной сети")
                
                # Граф
                pos = nx.spring_layout(abm.G, seed=42, k=1.5/np.sqrt(abm.n_agents))
                
                edge_x, edge_y = [], []
                for e in abm.G.edges():
                    x0, y0 = pos[e[0]]
                    x1, y1 = pos[e[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                
                type_colors = {
                    'initiator': '#ef4444', 'amplifier': '#f59e0b', 
                    'mediator': '#3b82f6', 'recipient': '#6b7280'
                }
                type_sizes = {
                    'initiator': 18, 'amplifier': 10, 
                    'mediator': 14, 'recipient': 7
                }
                
                fig_graph = go.Figure()
                fig_graph.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode='lines',
                    line=dict(width=0.3, color='rgba(100,120,140,0.3)'),
                    hoverinfo='none'
                ))
                
                for t_name, t_color in type_colors.items():
                    nodes_of_type = [i for i, a in enumerate(abm.agents) if a['type'] == t_name]
                    fig_graph.add_trace(go.Scatter(
                        x=[pos[n][0] for n in nodes_of_type],
                        y=[pos[n][1] for n in nodes_of_type],
                        mode='markers',
                        name=t_name.capitalize(),
                        marker=dict(
                            size=[type_sizes[t_name] * (1.5 if abm.agents[n]['active'] else 0.7) for n in nodes_of_type],
                            color=[t_color if abm.agents[n]['active'] else '#2a3a4a' for n in nodes_of_type],
                            line=dict(width=1, color=t_color),
                            opacity=[0.9 if abm.agents[n]['active'] else 0.3 for n in nodes_of_type]
                        ),
                        text=[f"Agent {n} ({t_name})<br>Active: {abm.agents[n]['active']}<br>Reach: {abm.agents[n]['reach']}" for n in nodes_of_type],
                        hoverinfo='text'
                    ))
                
                fig_graph.update_layout(
                    template='plotly_dark', height=500,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                    showlegend=True, legend=dict(orientation='h', y=-0.05),
                    margin=dict(l=0, r=0, t=10, b=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
                st.plotly_chart(fig_graph, use_container_width=True)
            
            with col_stats:
                st.markdown("##### 📊 Динамика симуляции")
                
                hist_df = pd.DataFrame(abm.history)
                
                # Активные агенты по шагам
                fig_active = go.Figure()
                fig_active.add_trace(go.Scatter(
                    x=hist_df['tick'], y=hist_df['active_count'],
                    mode='lines+markers', name='Активные агенты',
                    line=dict(color='#4a9eff', width=2),
                    marker=dict(size=4)
                ))
                fig_active.update_layout(
                    template='plotly_dark', height=200,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                    margin=dict(l=40, r=10, t=10, b=30),
                    yaxis_title='Агентов'
                )
                st.plotly_chart(fig_active, use_container_width=True)
                
                # Синхронность
                fig_sync = go.Figure()
                fig_sync.add_trace(go.Scatter(
                    x=hist_df['tick'], y=hist_df['sync_score'],
                    mode='lines', name='Синхронность',
                    line=dict(color='#f59e0b', width=2),
                    fill='tozeroy', fillcolor='rgba(245,158,11,0.1)'
                ))
                fig_sync.add_hline(y=0.35, line_dash='dash', line_color='#f87171',
                                   annotation_text='Порог координации')
                fig_sync.update_layout(
                    template='plotly_dark', height=200,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                    margin=dict(l=40, r=10, t=10, b=30),
                    yaxis_title='Sync'
                )
                st.plotly_chart(fig_sync, use_container_width=True)
                
                # Состав агентов
                type_counts = {}
                for a in abm.agents:
                    k = a['type']
                    if k not in type_counts:
                        type_counts[k] = {'total': 0, 'active': 0}
                    type_counts[k]['total'] += 1
                    if a['active']:
                        type_counts[k]['active'] += 1
                
                st.markdown("##### Состав агентов")
                for t_name, counts in type_counts.items():
                    pct = counts['active'] / counts['total'] * 100
                    color = type_colors.get(t_name, '#6b7280')
                    st.markdown(f"**{t_name.capitalize()}**: {counts['active']}/{counts['total']} ({pct:.0f}%)")
                    st.progress(pct / 100)
    
    # ─────────── TAB 3: МАРКОВСКИЕ ЦЕПИ ───────────
    with tabs[2]:
        st.markdown("#### ⛓️ Марковские цепи: состояния нарратива")
        st.markdown("""
        Нарратив в информационном пространстве проходит через дискретные состояния. 
        Матрица переходов модифицируется в реальном времени в зависимости от числа 
        активных усилителей и медиаторов.
        """)
        
        col_matrix, col_diagram = st.columns(2)
        
        with col_matrix:
            st.markdown("##### Матрица переходов (базовая)")
            
            fig_matrix = go.Figure(data=go.Heatmap(
                z=InformationABM.BASE_TRANSITION,
                x=InformationABM.STATES,
                y=InformationABM.STATES,
                colorscale='RdYlBu_r',
                text=np.round(InformationABM.BASE_TRANSITION, 2),
                texttemplate='%{text}',
                textfont=dict(size=14),
                hovertemplate='%{y} → %{x}: %{z:.2f}<extra></extra>'
            ))
            fig_matrix.update_layout(
                template='plotly_dark', height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title='В состояние →',
                yaxis_title='Из состояния ↓',
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        with col_diagram:
            st.markdown("##### Диаграмма состояний")
            
            state_labels = ['🔵 LATENT', '🟢 EMERGING', '🟡 GROWING', '🔴 VIRAL', '🟣 DECLINING']
            state_y = [4, 3, 2, 1, 0]
            
            fig_states = go.Figure()
            
            # Узлы
            colors = ['#6b7280', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
            for i, (label, y, c) in enumerate(zip(state_labels, state_y, colors)):
                fig_states.add_trace(go.Scatter(
                    x=[0.5], y=[y], mode='markers+text',
                    marker=dict(size=50, color=c, line=dict(width=2, color='white')),
                    text=[label], textposition='middle right', textfont=dict(size=14, color=c),
                    showlegend=False, hoverinfo='skip'
                ))
            
            # Стрелки (основные переходы)
            for i in range(4):
                fig_states.add_annotation(
                    x=0.5, y=state_y[i] - 0.15, ax=0.5, ay=state_y[i+1] + 0.15,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
                    arrowcolor=colors[i+1], opacity=0.7
                )
            
            fig_states.update_layout(
                template='plotly_dark', height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.3)',
                margin=dict(l=20, r=120, t=20, b=20),
                xaxis=dict(range=[0, 2], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-0.5, 4.5], showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_states, use_container_width=True)
        
        # Траектория нарратива (если есть симуляция)
        if st.session_state.get('abm_done'):
            abm = st.session_state['abm']
            st.markdown("##### 📈 Траектория нарратива (результат симуляции)")
            
            fig_trajectory = go.Figure()
            colors_map = [InformationABM.STATE_COLORS[InformationABM.STATES[s]] for s in abm.state_history]
            
            fig_trajectory.add_trace(go.Scatter(
                x=list(range(len(abm.state_history))),
                y=abm.state_history,
                mode='lines+markers',
                line=dict(color='#4a9eff', width=2),
                marker=dict(size=8, color=colors_map, line=dict(width=1, color='white')),
                text=[InformationABM.STATES[s] for s in abm.state_history],
                hovertemplate='Шаг %{x}<br>Состояние: %{text}<extra></extra>'
            ))
            
            fig_trajectory.update_layout(
                template='plotly_dark', height=300,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                margin=dict(l=40, r=20, t=10, b=40),
                xaxis_title='Шаг симуляции',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=InformationABM.STATES,
                    title='Состояние'
                )
            )
            st.plotly_chart(fig_trajectory, use_container_width=True)
        else:
            st.info("Запустите ABM-симуляцию (вкладка «ABM Симуляция»), чтобы увидеть траекторию нарратива.")
    
    # ─────────── TAB 4: MONTE CARLO ───────────
    with tabs[3]:
        st.markdown("#### 🎲 Симуляция Monte Carlo")
        st.markdown(f"**Параметры:** {mc_n} итераций × {mc_steps} шагов | Сценарий: {scenario['name']}")
        
        if st.button("▶ Запуск Monte Carlo", type="primary", key="run_mc"):
            abm_mc = InformationABM(n_agents=n_agents)
            
            with st.spinner(f"Выполняется {mc_n} итераций..."):
                mc_results = abm_mc.run_monte_carlo(n_simulations=mc_n, n_steps=mc_steps, scenario_amp=scenario['amp_ratio'])
            
            st.session_state['mc_results'] = mc_results
            st.session_state['mc_done'] = True
        
        if st.session_state.get('mc_done'):
            mc = st.session_state['mc_results']
            
            # Ключевые метрики MC
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("P(VIRAL)", f"{mc['viral_probability']:.1%}")
            with col2:
                if mc['mean_steps_to_viral']:
                    st.metric("Ср. шагов до VIRAL", f"{mc['mean_steps_to_viral']:.1f}")
                else:
                    st.metric("Ср. шагов до VIRAL", "N/A")
            with col3:
                st.metric("Итераций", f"{mc['n']}")
            with col4:
                st.metric("Шагов / итерацию", f"{mc['n_steps']}")
            
            st.markdown("---")
            
            col_hist, col_probs = st.columns([3, 2])
            
            with col_hist:
                st.markdown("##### Распределение пиковых состояний")
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(
                    x=InformationABM.STATES,
                    y=mc['peak_distribution'],
                    marker_color=['#6b7280', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'],
                    text=[f"{p:.1%}" for p in mc['peak_distribution']],
                    textposition='auto', textfont=dict(size=14)
                ))
                fig_hist.update_layout(
                    template='plotly_dark', height=350,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,22,40,0.5)',
                    margin=dict(l=40, r=20, t=10, b=40),
                    yaxis_title='Вероятность',
                    xaxis_title='Максимальное достигнутое состояние'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_probs:
                st.markdown("##### Финальные состояния")
                
                fig_final = go.Figure(data=[go.Pie(
                    labels=InformationABM.STATES,
                    values=mc['final_state_dist'],
                    hole=0.5,
                    marker_colors=['#6b7280', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
                )])
                fig_final.update_layout(
                    template='plotly_dark', height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=10, b=20),
                )
                st.plotly_chart(fig_final, use_container_width=True)
            
            # Гистограмма шагов до VIRAL
            if mc['mean_steps_to_viral']:
                st.markdown("##### Распределение: шагов до достижения VIRAL")
                viral_steps = [r for r in range(mc['n']) if mc['peak_states'][r] >= 3]
                st.markdown(f"Из {mc['n']} итераций нарратив достиг состояния VIRAL в **{len(viral_steps)}** случаях ({len(viral_steps)/mc['n']:.1%})")
        else:
            st.info("Нажмите «Запуск Monte Carlo» для выполнения симуляции.")
    
    # ─────────── TAB 5: ИНДИКАТОРЫ ───────────
    with tabs[4]:
        st.markdown("#### 🔍 Система индикаторов обнаружения")
        st.markdown("Светофорная система: 🟢 Норма | 🟡 Отклонение | 🔴 Аномалия")
        
        # Генерация текущих значений (демо)
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        for key, ind in INDICATORS.items():
            # Симулируем значение в зависимости от сценария
            if scenario_key == 'organic':
                val = np.random.uniform(0, ind['thresholds']['yellow'] * 0.8)
            elif scenario_key == 'amplified':
                val = np.random.uniform(ind['thresholds']['green'], ind['thresholds']['red'])
            elif scenario_key == 'coordinated':
                val = np.random.uniform(ind['thresholds']['yellow'], ind['thresholds']['red'] * 1.3)
            else:
                val = np.random.uniform(ind['thresholds']['green'] * 0.5, ind['thresholds']['red'] * 0.9)
            
            # Определяем цвет
            if val < ind['thresholds']['green']:
                color, status, emoji = '#4ade80', 'Норма', '🟢'
            elif val < ind['thresholds']['yellow']:
                color, status, emoji = '#fbbf24', 'Внимание', '🟡'
            else:
                color, status, emoji = '#f87171', 'Аномалия', '🔴'
            
            with st.expander(f"{emoji} **{ind['name']}** — {status} ({val:.2f})", expanded=(color == '#f87171')):
                col_desc, col_val = st.columns([3, 1])
                with col_desc:
                    st.markdown(f"**Описание:** {ind['desc']}")
                    st.markdown(f"**Формула:** `{ind['method']}`")
                    st.markdown(f"**Источник:** {ind['source']}")
                with col_val:
                    st.markdown(f"### <span style='color:{color}'>{val:.3f}</span>", unsafe_allow_html=True)
                    st.markdown(f"Пороги: 🟢 <{ind['thresholds']['green']} | 🟡 <{ind['thresholds']['yellow']} | 🔴 ≥{ind['thresholds']['red']}")
        
        # Агрегированный индекс
        st.markdown("---")
        st.markdown("#### 🎯 Агрегированный индекс угрозы")
        
        if scenario_key == 'organic':
            threat_idx = np.random.uniform(0.1, 0.25)
        elif scenario_key == 'amplified':
            threat_idx = np.random.uniform(0.35, 0.55)
        elif scenario_key == 'coordinated':
            threat_idx = np.random.uniform(0.65, 0.90)
        else:
            threat_idx = np.random.uniform(0.30, 0.60)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=threat_idx * 100,
            title={'text': "Threat Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#4a9eff'},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(74,222,128,0.2)'},
                    {'range': [30, 60], 'color': 'rgba(251,191,36,0.2)'},
                    {'range': [60, 100], 'color': 'rgba(248,113,113,0.2)'}
                ],
                'threshold': {
                    'line': {'color': '#f87171', 'width': 4},
                    'thickness': 0.75, 'value': threat_idx * 100
                }
            }
        ))
        fig_gauge.update_layout(
            template='plotly_dark', height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=60, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ─────────── TAB 6: МЕТОДОЛОГИЯ ───────────
    with tabs[5]:
        st.markdown("#### 📋 Методология проекта AI-OSINT")
        
        st.markdown("""
        ##### 1. Архитектура системы
        
        Система AI-OSINT интегрирует четыре аналитических модуля:
        
        | Модуль | Назначение | Технология |
        |--------|-----------|------------|
        | **Data Collection** | Сбор данных из открытых источников | GDELT API, Telethon, RSS |
        | **NLP Pipeline** | Анализ тональности, кластеризация нарративов, NER | XLM-RoBERTa, BERTopic, spaCy |
        | **ABM Engine** | Моделирование информационных кампаний | Агентная модель + Markov Chains |
        | **Monte Carlo** | Стохастическое прогнозирование | N итераций с вариацией параметров |
        
        ##### 2. Агентно-ориентированная модель (ABM)
        
        Модель включает 4 типа агентов, взаимодействующих в сетевой топологии Барабаши-Альберт:
        
        - **Инициатор** (5%): генерирует оригинальный нарратив, высокий импакт, низкая частота
        - **Усилитель** (20%): бот-сети, координированные аккаунты, высокая частота репостов
        - **Медиатор** (10%): СМИ, крупные каналы, легитимизируют нарратив
        - **Реципиент** (65%): обычные пользователи, порог восприимчивости варьируется
        
        ##### 3. Марковские цепи
        
        Нарратив проходит 5 дискретных состояний: **LATENT → EMERGING → GROWING → VIRAL → DECLINING**.
        
        Матрица переходов P(i→j) модифицируется динамически:
        - Рост числа усилителей → увеличение P(переход вправо)
        - Подключение медиаторов → ускорение GROWING → VIRAL
        - Базовые вероятности откалиброваны на данных Stanford IO ("Unheard Voice", 2022) и Meta CIB Reports
        
        ##### 4. Monte Carlo
        
        N симуляций (по умолчанию 1000) с независимой стохастической вариацией:
        - Параметры матрицы переходов: ±5% нормальный шум
        - Доля усилителей: uniform(0, 0.5)
        - Результат: распределение P(VIRAL), среднее число шагов до VIRAL, квантили
        
        ##### 5. Источники данных
        
        | Источник | Описание | URL |
        |----------|----------|-----|
        | GDELT Project | Глобальная база медиа-событий, 300+ категорий | [gdeltproject.org](https://www.gdeltproject.org/) |
        | GDELT DOC API | Полнотекстовый поиск по мировым СМИ | [API](https://api.gdeltproject.org/api/v2/doc/doc?query=Kazakhstan&mode=artlist&format=json) |
        | GDELT BigQuery | SQL-доступ ко всей базе | [BigQuery](https://console.cloud.google.com/marketplace/product/the-gdelt-project/gdelt-2-events) |
        | Stanford IO | "Unheard Voice" — координированные операции в ЦА | [Report](https://cyber.fsi.stanford.edu/io/news/sio-aug-22-takedowns) |
        | IRA Troll Tweets | ~3 млн твитов фабрики троллей (FiveThirtyEight) | [GitHub](https://github.com/fivethirtyeight/russian-troll-tweets) |
        | Meta CIB Reports | Отчёты по координированному недостоверному поведению | [Meta](https://transparency.meta.com/metasecurity/threat-reporting) |
        | Fake News Datasets | Датасеты для обучения NLP-моделей | [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) |
        
        ##### 6. Связь с ПЦФ-грантом и диссертацией
        
        | Аспект | ПЦФ-грант | AI-OSINT (конкурс) | Диссертация |
        |--------|-----------|-------------------|-------------|
        | Фокус | AI-прогнозирование угроз нацбезопасности КЗ | AI-анализ информационного поля КЗ | AI-анализ безопасности в ИТР |
        | Метод | NLP + ABM + OSINT | NLP + ABM + Markov + OSINT | NLP + ABM + Monte Carlo |
        | Связь | Методология | Апробация методологии | Теоретическая рамка |
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #556677; font-size: 0.85rem; padding: 1.5rem;">
            AI-OSINT v0.1α | КазУМОиМЯ имени Абылай хана | 7М02211 — Востоковедение | 2026<br>
            Науч. руководитель: к.полит.н., ассоц. проф. Абсаттаров Г.Р.
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
