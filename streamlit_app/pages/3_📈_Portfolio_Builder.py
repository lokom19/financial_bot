"""
Portfolio Builder Page - Build portfolio based on model recommendations.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, '..')
from db_utils import get_engine, get_latest_signals, get_buy_recommendations

st.set_page_config(
    page_title="Конструктор Портфеля",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Конструктор Портфеля")
st.markdown("*Создайте портфель на основе рекомендаций моделей*")

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Ошибка подключения: {e}")
    st.stop()

# Sidebar settings
st.sidebar.markdown("### Настройки портфеля")

portfolio_size = st.sidebar.number_input(
    "Размер портфеля (₽)",
    min_value=10000,
    max_value=100000000,
    value=100000,
    step=10000
)

max_positions = st.sidebar.slider(
    "Максимум позиций",
    min_value=1,
    max_value=20,
    value=5
)

min_consensus = st.sidebar.slider(
    "Минимальный консенсус моделей (%)",
    min_value=50,
    max_value=100,
    value=60
)

risk_per_position = st.sidebar.slider(
    "Риск на позицию (%)",
    min_value=5,
    max_value=50,
    value=20
)

st.markdown("---")

# Get recommendations
buy_df = get_buy_recommendations(min_consensus / 100, engine)

if buy_df.empty:
    st.warning(f"""
    Нет рекомендаций с консенсусом >= {min_consensus}%.

    Попробуйте:
    - Снизить минимальный консенсус
    - Запустить обучение моделей: `make train`
    """)
    st.stop()

# Portfolio composition
st.markdown("### 🎯 Рекомендуемый портфель")

# Calculate position sizes
position_size = portfolio_size * (risk_per_position / 100)
max_per_position = portfolio_size / max_positions

# Select top recommendations
portfolio_df = buy_df.head(max_positions).copy()
portfolio_df['position_size'] = min(position_size, max_per_position)
portfolio_df['weight'] = portfolio_df['position_size'] / portfolio_size * 100

# Display portfolio
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Состав портфеля")

    for i, row in portfolio_df.iterrows():
        ticker = row['ticker'] or row['figi'][:10]
        change = row['expected_change']
        consensus = row['consensus_pct']
        pos_size = row['position_size']
        weight = row['weight']

        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 15px; border-radius: 8px;
                    border-left: 5px solid #28a745; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <b style="font-size: 20px;">{ticker}</b><br>
                    <small>Консенсус: {consensus:.0f}% | Ожидаемый рост: +{change:.2f}%</small>
                </div>
                <div style="text-align: right;">
                    <b style="font-size: 18px;">{pos_size:,.0f} ₽</b><br>
                    <small>{weight:.1f}% портфеля</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Pie chart
    fig = px.pie(
        portfolio_df,
        values='position_size',
        names=portfolio_df['ticker'].fillna(portfolio_df['figi'].str[:10]),
        title='Распределение портфеля',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Expected returns
st.markdown("### 💰 Ожидаемая доходность")

portfolio_df['expected_profit'] = portfolio_df['position_size'] * portfolio_df['expected_change'] / 100
total_expected_profit = portfolio_df['expected_profit'].sum()
total_invested = portfolio_df['position_size'].sum()
expected_return_pct = (total_expected_profit / total_invested * 100) if total_invested > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Инвестировано", f"{total_invested:,.0f} ₽")

with col2:
    st.metric("Остаток", f"{portfolio_size - total_invested:,.0f} ₽")

with col3:
    st.metric(
        "Ожидаемая прибыль",
        f"{total_expected_profit:,.0f} ₽",
        delta=f"+{expected_return_pct:.2f}%"
    )

with col4:
    avg_consensus = portfolio_df['consensus_pct'].mean()
    st.metric("Средний консенсус", f"{avg_consensus:.0f}%")

st.markdown("---")

# Risk analysis
st.markdown("### ⚠️ Анализ рисков")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Факторы риска:**

    - **Модельный риск**: Прогнозы основаны на исторических данных
    - **Рыночный риск**: Внешние факторы могут повлиять на цены
    - **Ликвидность**: Возможны сложности с быстрой продажей

    **Рекомендации:**
    - Используйте стоп-лоссы (-5% от цены входа)
    - Диверсифицируйте по секторам
    - Не инвестируйте больше, чем готовы потерять
    """)

with col2:
    # Risk metrics
    worst_case = -total_invested * 0.1  # 10% drawdown
    best_case = total_expected_profit * 1.5

    scenarios = pd.DataFrame({
        'Сценарий': ['Оптимистичный', 'Базовый', 'Пессимистичный'],
        'Результат': [best_case, total_expected_profit, worst_case],
        'Доходность %': [
            best_case / total_invested * 100,
            expected_return_pct,
            worst_case / total_invested * 100
        ]
    })

    fig = px.bar(
        scenarios,
        x='Сценарий',
        y='Результат',
        color='Результат',
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Сценарии доходности'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Detailed table
st.markdown("### 📋 Детали позиций")

display_df = portfolio_df.copy()
display_df['ticker'] = display_df['ticker'].fillna(display_df['figi'].str[:10])

columns = ['ticker', 'consensus_pct', 'expected_change', 'avg_r2', 'position_size', 'weight', 'expected_profit']
display_df = display_df[columns]

display_df.columns = ['Тикер', 'Консенсус %', 'Ожид. изменение %', 'R²', 'Размер позиции ₽', 'Вес %', 'Ожид. прибыль ₽']

st.dataframe(
    display_df.style.format({
        'Консенсус %': '{:.0f}%',
        'Ожид. изменение %': '+{:.2f}%',
        'R²': '{:.4f}',
        'Размер позиции ₽': '{:,.0f}',
        'Вес %': '{:.1f}%',
        'Ожид. прибыль ₽': '{:+,.0f}'
    }),
    use_container_width=True
)

# Disclaimer
st.markdown("---")
st.warning("""
**⚠️ Важно:**
- Данные рекомендации носят информационный характер
- Не являются индивидуальной инвестиционной рекомендацией
- Прошлые результаты не гарантируют будущую доходность
- Проконсультируйтесь с финансовым советником перед инвестированием
""")
