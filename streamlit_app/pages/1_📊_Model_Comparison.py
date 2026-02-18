"""
Model Comparison Page - Compare performance of different ML models.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.insert(0, '..')
from db_utils import get_engine, get_model_comparison

st.set_page_config(
    page_title="Сравнение Моделей",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Сравнение ML Моделей")
st.markdown("*Анализ эффективности различных алгоритмов*")

try:
    engine = get_engine()
    df = get_model_comparison(engine)
except Exception as e:
    st.error(f"Ошибка подключения: {e}")
    st.stop()

if df.empty:
    st.warning("Нет данных для сравнения. Запустите обучение: `make train`")
    st.stop()

# Metrics overview
st.markdown("### Ключевые метрики по моделям")

col1, col2 = st.columns(2)

with col1:
    # R² comparison
    fig_r2 = px.bar(
        df,
        x='model_name',
        y='avg_r2',
        title='Средний R² (коэффициент детерминации)',
        labels={'model_name': 'Модель', 'avg_r2': 'R²'},
        color='avg_r2',
        color_continuous_scale='Greens'
    )
    fig_r2.update_layout(showlegend=False)
    st.plotly_chart(fig_r2, use_container_width=True)

with col2:
    # Direction accuracy comparison
    fig_dir = px.bar(
        df,
        x='model_name',
        y='avg_direction_accuracy',
        title='Точность направления (%)',
        labels={'model_name': 'Модель', 'avg_direction_accuracy': 'Точность %'},
        color='avg_direction_accuracy',
        color_continuous_scale='Blues'
    )
    fig_dir.update_layout(showlegend=False)
    st.plotly_chart(fig_dir, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    # Win rate comparison
    if 'avg_win_rate' in df.columns and df['avg_win_rate'].notna().any():
        fig_win = px.bar(
            df,
            x='model_name',
            y='avg_win_rate',
            title='Средний Win Rate (% прибыльных сделок)',
            labels={'model_name': 'Модель', 'avg_win_rate': 'Win Rate %'},
            color='avg_win_rate',
            color_continuous_scale='Oranges'
        )
        fig_win.update_layout(showlegend=False)
        st.plotly_chart(fig_win, use_container_width=True)
    else:
        st.info("Данные о Win Rate недоступны")

with col4:
    # Profit factor comparison
    if 'avg_profit_factor' in df.columns and df['avg_profit_factor'].notna().any():
        fig_pf = px.bar(
            df,
            x='model_name',
            y='avg_profit_factor',
            title='Средний Profit Factor',
            labels={'model_name': 'Модель', 'avg_profit_factor': 'Profit Factor'},
            color='avg_profit_factor',
            color_continuous_scale='Purples'
        )
        fig_pf.update_layout(showlegend=False)
        st.plotly_chart(fig_pf, use_container_width=True)
    else:
        st.info("Данные о Profit Factor недоступны")

st.markdown("---")

# Signal distribution
st.markdown("### Распределение сигналов по моделям")

signal_data = df[['model_name', 'buy_signals', 'sell_signals', 'hold_signals']].melt(
    id_vars=['model_name'],
    var_name='signal_type',
    value_name='count'
)

signal_data['signal_type'] = signal_data['signal_type'].map({
    'buy_signals': 'BUY',
    'sell_signals': 'SELL',
    'hold_signals': 'HOLD'
})

fig_signals = px.bar(
    signal_data,
    x='model_name',
    y='count',
    color='signal_type',
    title='Количество сигналов по типам',
    labels={'model_name': 'Модель', 'count': 'Количество', 'signal_type': 'Тип сигнала'},
    color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'},
    barmode='group'
)
st.plotly_chart(fig_signals, use_container_width=True)

st.markdown("---")

# Detailed table
st.markdown("### Детальная таблица метрик")

display_df = df.copy()

# Rename columns
column_names = {
    'model_name': 'Модель',
    'tickers_count': 'Тикеров',
    'avg_r2': 'R²',
    'avg_rmse': 'RMSE',
    'avg_mae': 'MAE',
    'avg_direction_accuracy': 'Точность направления',
    'avg_win_rate': 'Win Rate',
    'avg_return': 'Ср. доходность %',
    'avg_profit_factor': 'Profit Factor',
    'buy_signals': 'BUY',
    'sell_signals': 'SELL',
    'hold_signals': 'HOLD'
}

display_df = display_df.rename(columns=column_names)

st.dataframe(
    display_df.style.format({
        'R²': '{:.4f}',
        'RMSE': '{:.4f}',
        'MAE': '{:.4f}',
        'Точность направления': '{:.1f}%',
        'Win Rate': '{:.1f}%',
        'Ср. доходность %': '{:.2f}%',
        'Profit Factor': '{:.2f}'
    }, na_rep='-'),
    use_container_width=True
)

# Recommendations
st.markdown("---")
st.markdown("### 💡 Рекомендации")

if not df.empty:
    best_r2 = df.loc[df['avg_r2'].idxmax()]
    best_direction = df.loc[df['avg_direction_accuracy'].idxmax()] if df['avg_direction_accuracy'].notna().any() else None

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.success(f"""
        **Лучшая модель по R²:** {best_r2['model_name']}
        - R² = {best_r2['avg_r2']:.4f}
        - Модель лучше всего объясняет вариацию цен
        """)

    with col_rec2:
        if best_direction is not None:
            st.info(f"""
            **Лучшая модель по направлению:** {best_direction['model_name']}
            - Точность = {best_direction['avg_direction_accuracy']:.1f}%
            - Лучше предсказывает направление движения
            """)
