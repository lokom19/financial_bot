"""
Ticker Analysis Page - Deep dive into individual ticker predictions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

sys.path.insert(0, '..')
from db_utils import (
    get_engine,
    get_available_tickers,
    get_ticker_history,
    get_price_data
)

st.set_page_config(
    page_title="Анализ Тикера",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Анализ по Тикеру")
st.markdown("*Детальный анализ прогнозов для конкретного инструмента*")

try:
    engine = get_engine()
    tickers_df = get_available_tickers(engine)
except Exception as e:
    st.error(f"Ошибка подключения: {e}")
    st.stop()

if tickers_df.empty:
    st.warning("Нет данных о тикерах. Запустите обучение: `make train`")
    st.stop()

# Ticker selector
st.sidebar.markdown("### Выбор инструмента")

# Create ticker options
ticker_options = {}
for _, row in tickers_df.iterrows():
    label = f"{row['ticker'] or row['figi'][:10]}"
    if row['name']:
        label += f" - {row['name'][:30]}"
    ticker_options[label] = row['figi']

selected_label = st.sidebar.selectbox(
    "Тикер",
    options=list(ticker_options.keys())
)

selected_figi = ticker_options[selected_label]
ticker_info = tickers_df[tickers_df['figi'] == selected_figi].iloc[0]

# Ticker info header
st.markdown(f"## {ticker_info['ticker'] or selected_figi[:10]}")
if ticker_info['name']:
    st.markdown(f"*{ticker_info['name']}*")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("FIGI", selected_figi)
with col2:
    st.metric("Прогнозов", ticker_info['predictions_count'])
with col3:
    if ticker_info['last_prediction']:
        st.metric("Последний прогноз", pd.to_datetime(ticker_info['last_prediction']).strftime("%Y-%m-%d %H:%M"))

st.markdown("---")

# Price chart
st.markdown("### 📈 График цены")

price_df = get_price_data(selected_figi, engine)

if not price_df.empty:
    # Candlestick chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Цена', 'Объем'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_df['timestamp'],
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green'
              for _, row in price_df.iterrows()]

    fig.add_trace(
        go.Bar(
            x=price_df['timestamp'],
            y=price_df['volume'],
            marker_color=colors,
            name='Объем'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Исторические данные цены недоступны")

st.markdown("---")

# Prediction history
st.markdown("### 📊 История прогнозов")

history_df = get_ticker_history(selected_figi, engine)

if history_df.empty:
    st.info("Нет истории прогнозов для этого тикера")
else:
    # Latest predictions by model
    st.markdown("#### Последние прогнозы по моделям")

    latest_by_model = history_df.groupby('model_name').first().reset_index()

    for _, row in latest_by_model.iterrows():
        signal = row['signal']
        model = row['model_name']
        change = row['expected_change']
        r2 = row['r2']

        if signal == 'BUY':
            color = "#d4edda"
            border = "#28a745"
            icon = "🟢"
        elif signal == 'SELL':
            color = "#f8d7da"
            border = "#dc3545"
            icon = "🔴"
        else:
            color = "#fff3cd"
            border = "#ffc107"
            icon = "🟡"

        st.markdown(f"""
        <div style="background-color: {color}; padding: 15px; border-radius: 5px;
                    border-left: 5px solid {border}; margin-bottom: 10px;">
            <b>{icon} {model}</b>: {signal}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Ожидаемое изменение: <b>{change:+.2f}%</b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            R²: {r2:.4f if r2 else 'N/A'}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Цена: {row['current_price']:.2f} → {row['predicted_price']:.2f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Predictions over time
    st.markdown("#### Динамика прогнозов")

    col1, col2 = st.columns(2)

    with col1:
        # Expected change by model over time
        fig_change = px.line(
            history_df,
            x='timestamp',
            y='expected_change',
            color='model_name',
            title='Ожидаемое изменение цены (%)',
            labels={'timestamp': 'Дата', 'expected_change': 'Изменение %', 'model_name': 'Модель'}
        )
        fig_change.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_change, use_container_width=True)

    with col2:
        # R² over time
        if history_df['r2'].notna().any():
            fig_r2 = px.line(
                history_df,
                x='timestamp',
                y='r2',
                color='model_name',
                title='R² по времени',
                labels={'timestamp': 'Дата', 'r2': 'R²', 'model_name': 'Модель'}
            )
            st.plotly_chart(fig_r2, use_container_width=True)

    # Signal distribution
    st.markdown("#### Распределение сигналов")

    signal_counts = history_df['signal'].value_counts()

    fig_pie = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        title='Распределение сигналов',
        color=signal_counts.index,
        color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107', 'NEUTRAL': '#6c757d'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed history table
    st.markdown("#### Детальная история")

    display_df = history_df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

    column_names = {
        'timestamp': 'Время',
        'model_name': 'Модель',
        'signal': 'Сигнал',
        'current_price': 'Текущая цена',
        'predicted_price': 'Прогноз',
        'expected_change': 'Изменение %',
        'r2': 'R²',
        'direction_accuracy': 'Точность',
        'win_rate': 'Win Rate',
        'cumulative_return': 'Доходность %'
    }

    display_df = display_df.rename(columns=column_names)

    def highlight_signal(val):
        if val == 'BUY':
            return 'background-color: #d4edda'
        elif val == 'SELL':
            return 'background-color: #f8d7da'
        return 'background-color: #fff3cd'

    styled = display_df.style.applymap(highlight_signal, subset=['Сигнал'])

    st.dataframe(styled, use_container_width=True, height=400)
