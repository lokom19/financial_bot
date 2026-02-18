"""
Main Streamlit Dashboard for Trading Signals.

Run with: streamlit run streamlit_app/app.py
"""
import streamlit as st
import pandas as pd
from datetime import datetime

# Import local modules
import sys
sys.path.insert(0, '..')
from db_utils import (
    get_engine,
    get_latest_signals,
    get_summary_stats,
    get_buy_recommendations,
    get_sell_recommendations
)
from grading import (
    calculate_signal_grade,
    add_grades_to_dataframe,
)

# Page config
st.set_page_config(
    page_title="Trading Signals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .signal-buy {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin-bottom: 10px;
    }
    .signal-sell {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin-bottom: 10px;
    }
    .signal-hold {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin-bottom: 10px;
    }
    .grade-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        color: white;
        margin-left: 10px;
    }
    .grade-A { background-color: #28a745; }
    .grade-B { background-color: #5cb85c; }
    .grade-C { background-color: #f0ad4e; }
    .grade-D { background-color: #d9534f; }
    .grade-F { background-color: #721c24; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .ticker-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)


def render_signal_card(row: pd.Series, signal_type: str):
    """Render a signal card with grade badge."""
    ticker = row['ticker'] or row['figi'][:10]
    change = row['expected_change'] if pd.notna(row['expected_change']) else 0
    consensus = row['consensus_pct'] if pd.notna(row['consensus_pct']) else 0
    r2 = row['avg_r2'] if pd.notna(row['avg_r2']) else 0
    direction = row.get('direction_accuracy') if pd.notna(row.get('direction_accuracy')) else None

    # Calculate grade
    grade_info = calculate_signal_grade(
        consensus_pct=consensus,
        r2=r2,
        direction_accuracy=direction,
    )
    grade = grade_info['overall']
    grade_desc = grade_info['description']

    css_class = 'signal-buy' if signal_type == 'BUY' else 'signal-sell'
    arrow = '▲' if signal_type == 'BUY' else '▼'
    change_color = '#28a745' if signal_type == 'BUY' else '#dc3545'
    change_sign = '+' if change >= 0 else ''

    votes_key = 'buy_votes' if signal_type == 'BUY' else 'sell_votes'
    votes = row.get(votes_key, 0)
    total = row.get('total_models', 0)

    st.markdown(f"""
    <div class="{css_class}">
        <div class="ticker-header">
            <b style="font-size: 20px;">{ticker}</b>
            <span class="grade-badge grade-{grade}" title="{grade_desc}">Грейд {grade}</span>
        </div>
        <span style="color: {change_color}; font-size: 18px; font-weight: bold;">
            {arrow} {change_sign}{change:.2f}%
        </span><br>
        <small style="color: #666;">
            Консенсус: {consensus:.0f}% ({votes}/{total} моделей) |
            R²: {r2:.3f}
        </small><br>
        <small style="color: #888; font-style: italic;">{grade_desc}</small>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.title("📊 Навигация")
    st.sidebar.markdown("---")

    # Grade legend in sidebar
    st.sidebar.markdown("### 📊 Система грейдов")
    st.sidebar.markdown("""
    | Грейд | Надёжность |
    |:-----:|:----------:|
    | **A** | Отличная |
    | **B** | Хорошая |
    | **C** | Средняя |
    | **D** | Низкая |
    | **F** | Очень низкая |
    """)

    with st.sidebar.expander("Подробнее о грейдах"):
        st.markdown("""
**Пороговые значения:**

**Консенсус (вес 35%):**
- A: ≥80%, B: ≥60%, C: ≥40%, D: ≥20%

**R² (вес 25%):**
- A: ≥0.7, B: ≥0.5, C: ≥0.3, D: ≥0.1

**Direction Accuracy (вес 25%):**
- A: ≥65%, B: ≥55%, C: ≥50%, D: ≥45%

**Win Rate (вес 15%):**
- A: ≥60%, B: ≥55%, C: ≥50%, D: ≥45%
        """)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 Рекомендуется следовать сигналам с грейдом **B** и выше")

    # Main title
    st.title("📈 Торговые Сигналы")
    st.markdown("*Рекомендации на основе ML-моделей с системой грейдов надёжности*")

    # Get database connection
    try:
        engine = get_engine()
        stats = get_summary_stats(engine)
    except Exception as e:
        st.error(f"Ошибка подключения к базе данных: {e}")
        st.info("Убедитесь, что PostgreSQL запущен и настроен .env файл")
        return

    # Summary metrics
    st.markdown("### 📊 Общая статистика")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Тикеров", stats['total_tickers'])
    with col2:
        st.metric("Прогнозов", stats['total_predictions'])
    with col3:
        st.metric("Сегодня", stats['predictions_today'])
    with col4:
        st.metric("BUY сигналов", stats['buy_signals_today'], delta_color="normal")
    with col5:
        st.metric("SELL сигналов", stats['sell_signals_today'], delta_color="inverse")

    st.markdown("---")

    # Filters
    st.markdown("### ⚙️ Фильтры")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        min_consensus = st.slider(
            "Минимальный консенсус моделей (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=10,
            help="Минимальный процент моделей, согласных с сигналом"
        )

    with filter_col2:
        min_grade = st.selectbox(
            "Минимальный грейд",
            options=['F', 'D', 'C', 'B', 'A'],
            index=0,
            help="Показывать только сигналы с указанным грейдом или выше"
        )

    with filter_col3:
        sort_by = st.selectbox(
            "Сортировка",
            options=['По грейду', 'По консенсусу', 'По изменению цены'],
            index=0
        )

    grade_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    min_grade_value = grade_order[min_grade]

    st.markdown("---")

    # Two columns for BUY and SELL
    col_buy, col_sell = st.columns(2)

    # BUY Recommendations
    with col_buy:
        st.markdown("### 🟢 Рекомендации к ПОКУПКЕ")

        buy_df = get_buy_recommendations(min_consensus / 100, engine)

        if buy_df.empty:
            st.info("Нет сигналов на покупку с заданными фильтрами")
        else:
            # Add grades and filter
            buy_df = add_grades_to_dataframe(buy_df)
            buy_df['grade_value'] = buy_df['grade'].map(grade_order)
            buy_df = buy_df[buy_df['grade_value'] >= min_grade_value]

            if buy_df.empty:
                st.info(f"Нет сигналов с грейдом {min_grade} или выше")
            else:
                # Sort
                if sort_by == 'По грейду':
                    buy_df = buy_df.sort_values('grade_value', ascending=False)
                elif sort_by == 'По консенсусу':
                    buy_df = buy_df.sort_values('consensus_pct', ascending=False)
                else:
                    buy_df = buy_df.sort_values('expected_change', ascending=False)

                st.caption(f"Найдено: {len(buy_df)} сигналов")

                for _, row in buy_df.iterrows():
                    render_signal_card(row, 'BUY')

    # SELL Recommendations
    with col_sell:
        st.markdown("### 🔴 Рекомендации к ПРОДАЖЕ")

        sell_df = get_sell_recommendations(min_consensus / 100, engine)

        if sell_df.empty:
            st.info("Нет сигналов на продажу с заданными фильтрами")
        else:
            # Add grades and filter
            sell_df = add_grades_to_dataframe(sell_df)
            sell_df['grade_value'] = sell_df['grade'].map(grade_order)
            sell_df = sell_df[sell_df['grade_value'] >= min_grade_value]

            if sell_df.empty:
                st.info(f"Нет сигналов с грейдом {min_grade} или выше")
            else:
                # Sort
                if sort_by == 'По грейду':
                    sell_df = sell_df.sort_values('grade_value', ascending=False)
                elif sort_by == 'По консенсусу':
                    sell_df = sell_df.sort_values('consensus_pct', ascending=False)
                else:
                    sell_df = sell_df.sort_values('expected_change', ascending=True)

                st.caption(f"Найдено: {len(sell_df)} сигналов")

                for _, row in sell_df.iterrows():
                    render_signal_card(row, 'SELL')

    st.markdown("---")

    # All latest signals table
    st.markdown("### 📋 Все актуальные сигналы по моделям")

    signals_df = get_latest_signals(engine)

    if signals_df.empty:
        st.warning("Нет данных о сигналах. Запустите обучение моделей: `make train`")
    else:
        # Add color formatting
        def highlight_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'

        # Format display
        display_df = signals_df.copy()
        display_df['ticker'] = display_df['ticker'].fillna(display_df['figi'].str[:10])

        # Select columns to display
        columns_to_show = [
            'ticker', 'model_name', 'signal', 'current_price',
            'predicted_price', 'expected_change', 'r2', 'direction_accuracy'
        ]
        columns_to_show = [c for c in columns_to_show if c in display_df.columns]

        # Rename columns for display
        column_names = {
            'ticker': 'Тикер',
            'model_name': 'Модель',
            'signal': 'Сигнал',
            'current_price': 'Цена',
            'predicted_price': 'Прогноз',
            'expected_change': 'Изм. %',
            'r2': 'R²',
            'direction_accuracy': 'Точн. направления'
        }

        display_df = display_df[columns_to_show].rename(columns=column_names)

        # Apply styling
        styled_df = display_df.style.applymap(
            highlight_signal,
            subset=['Сигнал']
        ).format({
            'Цена': '{:.2f}',
            'Прогноз': '{:.2f}',
            'Изм. %': '{:+.2f}%',
            'R²': '{:.4f}',
            'Точн. направления': '{:.1f}%'
        }, na_rep='-')

        st.dataframe(styled_df, use_container_width=True, height=400)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 12px;">
        <p>⚠️ <b>Disclaimer:</b> Данные рекомендации носят информационный характер
        и не являются индивидуальной инвестиционной рекомендацией.</p>
        <p>Обновлено: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
