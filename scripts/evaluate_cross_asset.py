#!/usr/bin/env python3
"""
A/B тест: обучает модель дважды — с cross-asset фичами и без — сравнивает
direction accuracy. НЕ пишет в БД, не влияет на прод.

Использует catboost (быстрая и точная модель) на всех топ-тикерах.

Запуск:
    docker exec arima_scheduler python3 scripts/evaluate_cross_asset.py
    docker exec arima_scheduler python3 scripts/evaluate_cross_asset.py --model xgboost --ticker GAZP
"""
import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger("eval_ca")

TICKERS = ["SBER", "VTBR", "OZON", "YDEX", "MTSS", "GAZP", "AFLT"]


def _get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'postgres')}"
    )
    return create_engine(url)


def train_and_evaluate(ticker: str, figi: str, use_cross_asset: bool,
                       model_type: str = "catboost") -> dict:
    """
    Обучает модель, возвращает {direction_accuracy, mae, top_features}.
    """
    from utils.load_data_method import load_data
    from core.feature_engineering import create_features
    from core.data_pipeline import DataPipeline

    df = load_data(figi, add_fear_greed=False, add_cross_asset=use_cross_asset)
    if df.empty or len(df) < 100:
        return {"error": f"мало данных ({len(df)})"}

    # Считаем фичи через центральную функцию
    df_feat = create_features(df)
    df_feat = df_feat.dropna(subset=['next_return'])

    if len(df_feat) < 100:
        return {"error": f"после features мало данных ({len(df_feat)})"}

    # 80/20 chronological split (последние 20% на тест)
    pipeline = DataPipeline(test_size=0.2, target_col='next_return')
    split = pipeline.prepare_data(df_feat, shuffle=False)

    X_train = split.X_train.values
    X_test = split.X_test.values
    y_train = split.y_train.values
    y_test = split.y_test.values
    feature_names = split.X_train.columns.tolist()

    # Обучение
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            verbose=False, allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importances = dict(zip(feature_names, model.get_feature_importance()))
    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importances = dict(zip(feature_names, model.feature_importances_))
    else:
        return {"error": f"неизвестная модель {model_type}"}

    # Direction accuracy: предсказали ли мы знак движения
    y_test_sign = np.sign(y_test)
    y_pred_sign = np.sign(y_pred)
    valid = y_test_sign != 0
    if valid.sum() == 0:
        return {"error": "no valid samples"}
    dir_acc = (y_test_sign[valid] == y_pred_sign[valid]).mean() * 100

    # MAE
    mae = np.abs(y_test - y_pred).mean()

    # Top-10 features
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:10]

    return {
        "direction_accuracy": round(dir_acc, 1),
        "mae": round(float(mae), 4),
        "n_features": len(feature_names),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "top_features": top_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="catboost", choices=["catboost", "xgboost"])
    parser.add_argument("--ticker", default=None,
                        help="Конкретный тикер (по умолчанию — все)")
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать top-10 фичей")
    args = parser.parse_args()

    engine = _get_engine()

    tickers = [args.ticker] if args.ticker else TICKERS

    # FIGI по тикерам
    figis = {}
    with engine.connect() as conn:
        for t in tickers:
            r = conn.execute(text(
                "SELECT figi FROM public.tickers WHERE ticker = :t LIMIT 1"
            ), {"t": t}).fetchone()
            if r:
                figis[t] = r[0]
            else:
                print(f"⚠ FIGI не найден для {t}")

    print(f"\n{'='*75}")
    print(f"A/B тест cross-asset фичей на {args.model} ({len(figis)} тикеров)")
    print(f"{'='*75}")
    print(f"{'Ticker':8s} {'без_CA':>10s} {'с_CA':>10s} {'Δ dir%':>10s} {'n_feat':>10s}")
    print("-" * 60)

    totals = {"without": [], "with": []}

    for ticker, figi in figis.items():
        without = train_and_evaluate(ticker, figi, use_cross_asset=False,
                                      model_type=args.model)
        with_ca = train_and_evaluate(ticker, figi, use_cross_asset=True,
                                      model_type=args.model)

        if "error" in without or "error" in with_ca:
            err = without.get("error") or with_ca.get("error")
            print(f"{ticker:8s} {err}")
            continue

        dir_wo = without["direction_accuracy"]
        dir_w = with_ca["direction_accuracy"]
        delta = dir_w - dir_wo
        marker = "🟢" if delta > 1 else ("🔴" if delta < -1 else "  ")
        print(f"{ticker:8s} {dir_wo:9.1f}% {dir_w:9.1f}% {delta:+9.1f}% "
              f"{without['n_features']:4d}→{with_ca['n_features']:4d}  {marker}")

        totals["without"].append(dir_wo)
        totals["with"].append(dir_w)

        if args.verbose:
            print(f"  Top-10 фичей с CA:")
            for f, imp in with_ca["top_features"]:
                is_ca = any(f.startswith(p) for p in ('brent_', 'usd_rub_', 'imoex_'))
                mark = "★" if is_ca else " "
                print(f"    {mark} {f:35s} {imp:.4f}")
            print()

    if totals["without"]:
        print("-" * 60)
        avg_wo = sum(totals["without"]) / len(totals["without"])
        avg_w = sum(totals["with"]) / len(totals["with"])
        print(f"{'AVG':8s} {avg_wo:9.1f}% {avg_w:9.1f}% {avg_w - avg_wo:+9.1f}%")

    print()


if __name__ == "__main__":
    main()
