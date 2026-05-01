import os
import sqlite3
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from utils.load_data_method import load_data

# Disable all RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Function to load data from SQLite
# def load_data(db_path, table_name):
#     conn = sqlite3.connect(db_path)
#     query = f"SELECT * FROM {table_name} ORDER BY timestamp"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df


# Function to create technical indicators
def create_features(df):
    df_features = df.copy()

    # Basic technical indicators
    df_features['sma_5'] = df['close'].rolling(window=5).mean()
    df_features['sma_10'] = df['close'].rolling(window=10).mean()
    df_features['sma_20'] = df['close'].rolling(window=20).mean()
    df_features['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df_features['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df_features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Price relative to moving averages
    df_features['close_minus_sma_5'] = df['close'] - df_features['sma_5']
    df_features['close_minus_sma_10'] = df['close'] - df_features['sma_10']
    df_features['close_rel_sma_5'] = df['close'] / df_features['sma_5'] - 1
    df_features['close_rel_sma_10'] = df['close'] / df_features['sma_10'] - 1

    # Volume-based features
    df_features['volume_log'] = np.log1p(df['volume'])
    df_features['volume_sma_5'] = df_features['volume_log'].rolling(window=5).mean()
    df_features['volume_ratio'] = df_features['volume_log'] / df_features['volume_sma_5']

    # Price changes
    df_features['price_change_1'] = df['close'].pct_change(periods=1)
    df_features['price_change_3'] = df['close'].pct_change(periods=3)
    df_features['price_change_5'] = df['close'].pct_change(periods=5)

    # Volatility indicators
    df_features['volatility_5'] = df['close'].rolling(window=5).std() / df_features['sma_5']
    df_features['volatility_10'] = df['close'].rolling(window=10).std() / df_features['sma_10']

    # Range-based indicators
    df_features['high_low_ratio'] = df['high'] / df['low']

    # Calculate True Range and ATR
    df_features['prev_close'] = df['close'].shift(1)
    df_features['tr'] = df_features.apply(
        lambda x: max(
            x['high'] - x['low'],
            abs(x['high'] - x['prev_close']),
            abs(x['low'] - x['prev_close'])
        ) if not pd.isna(x['prev_close']) else np.nan,
        axis=1
    )
    df_features['atr_14'] = df_features['tr'].rolling(window=14).mean()
    df_features = df_features.drop(['prev_close', 'tr'], axis=1)

    # Add RSI indicator
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

    # Add MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema_12 - ema_26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

    # Bollinger Bands
    df_features['bb_middle'] = df['close'].rolling(window=20).mean()
    df_features['bb_std'] = df['close'].rolling(window=20).std()
    df_features['bb_upper'] = df_features['bb_middle'] + (df_features['bb_std'] * 2)
    df_features['bb_lower'] = df_features['bb_middle'] - (df_features['bb_std'] * 2)
    df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
    df_features['bb_pct'] = (df['close'] - df_features['bb_lower']) / (
                df_features['bb_upper'] - df_features['bb_lower'])

    # Target variable - next day's close price
    df_features['next_close'] = df_features['close'].shift(-1)

    # Clean up
    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # df_features = df_features.dropna()
    for col in df_features.columns:
        if col != 'next_close' and df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())

    return df_features


# Improved LSTM Model
class EnhancedLSTMModel:
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 200,
            'sequence_length': 20,  # Increased sequence length
            'patience': 30  # For early stopping
        }

        if params is not None:
            self.params.update(params)

        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.feature_importances = None
        self.data_quality_checked = False
        self.history = None
        self.sequence_length = self.params['sequence_length']

    def check_data_quality(self, features):
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        highly_correlated = [column for column in upper.columns if any(upper[column] > 0.95)]
        if highly_correlated:
            print(f"WARNING: Highly correlated features detected: {highly_correlated}")
            print("This may cause instability in the model. Consider removing some of them.")

        for col in features.columns:
            if features[col].std() / (features[col].mean() + 1e-10) > 100:
                print(f"WARNING: Very high dispersion in column {col}. Clipping outliers.")
                q1 = features[col].quantile(0.01)
                q3 = features[col].quantile(0.99)
                features[col] = features[col].clip(q1, q3)

            if features[col].std() < 1e-6:
                print(f"WARNING: Column {col} has near-zero variance. Consider removing it.")

        self.data_quality_checked = True
        return features

    def prepare_data(self, df):
        timestamps = df['timestamp']
        y = df['next_close']
        features = df.drop(['timestamp', 'next_close'], axis=1)

        if 'volume' in features.columns and 'volume_log' in features.columns:
            features = features.drop(['volume'], axis=1)

        if not self.data_quality_checked:
            features = self.check_data_quality(features)

        features = features.replace([np.inf, -np.inf], np.nan)

        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)

        for col in features.columns:
            q_low = features[col].quantile(0.01)
            q_high = features[col].quantile(0.99)
            features[col] = features[col].clip(lower=q_low, upper=q_high)

        cols_to_drop = [col for col in features.columns if features[col].std() < 1e-8 or features[col].isna().any()]
        if cols_to_drop:
            print(f"Removing columns: {cols_to_drop}")
            features = features.drop(cols_to_drop, axis=1)
            if len(features.columns) == 0:
                raise ValueError("No features left after removing problematic columns")

        self.feature_columns = features.columns

        print("\nFeature statistics after outlier processing:")
        print(features.describe().loc[['min', 'max', 'mean', 'std']].T.head())

        if features.isna().any().any():
            print("WARNING: NaN values remain in the data after processing.")
            features = features.fillna(features.median())

        X_original = features.copy()

        try:
            X_scaled = self.scaler.fit_transform(features)
        except Exception as e:
            print(f"Error during scaling: {e}")
            X_scaled = np.zeros(features.shape)
            for i, col in enumerate(features.columns):
                col_mean = features[col].mean()
                col_std = features[col].std()
                if col_std > 1e-10:
                    X_scaled[:, i] = (features[col] - col_mean) / col_std
                else:
                    X_scaled[:, i] = 0

        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("WARNING: NaN or Inf detected after scaling. Replacing with 0.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        if np.max(np.abs(X_scaled)) > 1e6:
            print("WARNING: Large values after scaling. Clipping.")
            X_scaled = np.clip(X_scaled, -1e6, 1e6)

        return X_original, X_scaled, y, timestamps

    def create_sequences(self, X, y=None):
        """Create sequences for LSTM input"""
        X_seq = []
        y_seq = []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y.iloc[i + self.sequence_length])

        X_seq = np.array(X_seq)

        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq

    def build_enhanced_model(self, input_shape):
        """Build an enhanced LSTM model with more layers and better architecture"""
        model = Sequential()

        # First Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(
            units=self.params['lstm_units'],
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=0.1
        )))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))

        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(
            units=self.params['lstm_units'] // 2,
            return_sequences=True,
            recurrent_dropout=0.1
        )))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))

        # Third LSTM layer
        model.add(LSTM(units=self.params['lstm_units'] // 4))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))

        # Dense layers
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(self.params['dropout_rate'] / 2))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1))

        # Compile with a lower learning rate for better convergence
        optimizer = Adam(learning_rate=self.params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        print(model.summary())
        return model

    def train(self, X_original, y, timestamps=None, X_scaled=None, df_features=None):
        if np.isnan(X_original.values).any() or np.isinf(X_original.values).any():
            print("CRITICAL ERROR: NaN or Inf found in data.")
            X_original = X_original.replace([np.inf, -np.inf], np.nan).fillna(X_original.median())

        # Use scaled data for LSTM
        if X_scaled is None:
            X_scaled = self.scaler.fit_transform(X_original)

        # Create sequences for LSTM
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # Train/test split - keeping chronological order
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Also get original data for metrics calculation
        X_train_orig, X_test_orig = X_original.iloc[
                                    self.sequence_length:split_idx + self.sequence_length], X_original.iloc[
                                                                                            split_idx + self.sequence_length:len(
                                                                                                X_seq) + self.sequence_length]
        y_train_orig, y_test_orig = y.iloc[self.sequence_length:split_idx + self.sequence_length], y.iloc[
                                                                                                   split_idx + self.sequence_length:len(
                                                                                                       X_seq) + self.sequence_length]

        # Timestamps for the test set
        if timestamps is not None:
            timestamps_test = timestamps.iloc[split_idx + self.sequence_length:len(X_seq) + self.sequence_length]

        # Build and train the enhanced model
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_enhanced_model(input_shape)

            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.params['patience'],
                restore_best_weights=True,
                verbose=1
            )

            # Add model checkpoint
            checkpoint_filepath = 'best_lstm_model.h5'
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            )

            # Use a learning rate schedule for better convergence
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )

            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint, lr_scheduler],
                verbose=1
            )

        except Exception as e:
            print(f"Error training enhanced LSTM model: {e}")
            # Fallback to simpler model if there's an error
            try:
                print("Attempting simplified model...")
                self.params['lstm_units'] = 32
                self.params['dropout_rate'] = 0.1
                self.params['batch_size'] = 16

                model = Sequential()
                model.add(LSTM(units=32, return_sequences=False, input_shape=input_shape))
                model.add(Dense(units=1))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

                self.model = model
                self.history = self.model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1
                )
            except Exception as e2:
                print(f"Second error: {e2}")
                return None, None, None, None

        # Predictions
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        if np.isnan(train_preds).any() or np.isinf(train_preds).any():
            print("WARNING: NaN or Inf in train predictions.")
            train_preds = np.nan_to_num(train_preds)

        if np.isnan(test_preds).any() or np.isinf(test_preds).any():
            print("WARNING: NaN or Inf in test predictions.")
            test_preds = np.nan_to_num(test_preds)

        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, train_preds.flatten())
        test_metrics = self.calculate_metrics(y_test, test_preds.flatten())

        print("\n===== Train metrics =====")
        self.print_metrics(train_metrics)
        print("\n===== Test metrics =====")
        self.print_metrics(test_metrics)

        if np.isnan(y.iloc[-1]) and df_features is not None:
            # Берем последнюю фактическую цену закрытия вместо NaN
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]

        return X_test_orig, y_test_orig, test_preds.flatten(), test_metrics


    def calculate_metrics(self, y_true, y_pred):
        try:
            mask = ~np.isnan(y_true)
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]

            if len(y_true_filtered) == 0:
                return {
                    'MSE': float('nan'),
                    'RMSE': float('nan'),
                    'MAE': float('nan'),
                    'R²': float('nan'),
                    'MAPE': float('nan'),
                    'Direction Accuracy': float('nan')
                }

            # mse = mean_squared_error(y_true, y_pred)
            mse = mean_squared_error(y_true_filtered, y_pred_filtered)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except Exception as e:
                print(f"Error calculating R²: {e}")
                r2 = float('nan')

            with np.errstate(divide='ignore', invalid='ignore'):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw)
                mape = np.mean(mape_raw) * 100

            # For direction accuracy, convert to numpy arrays
            if len(y_true_filtered) > 1:
                direction_true = np.diff(np.append(y_true_filtered.iloc[0], y_true_filtered.values)) > 0
                direction_pred = np.diff(np.append(y_true_filtered.iloc[0], y_pred_filtered)) > 0
                direction_accuracy = np.mean(direction_true == direction_pred) * 100
            else:
                direction_accuracy = float('nan')

            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'Direction Accuracy': direction_accuracy
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {k: float('nan') for k in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Direction Accuracy']}


    def print_metrics(self, metrics):
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"{name}: Error calculating")
            else:
                print(f"{name}: {value:.4f}")


    def predict_next(self, current_data):
        try:
            current_price = current_data['close'].iloc[0]

            missing_cols = []
            for col in self.feature_columns:
                if col not in current_data.columns:
                    if col == 'volume_log' and 'volume' in current_data.columns:
                        # Создаем логарифмированную версию, если она отсутствует
                        current_data['volume_log'] = np.log1p(current_data['volume'])
                    else:
                        missing_cols.append(col)


            if missing_cols:
                print(f"ВНИМАНИЕ: В данных отсутствуют колонки: {missing_cols}")
                # Создаем отсутствующие колонки с нулями
                for col in missing_cols:
                    current_data[col] = 0

            features = current_data[self.feature_columns].copy()
            features = features.replace([np.inf, -np.inf], np.nan)

            for col in features.columns:
                if features[col].isna().any():
                    features[col] = features[col].fillna(features[col].median())

            for col in features.columns:
                q_low = features[col].quantile(0.01) if len(features) > 10 else features[col].min()
                q_high = features[col].quantile(0.99) if len(features) > 10 else features[col].max()
                features[col] = features[col].clip(lower=q_low, upper=q_high)

            # Scale the features
            scaled_features = self.scaler.transform(features)

            # Need at least sequence_length data points for prediction
            if len(scaled_features) < self.sequence_length:
                print(f"WARNING: Not enough data points for prediction. Need at least {self.sequence_length} points.")
                # Pad with the same values if not enough data
                padding = np.tile(scaled_features[0], (self.sequence_length - len(scaled_features), 1))
                scaled_features = np.vstack([padding, scaled_features])

            # Take the last sequence_length data points
            input_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length,
                                                                             scaled_features.shape[1])

            # Predict
            predicted_price = self.model.predict(input_sequence)[0][0]

            if np.isnan(predicted_price) or np.isinf(predicted_price):
                print("WARNING: Predicted price is NaN or Inf. Using current price.")
                predicted_price = current_price

            price_change = (predicted_price - current_price) / current_price * 100

            # Limit extreme predictions
            if abs(price_change) > 10:
                print(f"WARNING: Price change ({price_change:.2f}%) too large. Limiting to ±10%.")
                price_change = np.sign(price_change) * 10
                predicted_price = current_price * (1 + price_change / 100)

            # Simple confidence score based on validation loss
            confidence = 0.0
            if self.history is not None:
                val_losses = self.history.history['val_loss']
                last_val_loss = val_losses[-1]
                confidence = max(0, min(1, 1 / (1 + last_val_loss)))

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change,
                'signal': 'BUY' if price_change > 0 else 'SELL',
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'current_price': self.last_price if hasattr(self, 'last_price') and self.last_price is not None else 0,
                'predicted_price': self.last_price if hasattr(self,
                                                              'last_price') and self.last_price is not None else 0,
                'price_change_pct': 0.0,
                'signal': 'NEUTRAL',
                'confidence': 0.0
            }


# Main function
def main(db_path):
    print("\n\n\n\n\nLoading data...")
    # table_name = os.path.basename(db_path)[:-3]  # Use file name without .db
    # df = load_data(db_path, table_name)

    df = load_data(db_path)
    df = df.drop(["figi"], axis=1)
    print(
        f"Loaded {len(df)} records for the period from {df['timestamp'].min() if not df.empty else 'N/A'} to {df['timestamp'].max() if not df.empty else 'N/A'}")

    if df.empty:
        print(f"ERROR: File {db_path} does not contain data.")
        return None, None

    print("\nCreating features...")
    df_features = create_features(df)
    print(f"Features created: {len(df_features.columns) - 2}")

    if df_features.empty:
        print(f"ERROR: No data left after creating features for {db_path}.")
        return None, None

    print("\nFeature statistics:")
    print(df_features.describe().T[['mean', 'min', 'max', 'std']])

    inf_check = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    if inf_check > 0:
        print(f"WARNING: {inf_check} infinite values detected.")

    print("\nTraining Enhanced LSTM model...")
    model = EnhancedLSTMModel()
    X_original, X_scaled, y, timestamps = model.prepare_data(df_features)

    if X_original.shape[0] == 0:
        print(f"ERROR: No records left after data preparation for {db_path}.")
        return None, None

    X_test, y_test, predictions, test_metrics = model.train(X_original, y, timestamps, X_scaled, df_features)

    if X_test is None:  # Check for training error
        return None, None

    print("\nLast 5 predictions:")
    last_indices = range(len(y_test) - 5, len(y_test))
    for i in last_indices:
        real_price = y_test.iloc[i]
        pred_price = predictions[i]
        error_pct = (pred_price - real_price) / real_price * 100 if not np.isnan(
            real_price) and real_price != 0 else float('nan')
        print(f"Реальная цена next_close: {real_price:.4f}, Предсказанная цена next_close: {pred_price:.4f}, Ошибка: {error_pct:.2f}%")

    print("\nПрогноз на следующий временной интервал:")
    latest_row = df.iloc[-1:].copy()

    # Создаем признаки только для этой строки
    latest_features = create_features(
        pd.concat([df.iloc[-30:].iloc[:-1], latest_row]))  # Берем предыдущие 30 дней для расчета индикаторов
    latest_features = latest_features.iloc[-1:].copy()  # Оставляем только последнюю строку с рассчитанными признаками

    # Если в latest_features есть NaN в next_close, заменяем его на 0 или другое значение
    if 'next_close' in latest_features.columns and latest_features['next_close'].isna().any():
        latest_features['next_close'] = 0  # или другое подходящее значение

    # last_data = df_features.iloc[-1:].copy()    prediction = model.predict_next(last_data)
    prediction = model.predict_next(latest_features)

    print(f"Текущая цена: {prediction['current_price']:.4f}")
    print(f"Прогнозируемая цена: {prediction['predicted_price']:.4f}")
    print(f"Ожидаемое изменение: {prediction['price_change_pct']:.2f}%")
    print(f"Торговый сигнал: {prediction['signal']}")

    print("\nРетроспективная оценка торговых сигналов:")
    valid_mask = ~np.isnan(y_test)
    y_test_valid = y_test[valid_mask]
    predictions_valid = predictions[valid_mask]

    if len(y_test_valid) > 1:
        y_test_shifted = y_test_valid.shift(1).fillna(y_test_valid.iloc[0])
        signals = np.sign(np.clip(predictions_valid - y_test_shifted, -1e10, 1e10))
        actual_returns = y_test_valid.pct_change().fillna(0)
        strategy_returns = signals[:-1] * actual_returns[1:].values
        cumulative_returns = (1 + strategy_returns).cumprod() - 1
        total_trades = np.sum(np.abs(np.diff(signals)) > 0) + 1
        profitable_trades = np.sum(strategy_returns > 0)
        profit_sum = np.sum(strategy_returns[strategy_returns > 0])
        loss_sum = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')

        # Sharpe Ratio (annualized) — filter NaN
        clean_returns = strategy_returns[~np.isnan(strategy_returns)]
        if len(clean_returns) > 1:
            sharpe_ratio = np.mean(clean_returns) / (np.std(clean_returns) + 1e-9) * np.sqrt(252)
            equity_curve = (1 + clean_returns).cumprod()
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdowns)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        print(f"Всего сделок: {total_trades}")
        print(f"Прибыльных сделок: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}% от общего числа)")
        print(f"Общая доходность: {cumulative_returns.iloc[-1] * 100:.2f}%")
        print(f"Коэффициент прибыли (Profit Factor): {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Максимальная просадка: {max_drawdown * 100:.2f}%")
    else:
        print("Недостаточно данных для оценки эффективности торговых сигналов")

    return model, df_features


if __name__ == "__main__":
    main("BBG000Q7ZZY2")