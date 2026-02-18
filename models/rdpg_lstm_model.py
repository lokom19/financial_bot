import os
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from utils.load_data_method import load_data


# Disable warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# # Function to load data from SQLite
# def load_data(db_path, table_name):
#     conn = sqlite3.connect(db_path)
#     query = f"SELECT * FROM {table_name} ORDER BY timestamp"
#     df = pd.read_sql_query(query, conn)
#     conn.close()
#     return df


# Function to create technical indicators
def create_features(df):
    df_features = df.copy()

    # Basic price indicators
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

    # Price range indicators
    df_features['high_low_ratio'] = df['high'] / df['low']

    # True Range and ATR
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

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
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


# RDPG-LSTM Model
class RDPGLSTMModel:
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 100,
            'sequence_length': 20,
            'patience': 25,
            'actor_lr': 0.0001,
            'critic_lr': 0.001,
            'gamma': 0.99,  # Discount factor for future rewards
            'tau': 0.001,  # Target network update rate
            'exploration_noise': 0.1
        }

        if params is not None:
            self.params.update(params)

        self.scaler = RobustScaler()
        self.feature_columns = None
        self.last_price = None
        self.data_quality_checked = False
        self.sequence_length = self.params['sequence_length']

        # Build the actor and critic models
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None

        # For storing trading results
        self.trading_history = []
        self.portfolio_value = 1.0  # Start with 1 unit
        self.positions = []  # Track positions taken

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

        # Remove problematic columns
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

    def build_actor_model(self, input_shape):
        """Build actor network (policy network)"""
        states_input = Input(shape=input_shape, name='states_input')

        # LSTM layers for processing sequential data
        x = LSTM(self.params['lstm_units'], return_sequences=True,
                 recurrent_dropout=0.1, name='lstm_actor_1')(states_input)
        x = BatchNormalization()(x)
        x = Dropout(self.params['dropout_rate'])(x)

        x = LSTM(self.params['lstm_units'] // 2,
                 recurrent_dropout=0.1, name='lstm_actor_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.params['dropout_rate'])(x)

        # Dense layers
        x = Dense(64, activation='relu', name='dense_actor_1')(x)
        x = Dropout(self.params['dropout_rate'] / 2)(x)
        x = Dense(32, activation='relu', name='dense_actor_2')(x)

        # Output layer for policy (action)
        # Output is between -1 (full sell) and 1 (full buy)
        actions_output = Dense(1, activation='tanh', name='actions_output')(x)

        # Create model
        model = Model(inputs=states_input, outputs=actions_output)
        model.compile(optimizer=Adam(learning_rate=self.params['actor_lr']), loss='mse')

        return model

    def build_critic_model(self, state_shape, action_shape=1):
        """Build critic network (value network)"""
        # Input for state sequence
        states_input = Input(shape=state_shape, name='critic_states_input')

        # Process state sequence with LSTM
        x = LSTM(self.params['lstm_units'], return_sequences=True,
                 recurrent_dropout=0.1, name='lstm_critic_1')(states_input)
        x = BatchNormalization()(x)
        x = Dropout(self.params['dropout_rate'])(x)

        x = LSTM(self.params['lstm_units'] // 2,
                 recurrent_dropout=0.1, name='lstm_critic_2')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.params['dropout_rate'])(x)

        # Input for action
        action_input = Input(shape=(action_shape,), name='action_input')

        # Combine state and action
        combined = Concatenate()([x, action_input])

        # Dense layers
        x = Dense(64, activation='relu', name='dense_critic_1')(combined)
        x = Dropout(self.params['dropout_rate'] / 2)(x)
        x = Dense(32, activation='relu', name='dense_critic_2')(x)

        # Output layer for Q-value
        q_value_output = Dense(1, name='q_value_output')(x)

        # Create model
        model = Model(inputs=[states_input, action_input], outputs=q_value_output)
        model.compile(optimizer=Adam(learning_rate=self.params['critic_lr']), loss='mse')

        return model

    def create_target_networks(self):
        """Create target networks for actor and critic"""
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_actor.set_weights(self.actor.get_weights())

        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_networks(self):
        """Update target networks using soft update"""
        tau = self.params['tau']

        # Update target actor weights
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()

        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]

        self.target_actor.set_weights(target_actor_weights)

        # Update target critic weights
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()

        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]

        self.target_critic.set_weights(target_critic_weights)

    def calculate_reward(self, action, current_price, next_price):
        """Calculate reward based on action and price change"""
        price_change_pct = (next_price - current_price) / current_price

        # If action aligned with price movement, get positive reward
        # Action is in range [-1, 1] where -1 is full sell, 1 is full buy
        reward = action * price_change_pct * 100  # Scale reward for better gradient

        return reward

    def train(self, X_original, y, timestamps=None, X_scaled=None, df_features=None):
        if np.isnan(X_original.values).any() or np.isinf(X_original.values).any():
            print("CRITICAL ERROR: NaN or Inf found in data.")
            X_original = X_original.replace([np.inf, -np.inf], np.nan).fillna(X_original.median())

        # Use scaled data
        if X_scaled is None:
            X_scaled = self.scaler.fit_transform(X_original)

        # Create sequences for LSTM
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # Train/test split - chronological
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Get price data for calculating rewards
        prices = y.values  # Using next day prices
        current_prices = X_original['close'].values  # Current prices

        # Build actor and critic models
        state_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
        self.actor = self.build_actor_model(state_shape)
        self.critic = self.build_critic_model(state_shape)

        # Create target networks
        self.create_target_networks()

        # Print model summaries
        print("Actor Model:")
        self.actor.summary()
        print("\nCritic Model:")
        self.critic.summary()

        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.params['patience'],
            restore_best_weights=True,
            verbose=1
        )

        replay_buffer = []
        batch_size = self.params['batch_size']
        epochs = self.params['epochs']
        exploration_noise = self.params['exploration_noise']
        gamma = self.params['gamma']

        # RDPG+LSTM Training Loop
        print("\nStarting RDPG+LSTM training...")
        for epoch in range(epochs):
            total_reward = 0
            critic_losses = []
            actor_losses = []

            # Process each sequence
            for i in range(len(X_train)):
                state = X_train[i].reshape(1, X_train.shape[1], X_train.shape[2])

                # Actor selects action with noise for exploration
                action = self.actor.predict(state)[0][0]
                action = np.clip(action + np.random.normal(0, exploration_noise), -1, 1)

                # Get the state index in the original data
                orig_idx = i + self.sequence_length

                # Calculate reward based on the action and price change
                if orig_idx < len(current_prices) - 1:
                    reward = self.calculate_reward(action, current_prices[orig_idx], prices[orig_idx])

                    # Get next state
                    if i < len(X_train) - 1:
                        next_state = X_train[i + 1].reshape(1, X_train.shape[1], X_train.shape[2])
                    else:
                        next_state = state  # Use current state if at the end

                    # Store experience in replay buffer
                    replay_buffer.append((state, action, reward, next_state))
                    total_reward += reward

                # Learn from replay buffer once it has enough samples
                if len(replay_buffer) >= batch_size:
                    # Sample a batch from replay buffer
                    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                    batch = [replay_buffer[idx] for idx in indices]

                    states = np.vstack([exp[0] for exp in batch])
                    actions = np.array([exp[1] for exp in batch]).reshape(-1, 1)
                    rewards = np.array([exp[2] for exp in batch]).reshape(-1, 1)
                    next_states = np.vstack([exp[3] for exp in batch])

                    # Compute target Q values using target networks
                    target_actions = self.target_actor.predict(next_states)
                    target_q_values = self.target_critic.predict([next_states, target_actions])

                    # Compute critic targets: r + gamma * Q'(s', a')
                    critic_targets = rewards + gamma * target_q_values

                    # Train critic
                    critic_loss = self.critic.train_on_batch([states, actions], critic_targets)
                    critic_losses.append(critic_loss)

                    # Train actor using the policy gradient
                    with tf.GradientTape() as tape:
                        pred_actions = self.actor(tf.convert_to_tensor(states, dtype=tf.float32))
                        # Ensure both inputs are tensors
                        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
                        actor_loss = -tf.reduce_mean(self.critic([states_tensor, pred_actions]))

                    actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                    tf.keras.optimizers.Adam(learning_rate=self.params['actor_lr']).apply_gradients(
                        zip(actor_grads, self.actor.trainable_variables))

                    actor_losses.append(float(actor_loss))

                    # Update target networks
                    self.update_target_networks()

            # Track progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                avg_reward = total_reward / len(X_train) if len(X_train) > 0 else 0
                avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
                avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
                print(f"Epoch {epoch + 1}/{epochs}, Avg Reward: {avg_reward:.4f}, "
                      f"Critic Loss: {avg_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}")

        # Evaluate on test data
        print("\n===== Evaluating on test data =====")
        test_rewards = []
        test_actions = []
        test_price_changes = []
        test_preds = []

        # Initial portfolio value
        portfolio_value = 1.0
        position = 0  # Current position

        for i in range(len(X_test)):
            state = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])
            action = self.actor.predict(state)[0][0]  # No noise in evaluation

            orig_idx = i + split_idx + self.sequence_length
            if orig_idx < len(current_prices) - 1:
                current_price = current_prices[orig_idx]
                next_price = prices[orig_idx]
                reward = self.calculate_reward(action, current_price, next_price)

                price_change_pct = (next_price - current_price) / current_price

                # Simulate trading
                # Convert action [-1, 1] to position size change [-1, 1]
                position_delta = action
                new_position = np.clip(position + position_delta, -1, 1)
                position_change = new_position - position

                # Update portfolio value
                if position_change > 0:  # Buying
                    portfolio_value = portfolio_value * (1 + position * price_change_pct)
                    # Reduce cash by the amount bought
                    portfolio_value -= position_change * 0.001  # Trading cost (0.1%)
                elif position_change < 0:  # Selling
                    portfolio_value = portfolio_value * (1 + position * price_change_pct)
                    # Add cash from selling
                    portfolio_value -= abs(position_change) * 0.001  # Trading cost (0.1%)
                else:  # Holding
                    portfolio_value = portfolio_value * (1 + position * price_change_pct)

                position = new_position

                test_rewards.append(reward)
                test_actions.append(action)
                test_price_changes.append(price_change_pct)
                test_preds.append(next_price)

        # Calculate performance metrics
        test_rewards = np.array(test_rewards)
        test_actions = np.array(test_actions)
        test_price_changes = np.array(test_price_changes)

        # Direction accuracy: Did the model predict price direction correctly?
        correct_directions = np.sign(test_actions) == np.sign(test_price_changes)
        direction_accuracy = np.mean(correct_directions) * 100

        # Calculate trading performance metrics
        cumulative_reward = np.sum(test_rewards)
        sharpe_ratio = np.mean(test_rewards) / (np.std(test_rewards) + 1e-9) * np.sqrt(252)  # Annualized

        print(f"Test Direction Accuracy: {direction_accuracy:.2f}%")
        print(f"Cumulative Reward: {cumulative_reward:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Final Portfolio Value: {portfolio_value:.4f} (Initial: 1.0)")

        # Calculate traditional prediction metrics
        if len(test_preds) > 0 and len(y_test) > 0:
            prediction_metrics = self.calculate_metrics(y_test[:len(test_preds)], test_preds)
            print("\n===== Traditional Prediction Metrics =====")
            self.print_metrics(prediction_metrics)

        # Save last price for future predictions
        if np.isnan(y.iloc[-1]) and df_features is not None:
            # Берем последнюю фактическую цену закрытия вместо NaN
            self.last_price = df_features.iloc[-1]['close']
        else:
            self.last_price = y.iloc[-1]

        # Store trading history for analysis
        self.trading_history = {
            'rewards': test_rewards,
            'actions': test_actions,
            'price_changes': test_price_changes,
            'portfolio_value': portfolio_value
        }

        return X_test, y_test, test_preds, {'direction_accuracy': direction_accuracy, 'sharpe_ratio': sharpe_ratio}

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

            mse = mean_squared_error(y_true_filtered, y_pred_filtered)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

            # Безопасное вычисление R² с обработкой исключений
            try:
                r2 = r2_score(y_true_filtered, y_pred_filtered)
            except Exception as e:
                print(f"Ошибка при вычислении R²: {e}")
                r2 = float('nan')

            # Безопасное вычисление MAPE с обработкой деления на ноль
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_raw = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
                mape_raw = np.where(np.isinf(mape_raw) | np.isnan(mape_raw), 0, mape_raw)
                mape = np.mean(mape_raw) * 100


            # Направление цены (бинарный классификатор)
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
            if self.actor is None:
                print("ERROR: Model not trained. Run train() first.")
                return None

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

            # Подготавливаем данные
            features = current_data[self.feature_columns].copy()  # тут последняя строка из датасета, где есть next_close

            # Заменяем бесконечные значения на NaN и затем заполняем медианами
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

            # Get action from actor model (trading signal)
            action = self.actor.predict(input_sequence)[0][0]

            # Get the current price
            current_price = features['close'].iloc[-1]

            # Get predicted Q-value from critic (expected future reward)
            q_value = self.critic.predict([input_sequence, np.array([[action]])])[0][0]

            # Estimate next price using action and current price
            # This is a heuristic, the model doesn't directly predict price but trading actions
            # We can use the critic's Q-value as a signal for price direction
            confidence = abs(action)  # Use absolute action value as confidence

            # Translate action to predicted price direction
            price_change_estimate = action * 0.01 * confidence * current_price  # Estimated 1% change per unit action
            predicted_price = current_price + price_change_estimate

            # Calculate percentage change
            price_change_pct = (predicted_price - current_price) / current_price * 100

            # Determine trading signal based on action
            if action > 0.3:
                signal = 'BUY'
            elif action < -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            # Convert confidence to 0-1 range
            confidence_score = 0.5 + (abs(q_value) / (10.0 + abs(q_value))) * 0.5

            return {
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'price_change_pct': float(price_change_pct),
                'signal': signal,
                'action': float(action),
                'confidence': float(confidence_score),
                'q_value': float(q_value)
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                'current_price': self.last_price if hasattr(self, 'last_price') and self.last_price is not None else 0,
                'predicted_price': self.last_price if hasattr(self,
                                                              'last_price') and self.last_price is not None else 0,
                'price_change_pct': 0.0,
                'signal': 'NEUTRAL',
                'action': 0.0,
                'confidence': 0.0,
                'q_value': 0.0
            }

    def get_action_for_state(self, state):
        """Get an action from the actor for a given state"""
        if self.actor is None:
            print("ERROR: Model not trained. Run train() first.")
            return 0.0

        # Reshape state for the model
        state_reshaped = state.reshape(1, state.shape[0], state.shape[1])

        # Get action from actor
        action = self.actor.predict(state_reshaped)[0][0]
        return action

    def backtest(self, X_scaled, prices, start_value=10000, transaction_cost=0.001):
        """Backtest the model on historical data"""
        if self.actor is None:
            print("ERROR: Model not trained. Run train() first.")
            return None

        # Create sequences for LSTM
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
        X_seq = np.array(X_seq)

        # Initialize backtest variables
        portfolio_value = start_value
        cash = start_value
        position = 0
        shares = 0

        actions = []
        portfolio_values = [portfolio_value]
        positions = [position]
        trades = []
        trade_returns = []

        # Simulate trading
        for i in range(len(X_seq)):
            state = X_seq[i]
            action = self.get_action_for_state(state)
            actions.append(action)

            # Convert action [-1, 1] to position size change [-1, 1]
            target_position = action  # Directly use action as target position
            position_change = target_position - position

            # Current price
            price = prices[i + self.sequence_length]

            # Execute trade if position change
            if abs(position_change) > 0.05:  # Threshold to avoid tiny trades
                # Calculate shares to buy/sell
                if position_change > 0:  # Buy
                    shares_to_trade = (position_change * portfolio_value) / price
                    cost = shares_to_trade * price * (1 + transaction_cost)
                    if cost <= cash:  # Check if we have enough cash
                        shares += shares_to_trade
                        cash -= cost
                        trades.append(('BUY', price, shares_to_trade, cost))
                else:  # Sell
                    shares_to_trade = abs(position_change) * shares
                    revenue = shares_to_trade * price * (1 - transaction_cost)
                    shares -= shares_to_trade
                    cash += revenue
                    trades.append(('SELL', price, shares_to_trade, revenue))

                # Update position
                if shares > 0:
                    position = (shares * price) / portfolio_value
                else:
                    position = 0

                positions.append(position)

            # Update portfolio value
            portfolio_value = cash + (shares * price)
            portfolio_values.append(portfolio_value)

            # Calculate trade return if we closed a position
            if len(trades) >= 2 and trades[-1][0] == 'SELL':
                buy_price = None
                for t in reversed(trades[:-1]):
                    if t[0] == 'BUY':
                        buy_price = t[1]
                        break

                if buy_price is not None:
                    trade_return = (price - buy_price) / buy_price
                    trade_returns.append(trade_return)

        # Calculate performance metrics
        final_return = (portfolio_values[-1] - start_value) / start_value
        sharpe_ratio = np.mean(trade_returns) / (np.std(trade_returns) + 1e-9) * np.sqrt(252)
        max_drawdown = 0
        peak = portfolio_values[0]

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Return backtest results
        return {
            'final_portfolio_value': portfolio_values[-1],
            'total_return': final_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'actions': actions,
            'positions': positions,
            'trades': trades,
            'trade_returns': trade_returns
        }

    def save_model(self, filepath):
        """Save the actor and critic models"""
        if self.actor is None or self.critic is None:
            print("ERROR: Models not initialized. Train models first.")
            return False

        try:
            # Save actor model
            actor_path = f"{filepath}_actor"
            self.actor.save(actor_path)

            # Save critic model
            critic_path = f"{filepath}_critic"
            self.critic.save(critic_path)

            # Save scaler and other parameters
            import pickle
            with open(f"{filepath}_params.pkl", 'wb') as f:
                pickle.dump({
                    'params': self.params,
                    'feature_columns': self.feature_columns,
                    'last_price': self.last_price,
                    'data_quality_checked': self.data_quality_checked,
                    'sequence_length': self.sequence_length
                }, f)

            # Save scaler
            with open(f"{filepath}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)

            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @classmethod
    def load_model(cls, filepath):
        """Load the actor and critic models"""
        try:
            import pickle
            from tensorflow.keras.models import load_model

            # Load parameters
            with open(f"{filepath}_params.pkl", 'rb') as f:
                saved_params = pickle.load(f)

            # Create a new instance with saved parameters
            instance = cls(params=saved_params['params'])

            # Load model attributes
            instance.feature_columns = saved_params['feature_columns']
            instance.last_price = saved_params['last_price']
            instance.data_quality_checked = saved_params['data_quality_checked']
            instance.sequence_length = saved_params['sequence_length']

            # Load scaler
            with open(f"{filepath}_scaler.pkl", 'rb') as f:
                instance.scaler = pickle.load(f)

            # Load actor and critic models
            instance.actor = load_model(f"{filepath}_actor")
            instance.critic = load_model(f"{filepath}_critic")

            # Create target networks
            instance.create_target_networks()

            print(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


# Main function
def main(db_path):
    print("\n\n\n\n\nLoading data...")
    # df = load_data(db_path, db_path.split("/")[2][:-3])
    # df = load_data(db_path, db_path.split("/")[-1][:-3])
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

    print("\nTraining RDPG+LSTM model...")
    model = RDPGLSTMModel()
    X_original, X_scaled, y, timestamps = model.prepare_data(df_features)

    if X_original.shape[0] == 0:
        print(f"ERROR: No records left after data preparation for {db_path}.")
        return None, None

    X_test, y_test, predictions, test_metrics = model.train(X_original, y, timestamps, X_scaled, df_features)

    if X_test is None:  # Check for training error
        return None, None

    print("\nLast 5 predictions vs actual:")
    if len(predictions) >= 5 and len(y_test) >= 5:
        last_indices = range(len(predictions) - 5, len(predictions))
        for i in last_indices:
            if i < len(y_test):
                real_price = y_test[i]
                pred_price = predictions[i]
                error_pct = (pred_price - real_price) / real_price * 100 if not np.isnan(
                    real_price) and real_price != 0 else float('nan')
                print(
                    f"Реальная цена next_close: {real_price:.4f}, Предсказанная цена next_close: {pred_price:.4f}, Ошибка: {error_pct:.2f}%")

    print("\nNext trading interval forecast:")

    # Берем последнюю строку из оригинального датасета
    latest_row = df.iloc[-1:].copy()

    # Создаем признаки только для этой строки
    latest_features = create_features(
        pd.concat([df.iloc[-30:].iloc[:-1], latest_row]))  # Берем предыдущие 30 дней для расчета индикаторов
    latest_features = latest_features.iloc[-1:].copy()  # Оставляем только последнюю строку с рассчитанными признаками

    # Если в latest_features есть NaN в next_close, заменяем его на 0 или другое значение
    if 'next_close' in latest_features.columns and latest_features['next_close'].isna().any():
        latest_features['next_close'] = 0  # или другое подходящее значение

    prediction = model.predict_next(latest_features)


    print(f"Current price: {prediction['current_price']:.4f}")
    print(f"Predicted price: {prediction['predicted_price']:.4f}")
    print(f"Expected change: {prediction['price_change_pct']:.2f}%")
    print(f"Action value: {prediction['action']:.4f}")
    print(f"Q-value: {prediction['q_value']:.4f}")
    print(f"Trading signal: {prediction['signal']}")
    print(f"Confidence: {prediction['confidence']:.2f}")

    # Save the model
    model.save_model("rdpg_lstm_model")

    print("\nRunning backtest...")
    prices = df_features['close'].values
    backtest_results = model.backtest(X_scaled, prices)

    if backtest_results:
        print(f"Final portfolio value: ${backtest_results['final_portfolio_value']:.2f}")
        print(f"Total return: {backtest_results['total_return'] * 100:.2f}%")
        print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.4f}")
        print(f"Maximum drawdown: {backtest_results['max_drawdown'] * 100:.2f}%")
        print(f"Total trades: {len(backtest_results['trades'])}")

        if backtest_results['trade_returns']:
            profitable_trades = sum(1 for r in backtest_results['trade_returns'] if r > 0)
            total_trades = len(backtest_results['trade_returns'])
            print(f"Profitable trades: {profitable_trades} ({profitable_trades / total_trades * 100:.2f}%)")

    return model, df_features


if __name__ == "__main__":
    main("BBG000F6YPH8")