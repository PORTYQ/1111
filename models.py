"""
Обновленный модуль с исправленными моделями машинного обучения
Поддержка новых архитектур и улучшенной обработки данных
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU,
    Bidirectional,
    Input,
    Concatenate,
    Conv1D,
    MaxPooling1D,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import joblib
import logging
from typing import Dict, Tuple, List, Optional
from config import MODEL_CONFIG, LSTM_CONFIG, GRU_CONFIG, DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Установка seed для воспроизводимости
tf.random.set_seed(42)
np.random.seed(42)


class ModelBuilder:
    """Класс для создания и обучения моделей с поддержкой новых архитектур"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.history = {}
        self.metrics = {}
        self.predictions = {}
        self.model_configs = {}

    def build_model(self, model_type: str, input_shape: Tuple[int, int]) -> Model:
        """Универсальный метод создания модели по типу"""
        model_type = model_type.upper()

        if model_type == "LSTM":
            return self.build_lstm_model(input_shape)
        elif model_type == "GRU":
            return self.build_gru_model(input_shape)
        elif model_type == "LSTM_ADVANCED":
            return self.build_advanced_lstm_model(input_shape)
        elif model_type == "TRANSFORMER":
            return self.build_transformer_model(input_shape)
        elif model_type == "CNN_LSTM":
            return self.build_cnn_lstm_hybrid(input_shape)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Создание базовой LSTM модели"""
        model = Sequential()

        model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(0.001),
            )
        )
        model.add(Dropout(0.2))

        model.add(LSTM(units=30, return_sequences=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))

        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1))

        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def build_gru_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Создание базовой GRU модели"""
        model = Sequential()

        model.add(
            GRU(
                units=50,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(0.001),
            )
        )
        model.add(Dropout(0.2))

        model.add(GRU(units=30, return_sequences=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))

        model.add(Dense(20))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1))

        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def build_advanced_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """Улучшенная LSTM модель с Attention и Bidirectional слоями"""
        inputs = Input(shape=input_shape)

        # Bidirectional LSTM слои
        x = Bidirectional(
            LSTM(
                units=128,
                return_sequences=True,
                kernel_regularizer=l2(0.001),
                recurrent_regularizer=l2(0.001),
            )
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Второй Bidirectional LSTM
        lstm_out = Bidirectional(
            LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001))
        )(x)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)

        # Attention механизм
        attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)

        # Объединение attention и LSTM выходов
        combined = Concatenate()([lstm_out, attention_out])
        combined = GlobalAveragePooling1D()(combined)

        # Dense слои с остаточными связями
        dense1 = Dense(128)(combined)
        dense1 = LeakyReLU(alpha=0.01)(dense1)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.2)(dense1)

        dense2 = Dense(64)(dense1)
        dense2 = LeakyReLU(alpha=0.01)(dense2)
        dense2 = BatchNormalization()(dense2)

        # Остаточная связь
        residual = Dense(64)(combined)
        dense2 = tf.keras.layers.Add()([dense2, residual])

        # Выходной слой
        outputs = Dense(1)(dense2)

        model = Model(inputs=inputs, outputs=outputs)

        # Оптимизатор с адаптивной скоростью обучения
        optimizer = AdamW(
            learning_rate=0.001, weight_decay=0.0001, beta_1=0.9, beta_2=0.999
        )

        model.compile(
            optimizer=optimizer,
            loss="huber",  # Более устойчивая к выбросам
            metrics=["mae", "mse"],
        )

        return model

    def build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Transformer-based модель для временных рядов"""
        inputs = Input(shape=input_shape)

        # Позиционное кодирование
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embeddings = tf.keras.layers.Embedding(
            input_dim=input_shape[0], output_dim=input_shape[1]
        )(positions)

        x = inputs + position_embeddings

        # Transformer блоки
        for _ in range(2):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=4, key_dim=input_shape[1] // 4, dropout=0.1
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed forward
            ffn = Sequential(
                [Dense(256, activation="relu"), Dropout(0.1), Dense(input_shape[1])]
            )
            ffn_output = ffn(x)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Выходные слои
        x = GlobalAveragePooling1D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])

        return model

    def build_cnn_lstm_hybrid(self, input_shape: Tuple[int, int]) -> Sequential:
        """Гибридная CNN-LSTM модель"""
        model = Sequential()

        # CNN слои для извлечения локальных паттернов
        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=input_shape,
                padding="same",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        # LSTM слои для временных зависимостей
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(50))
        model.add(Dropout(0.3))

        # Dense слои
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])

        return model

    def prepare_data(
        self, features_df: pd.DataFrame, lookback: int = None
    ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, RobustScaler]:
        """Подготовка данных с правильной нормализацией"""
        if lookback is None:
            lookback = MODEL_CONFIG["lookback_window"]

        # Отдельная нормализация для целевой переменной (цены)
        price_scaler = MinMaxScaler(feature_range=(0, 1))
        price_scaled = price_scaler.fit_transform(features_df[["Close"]].values)

        # Робастная нормализация для остальных признаков
        feature_scaler = RobustScaler()
        feature_cols = [col for col in features_df.columns if col != "Close"]
        features_scaled = feature_scaler.fit_transform(features_df[feature_cols].values)

        # Объединение
        scaled_data = np.column_stack([price_scaled, features_scaled])

        # Создание последовательностей
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback : i])
            y.append(price_scaled[i, 0])  # Только цена закрытия

        return np.array(X), np.array(y), price_scaler, feature_scaler

    def train_model(
        self, crypto_name: str, model_type: str, features_df: pd.DataFrame, **kwargs
    ) -> Dict:
        """Обучение модели с поддержкой новых типов"""
        logger.info(f"Обучение модели {model_type} для {crypto_name}")

        # Параметры
        epochs = kwargs.get("epochs", MODEL_CONFIG.get("epochs", 100))
        batch_size = kwargs.get("batch_size", MODEL_CONFIG.get("batch_size", 32))
        lookback = kwargs.get("lookback", MODEL_CONFIG.get("lookback_window", 40))

        # Подготовка данных
        X, y, price_scaler, feature_scaler = self.prepare_data(features_df, lookback)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Сохранение скейлеров
        model_key = f"{crypto_name}_{model_type}"
        self.scalers[model_key] = price_scaler
        self.feature_scalers[model_key] = feature_scaler

        # Создание модели
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_model(model_type, input_shape)

        # Сохранение конфигурации модели
        self.model_configs[model_key] = {
            "type": model_type,
            "input_shape": input_shape,
            "lookback": lookback,
            "features_count": features_df.shape[1],
        }

        # Callbacks с адаптивными параметрами
        callbacks = self._create_callbacks(model_key, model_type)

        # Обучение
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
        )

        # Сохранение результатов
        self.models[model_key] = model
        self.history[model_key] = history

        # Предсказания и метрики
        results = self._evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            price_scaler,
            features_df,
            lookback,
            model_key,
        )

        # Сохранение артефактов
        self._save_model_artifacts(
            model_key, model, price_scaler, feature_scaler, results["metrics"]
        )

        logger.info(f"Модель {model_key} обучена. Метрики: {results['metrics']}")

        return results

    def _create_callbacks(self, model_key: str, model_type: str) -> List:
        """Создание callbacks с адаптивными параметрами"""
        # Адаптивные параметры в зависимости от типа модели
        patience_map = {
            "LSTM": 20,
            "GRU": 20,
            "LSTM_ADVANCED": 30,
            "TRANSFORMER": 40,
            "CNN_LSTM": 25,
        }

        patience = patience_map.get(model_type.upper(), 20)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(DIRS["models"] / f"{model_key}.h5"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        return callbacks

    def _evaluate_model(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        price_scaler,
        features_df,
        lookback,
        model_key,
    ) -> Dict:
        """Оценка модели и создание предсказаний"""
        # Предсказания
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)

        # Обратное масштабирование только для цен
        train_pred_rescaled = price_scaler.inverse_transform(train_pred)
        test_pred_rescaled = price_scaler.inverse_transform(test_pred)
        y_train_rescaled = price_scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_rescaled = price_scaler.inverse_transform(y_test.reshape(-1, 1))

        # Создание индексов дат
        original_dates = features_df.index[lookback:]
        train_dates = original_dates[: len(train_pred)]
        test_dates = original_dates[len(train_pred) : len(train_pred) + len(test_pred)]

        # Сохранение предсказаний
        predictions = {
            "train_pred": train_pred_rescaled.flatten(),
            "test_pred": test_pred_rescaled.flatten(),
            "y_train": y_train_rescaled.flatten(),
            "y_test": y_test_rescaled.flatten(),
            "train_dates": train_dates,
            "test_dates": test_dates,
        }
        self.predictions[model_key] = predictions

        # Расчет метрик
        metrics = self._calculate_metrics(
            y_test_rescaled.flatten(), test_pred_rescaled.flatten()
        )
        self.metrics[model_key] = metrics

        return {
            "model": model,
            "scaler": price_scaler,
            "feature_scaler": feature_scaler,
            "history": self.history[model_key],
            "metrics": metrics,
            "predictions": predictions,
        }

    def _save_model_artifacts(
        self, model_key, model, price_scaler, feature_scaler, metrics
    ):
        """Сохранение всех артефактов модели"""
        # Сохранение скейлеров
        joblib.dump(price_scaler, DIRS["models"] / f"{model_key}_price_scaler.pkl")
        joblib.dump(feature_scaler, DIRS["models"] / f"{model_key}_feature_scaler.pkl")

        # Сохранение информации о модели
        self.save_model_artifacts(
            model_key.split("_")[0],
            model_key.split("_", 1)[1],
            {"metrics": metrics, "config": self.model_configs.get(model_key, {})},
        )

    def split_data(
        self, X: np.ndarray, y: np.ndarray, split_ratio: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделение данных на обучающую и тестовую выборки"""
        if split_ratio is None:
            split_ratio = MODEL_CONFIG["train_test_split"]

        train_size = int(len(X) * split_ratio)

        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Расчет метрик качества модели"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE с защитой от деления на ноль
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0

        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0

        # Нормализованный RMSE (процент от среднего)
        nrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0

        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "NRMSE": float(nrmse),
            "MAE": float(mae),
            "R2": float(r2),
            "MAPE": float(mape),
            "Directional_Accuracy": float(directional_accuracy),
        }

    def create_forecast(
        self,
        crypto_name: str,
        model_type: str,
        features_df: pd.DataFrame,
        days_ahead: int = 30,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """Создание прогноза на будущее с учетом типа модели"""
        model_key = f"{crypto_name}_{model_type}"

        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не обучена")

        model = self.models[model_key]
        price_scaler = self.scalers[model_key]
        feature_scaler = self.feature_scalers[model_key]

        # Получаем конфигурацию модели
        model_config = self.model_configs.get(model_key, {})
        lookback = model_config.get("lookback", 40)

        # Нормализация
        price_scaled = price_scaler.transform(features_df[["Close"]].values)
        feature_cols = [col for col in features_df.columns if col != "Close"]
        features_scaled = feature_scaler.transform(features_df[feature_cols].values)
        scaled_data = np.column_stack([price_scaled, features_scaled])

        # Последняя последовательность
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, -1)

        # Прогноз
        forecast = []
        current_sequence = last_sequence.copy()

        for _ in range(days_ahead):
            pred = model.predict(current_sequence, verbose=0)
            forecast.append(pred[0, 0])

            # Обновление последовательности
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]

            current_sequence = np.append(
                current_sequence[0, 1:, :], [new_row], axis=0
            ).reshape(1, lookback, -1)

        # Обратное преобразование
        forecast_array = np.array(forecast).reshape(-1, 1)
        forecast_rescaled = price_scaler.inverse_transform(forecast_array).flatten()

        # Даты прогноза
        last_date = features_df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D"
        )

        return forecast_dates, forecast_rescaled

    def ensemble_predict(
        self,
        crypto_name: str,
        features_df: pd.DataFrame,
        days_ahead: int = 30,
        model_types: List[str] = None,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray, Dict]:
        """Ансамблевый прогноз из доступных моделей"""
        if model_types is None:
            # Используем все доступные модели для данной криптовалюты
            model_types = [
                key.split("_", 1)[1]
                for key in self.models.keys()
                if key.startswith(crypto_name)
            ]

        predictions = []
        individual_predictions = {}
        dates = None

        # Получаем прогнозы от каждой модели
        for model_type in model_types:
            model_key = f"{crypto_name}_{model_type}"
            if model_key in self.models:
                try:
                    forecast_dates, forecast_values = self.create_forecast(
                        crypto_name, model_type, features_df, days_ahead
                    )
                    predictions.append(forecast_values)
                    individual_predictions[model_type] = forecast_values
                    dates = forecast_dates
                    logger.info(f"Получен прогноз от модели {model_type}")
                except Exception as e:
                    logger.warning(f"Ошибка получения прогноза от {model_type}: {e}")

        if not predictions:
            raise ValueError(f"Нет доступных моделей для {crypto_name}")

        # Расчет весов на основе метрик
        weights = self._calculate_ensemble_weights(crypto_name, model_types)

        # Взвешенное усреднение
        ensemble_forecast = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_forecast += pred * weight

        logger.info(f"Ансамблевый прогноз из {len(predictions)} моделей")

        # Расчет доверительных интервалов
        predictions_array = np.array(predictions)
        confidence_intervals = {
            "lower_95": np.percentile(predictions_array, 2.5, axis=0),
            "upper_95": np.percentile(predictions_array, 97.5, axis=0),
            "std": np.std(predictions_array, axis=0),
        }

        return (
            dates,
            ensemble_forecast,
            {
                "individual_predictions": individual_predictions,
                "weights": dict(zip(model_types, weights)),
                "confidence_intervals": confidence_intervals,
            },
        )

    def _calculate_ensemble_weights(
        self, crypto_name: str, model_types: List[str]
    ) -> List[float]:
        """Расчет весов для ансамбля на основе метрик"""
        weights = []

        for model_type in model_types:
            model_key = f"{crypto_name}_{model_type}"
            if model_key in self.metrics:
                # Используем комбинацию метрик для расчета веса
                metrics = self.metrics[model_key]
                # Чем меньше RMSE и больше R2, тем больше вес
                score = (1 / (metrics["RMSE"] + 1)) * (metrics["R2"] + 0.1)
                weights.append(score)
            else:
                weights.append(1.0)

        # Нормализация весов
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        return normalized_weights

    def load_model(self, crypto_name: str, model_type: str) -> bool:
        """Загрузка сохраненной модели"""
        model_key = f"{crypto_name}_{model_type}"

        try:
            # Загрузка модели
            model_path = DIRS["models"] / f"{model_key}.h5"
            model = load_model(
                str(model_path),
                custom_objects={
                    "AdamW": AdamW,
                    "MultiHeadAttention": MultiHeadAttention,
                    "LayerNormalization": LayerNormalization,
                },
            )

            # Загрузка скейлеров
            price_scaler_path = DIRS["models"] / f"{model_key}_price_scaler.pkl"
            feature_scaler_path = DIRS["models"] / f"{model_key}_feature_scaler.pkl"

            price_scaler = joblib.load(price_scaler_path)
            feature_scaler = joblib.load(feature_scaler_path)

            self.models[model_key] = model
            self.scalers[model_key] = price_scaler
            self.feature_scalers[model_key] = feature_scaler

            logger.info(f"Модель {model_key} загружена")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_key}: {e}")
            return False

    def get_model_summary(self, crypto_name: str, model_type: str) -> str:
        """Получение описания модели"""
        model_key = f"{crypto_name}_{model_type}"

        if model_key not in self.models:
            return "Модель не найдена"

        model = self.models[model_key]

        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))

        return "\n".join(summary_list)

    def save_model_artifacts(
        self, crypto_name: str, model_type: str, additional_info: Dict = None
    ):
        """Сохранение дополнительных артефактов модели"""
        model_key = f"{crypto_name}_{model_type}"

        if model_key not in self.models:
            logger.warning(f"Модель {model_key} не найдена для сохранения артефактов")
            return

        # Создание информационного файла
        info = {
            "model_name": model_key,
            "architecture": model_type,
            "training_completed": True,
            "metrics": self.metrics.get(model_key, {}),
            "creation_date": pd.Timestamp.now().isoformat(),
            "config": self.model_configs.get(model_key, {}),
        }

        if additional_info:
            info.update(additional_info)

        # Сохранение в JSON
        info_path = DIRS["models"] / f"{model_key}_info.json"
        import json

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, default=str)

        logger.info(f"Артефакты модели {model_key} сохранены")
