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
import shap
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l1, l1_l2
from data_preprocessing import DataPreprocessor

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
        self.overfitting_history = {}
        self.shap_values = {}
        self.preprocessor = DataPreprocessor()

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
        model.add(GaussianNoise(0.01, input_shape=input_shape))

        model.add(
            LSTM(
                units=50,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                recurrent_regularizer=l2(0.001),
                dropout=0.1,
                recurrent_dropout=0.1,
            )
        )
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(
            LSTM(
                units=30,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                dropout=0.1,
                recurrent_dropout=0.1,
            )
        )
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(20, kernel_regularizer=l2(0.001)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.1))
        model.add(Dense(10, kernel_regularizer=l2(0.001)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.00005, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="huber", metrics=["mae"])
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

    def prepare_data(self, features_df: pd.DataFrame, lookback: int = None) -> Tuple[np.ndarray, np.ndarray, DataPreprocessor]:
        """НОВАЯ версия подготовки данных с правильной предобработкой"""
        if lookback is None:
            lookback = MODEL_CONFIG["lookback_window"]
        # Используем новый препроцессор
        logger.info("Применение улучшенной предобработки данных...")
        X_processed, y = self.preprocessor.fit_transform(features_df)
        # Добавляем целевую переменную обратно для создания последовательностей
        data_combined = pd.concat([pd.Series(y, index=X_processed.index, name='Close'), X_processed], axis=1)
        # Создание последовательностей
        X, y = [], []
        for i in range(lookback, len(data_combined)):
            X.append(data_combined.iloc[i-lookback:i].values)
            y.append(data_combined.iloc[i, 0])  # Close price    

        return np.array(X), np.array(y), self.preprocessor

    def train_model(self, crypto_name: str, model_type: str, features_df: pd.DataFrame, **kwargs) -> Dict:
        """Обучение модели с ПРАВИЛЬНОЙ валидацией временных рядов"""
        logger.info(f"Обучение модели {model_type} для {crypto_name}")

        # Параметры
        epochs = kwargs.get("epochs", MODEL_CONFIG.get("epochs", 100))
        batch_size = kwargs.get("batch_size", MODEL_CONFIG.get("batch_size", 32))
        lookback = kwargs.get("lookback", MODEL_CONFIG.get("lookback_window", 40))

        # Подготовка данных
        X, y, preprocessor = self.prepare_data(features_df, lookback)

        # Сохранение скейлеров
        model_key = f"{crypto_name}_{model_type}"
        self.scalers[model_key] = preprocessor
        # ПРАВИЛЬНАЯ валидация временных рядов
        splits = preprocessor.get_time_series_splits(X, y, n_splits=5, gap=5)
        # Используем последний сплит для финального обучения
        train_idx, test_idx = splits[-1]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")
        logger.info(f"Количество признаков после отбора: {X_train.shape[2]}")
        # Создание модели
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_model(model_type, input_shape)
        # Сохранение конфигурации
        self.model_configs[model_key] = {
            "type": model_type,
            "input_shape": input_shape,
            "lookback": lookback,
            "features_count": X_train.shape[2],
            "selected_features": preprocessor.selected_features
        }
        # Callbacks
        callbacks = self._create_callbacks(model_key, model_type)
        # КРОСС-ВАЛИДАЦИЯ для оценки переобучения
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(splits[:-1]):  # Используем все кроме последнего
            logger.info(f"Кросс-валидация Fold {fold+1}/{len(splits)-1}")
            X_cv_train, X_cv_val = X[train_idx], X[val_idx]
            y_cv_train, y_cv_val = y[train_idx], y[val_idx]
            # Клонируем модель для CV
            cv_model = self.build_model(model_type, input_shape)
            history = cv_model.fit(
                X_cv_train, y_cv_train,
                validation_data=(X_cv_val, y_cv_val),
                epochs=min(50, epochs),  # Меньше эпох для CV
                batch_size=batch_size,
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            val_loss = min(history.history['val_loss'])
            cv_scores.append(val_loss)
            logger.info(f"Fold {fold+1} val_loss: {val_loss:.4f}")
        avg_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        logger.info(f"CV Score: {avg_cv_score:.4f} (+/- {std_cv_score:.4f})")
        # Финальное обучение на последнем сплите
        logger.info("Финальное обучение модели...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Важно для временных рядов!
        )
        # Сохранение результатов
        self.models[model_key] = model
        self.history[model_key] = history
        # Оценка модели
        results = self._evaluate_model(
            model, X_train, X_test, y_train, y_test,
            preprocessor, features_df, lookback, model_key
        )
        # Добавляем CV результаты   
        results['cv_scores'] = cv_scores
        results['cv_mean'] = avg_cv_score
        results['cv_std'] = std_cv_score
        # Сохранение
        self._save_model_artifacts(model_key, model, preprocessor, results["metrics"])
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
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.check_overfitting(
                    epoch, logs, model_key
                )
            ),
        ]

        return callbacks

    def check_overfitting(self, epoch, logs, model_key):
        """Проверка на переобучение"""
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if val_loss and train_loss:
            overfitting_ratio = val_loss / train_loss
            if overfitting_ratio > 1.5:
                logger.warning(
                    f"⚠️ Модель {model_key} - возможное переобучение на эпохе {epoch}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"ratio={overfitting_ratio:.2f}"
                )
            if model_key not in self.overfitting_history:
                self.overfitting_history[model_key] = []
            self.overfitting_history[model_key].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "overfitting_ratio": overfitting_ratio,
                }
            )

    def calculate_shap_values(
        self, model_key: str, X_test: np.ndarray, feature_names: List[str] = None
    ) -> Dict:
        """Расчет SHAP значений для интерпретации модели"""
        if model_key not in self.models:
            logger.error(f"Модель {model_key} не найдена")
            return {}
        model = self.models[model_key]
        try:
            background = X_test[:100]
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_test[:100])
            self.shap_values[model_key] = {
                "values": shap_values,
                "expected_value": explainer.expected_value,
                "feature_names": feature_names,
                "data": X_test[:100],
            }
            # Анализ важности признаков
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            # Средняя важность по всем примерам
            mean_shap = np.abs(shap_values).mean(axis=0)
            # Важность по временным шагам
            time_importance = mean_shap.mean(axis=1)
            feature_importance = mean_shap.mean(axis=0)
            result = {
                "time_importance": time_importance.tolist(),
                "feature_importance": feature_importance.tolist(),
                "top_features": self._get_top_features(
                    feature_importance, feature_names
                ),
                "model_key": model_key,
            }
            logger.info(f"SHAP анализ выполнен для {model_key}")
            return result
        except Exception as e:
            logger.error(f"Ошибка при расчете SHAP для {model_key}: {e}")
            return {}

    def _get_top_features(
        self,
        importance_values: np.ndarray,
        feature_names: List[str] = None,
        top_n: int = 10,
    ) -> List[Dict]:
        """Получение топ важных признаков"""
        indices = np.argsort(importance_values)[-top_n:][::-1]
        top_features = []
        for idx in indices:
            feature_info = {
                "index": int(idx),
                "importance": float(importance_values[idx]),
            }
            if feature_names and idx < len(feature_names):
                feature_info["name"] = feature_names[idx]
            top_features.append(feature_info)
        return top_features
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test,
                   preprocessor, features_df, lookback, model_key) -> Dict:
        """Оценка модели с SHAP на отобранных признаках"""
        # Предсказания
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        # Обратное масштабирование для цен
        # Так как y уже в исходном масштабе, просто используем его
        train_pred_rescaled = train_pred.flatten()
        test_pred_rescaled = test_pred.flatten()
        y_train_rescaled = y_train
        y_test_rescaled = y_test
        # Создание индексов дат
        original_dates = features_df.index[lookback:]
        train_dates = original_dates[:len(train_pred)]
        test_dates = original_dates[len(train_pred):len(train_pred) + len(test_pred)]
        # Сохранение предсказаний
        predictions = {
            "train_pred": train_pred_rescaled,
            "test_pred": test_pred_rescaled,
            "y_train": y_train_rescaled,
            "y_test": y_test_rescaled,
            "train_dates": train_dates,
            "test_dates": test_dates,
        }
        self.predictions[model_key] = predictions
        # Расчет метрик
        metrics = self._calculate_metrics(y_test_rescaled, test_pred_rescaled)
        self.metrics[model_key] = metrics
        # SHAP анализ на ОТОБРАННЫХ признаках
        logger.info("Выполнение SHAP анализа...")
        feature_names = preprocessor.selected_features
        shap_analysis = self.calculate_shap_values(model_key, X_test[:100], feature_names)
        # Анализ переобучения
        overfitting_analysis = self.analyze_overfitting(model_key)
        return {
            "model": model,
            "preprocessor": preprocessor,
            "history": self.history[model_key],
            "metrics": metrics,
            "predictions": predictions,
            "shap_analysis": shap_analysis,
            "overfitting_analysis": overfitting_analysis,
            "selected_features": feature_names
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

    def create_forecast(self, crypto_name: str, model_type: str, 
                   features_df: pd.DataFrame, days_ahead: int = 30) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """Создание прогноза с использованием препроцессора"""
        model_key = f"{crypto_name}_{model_type}"
        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не обучена")
        model = self.models[model_key]
        preprocessor = self.scalers[model_key]  # Теперь это полный препроцессор
        # Получаем конфигурацию
        model_config = self.model_configs.get(model_key, {})
        lookback = model_config.get("lookback", 40)
        # Применяем препроцессинг
        X_processed, y = preprocessor.transform(features_df)
        # Добавляем целевую переменную обратно
        data_combined = pd.concat([pd.Series(y, index=X_processed.index, name='Close'), X_processed], axis=1)
        # Последняя последовательность
        last_sequence = data_combined.iloc[-lookback:].values.reshape(1, lookback, -1)
        # Прогноз
        forecast = []
        current_sequence = last_sequence.copy()
        for _ in range(days_ahead):
            pred = model.predict(current_sequence, verbose=0)
            forecast.append(pred[0, 0])
            # Обновление последовательности
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]  # Обновляем предсказанную цену
            current_sequence = np.append(
                current_sequence[0, 1:, :], [new_row], axis=0
            ).reshape(1, lookback, -1)
        # Даты прогноза
        last_date = features_df.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        return forecast_dates, np.array(forecast)        
        
    
    

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

    def analyze_overfitting(self, model_key: str) -> Dict:
        """Анализ переобучения для модели"""
        if model_key not in self.overfitting_history:
            return {"status": "No overfitting data available"}
        history = self.overfitting_history[model_key]
        # Находим эпоху с минимальной val_loss
        best_epoch = min(history, key=lambda x: x["val_loss"])
        # Находим точку начала переобучения
        overfitting_start = None
        for i in range(1, len(history)):
            if history[i]["val_loss"] > history[i - 1]["val_loss"] * 1.02:
                overfitting_start = history[i]["epoch"]
                break
        # Средний коэффициент переобучения
        avg_overfit_ratio = np.mean([h["overfitting_ratio"] for h in history])
        return {
            "best_epoch": best_epoch["epoch"],
            "best_val_loss": best_epoch["val_loss"],
            "overfitting_start_epoch": overfitting_start,
            "average_overfit_ratio": avg_overfit_ratio,
            "final_overfit_ratio": history[-1]["overfitting_ratio"],
            "recommendation": self._get_overfitting_recommendation(avg_overfit_ratio),
        }

    def _get_overfitting_recommendation(self, avg_ratio: float) -> str:
        """Получение рекомендаций по переобучению"""
        if avg_ratio < 1.2:
            return "Модель хорошо генерализуется"
        elif avg_ratio < 1.5:
            return "Умеренное переобучение - рекомендуется увеличить регуляризацию"
        elif avg_ratio < 2.0:
            return "Значительное переобучение - требуется больше dropout или меньше параметров"
        else:
            return "Сильное переобучение - необходимо упростить модель или увеличить данные"
    def post_training_analysis(self, model_key: str) -> Dict:
        """Полный пост-анализ после обучения"""
        if model_key not in self.models:
            raise ValueError(f"Модель {model_key} не найдена")
        analysis = {
            "model_key": model_key,
            "timestamp": pd.Timestamp.now()
        }
        # 1. Метрики производительности
        if model_key in self.metrics:
            analysis["metrics"] = self.metrics[model_key]
        # 2. Анализ переобучения
        if model_key in self.overfitting_history:
            overfit_data = self.analyze_overfitting(model_key)
            analysis["overfitting"] = overfit_data
            # Проверка на переобучение
            if overfit_data["average_overfit_ratio"] > 1.5:
                analysis["warnings"] = analysis.get("warnings", [])
                analysis["warnings"].append("Высокий риск переобучения!")
        # 3. SHAP анализ
        if model_key in self.shap_values:
            shap_data = self.shap_values[model_key]
            top_features = shap_data.get("top_features", [])[:5]
            analysis["top_features"] = top_features
        # 4. Сравнение train vs test
        if model_key in self.predictions:
            pred = self.predictions[model_key]
            train_rmse = np.sqrt(np.mean((pred["y_train"] - pred["train_pred"])**2))
            test_rmse = np.sqrt(np.mean((pred["y_test"] - pred["test_pred"])**2))
            analysis["train_test_gap"] = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "gap_ratio": test_rmse / train_rmse
            }
            if test_rmse > train_rmse * 1.3:
                analysis["warnings"] = analysis.get("warnings", [])
                analysis["warnings"].append("Большой разрыв между train и test!")
        # 5. Рекомендации
        analysis["recommendations"] = self._generate_recommendations(analysis)
        return analysis
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        recommendations = []
        # На основе метрик
        if "metrics" in analysis:
            if analysis["metrics"]["R2"] < 0.5:
                 recommendations.append("Низкий R2 - попробуйте добавить больше данных или изменить архитектуру")
            if analysis["metrics"]["MAPE"] > 10:
                recommendations.append("Высокий MAPE - модель плохо предсказывает масштаб изменений")
        # На основе переобучения
        if "overfitting" in analysis:
            ratio = analysis["overfitting"]["average_overfit_ratio"]
            if ratio > 2.0:
                recommendations.append("Сильное переобучение - увеличьте dropout или упростите модель")
            elif ratio > 1.5:
                recommendations.append("Умеренное переобучение - добавьте регуляризацию")
        # На основе train/test gap
        if "train_test_gap" in analysis:
            if analysis["train_test_gap"]["gap_ratio"] > 1.5:
                recommendations.append("Используйте больше данных для обучения или примените data augmentation")
        if not recommendations:
            recommendations.append("Модель показывает хорошие результаты!")
        return recommendations            
                             
         

