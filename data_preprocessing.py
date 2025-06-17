"""
Модуль для правильной предобработки данных
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Класс для правильной предобработки данных"""

    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None

    def fit_transform(
        self, features_df: pd.DataFrame, target_col: str = "Close"
    ) -> tuple:
        """Полный пайплайн предобработки"""

        # Сохраняем целевую переменную отдельно
        y = features_df[target_col].values
        X = features_df.drop(columns=[target_col])

        logger.info(f"Исходное количество признаков: {X.shape[1]}")

        # 1. ЕДИНАЯ НОРМАЛИЗАЦИЯ
        logger.info("Шаг 1: Нормализация данных...")
        self.scaler = RobustScaler(quantile_range=(5, 95))
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # 2. ЖЕСТКИЙ ОТБОР ПРИЗНАКОВ
        logger.info("Шаг 2: Отбор признаков...")

        # 2.1 Удаление признаков с низкой дисперсией
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X_scaled)
        kept_features = X.columns[variance_selector.get_support()]
        X_scaled = pd.DataFrame(X_variance, columns=kept_features, index=X.index)
        logger.info(f"После VarianceThreshold: {X_scaled.shape[1]} признаков")

        # 2.2 Удаление высококоррелированных признаков
        X_scaled = self._remove_correlated_features(X_scaled, threshold=0.95)
        logger.info(f"После удаления коррелированных: {X_scaled.shape[1]} признаков")

        # 2.3 LassoCV для финального отбора
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)

        # Отбираем признаки с ненулевыми коэффициентами
        selected_mask = np.abs(lasso.coef_) > 0
        self.selected_features = X_scaled.columns[selected_mask].tolist()
        X_final = X_scaled[self.selected_features]

        logger.info(f"Финальное количество признаков: {len(self.selected_features)}")
        logger.info(
            f"Отобранные признаки: {self.selected_features[:10]}..."
        )  # Первые 10

        # Сохраняем информацию для transform
        self.feature_names = self.selected_features
        self.feature_selector = lasso

        # Возвращаем обработанные данные
        return X_final, y

    def transform(self, features_df: pd.DataFrame, target_col: str = "Close") -> tuple:
        """Применение предобработки к новым данным"""
        if self.scaler is None:
            raise ValueError("Сначала выполните fit_transform!")

        y = features_df[target_col].values if target_col in features_df else None
        X = (
            features_df.drop(columns=[target_col])
            if target_col in features_df
            else features_df
        )

        # Применяем нормализацию
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Оставляем только отобранные признаки
        X_final = X_scaled[self.selected_features]

        return X_final, y

    def _remove_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Удаление высококоррелированных признаков"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)

    def get_time_series_splits(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, gap: int = 5
    ) -> list:
        """Правильное разбиение для временных рядов"""
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        splits = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(
                f"Fold {fold+1}: Train size: {len(train_idx)}, Test size: {len(test_idx)}"
            )
            splits.append((train_idx, test_idx))

        return splits
