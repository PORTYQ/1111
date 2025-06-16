# Новый модуль feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.seasonal import seasonal_decompose
import ta  # Technical Analysis library
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineering:
    """Продвинутая инженерия признаков для криптовалют"""

    def __init__(self):
        self.feature_importance = {}
        self.pca_components = None

    def create_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков рыночной микроструктуры"""
        features = pd.DataFrame(index=data.index)

        # 1. Bid-Ask Spread прокси (используя High-Low)
        features["spread_proxy"] = (data["High"] - data["Low"]) / data["Close"]

        # 2. Амихуд иллюидность
        returns = data["Close"].pct_change()
        features["amihud_illiquidity"] = abs(returns) / (data["Volume"] + 1)

        # 3. Kyle's Lambda (ценовое воздействие)
        price_changes = data["Close"].diff()
        signed_volume = data["Volume"] * np.sign(price_changes)
        features["kyle_lambda"] = price_changes / (signed_volume + 1)

        # 4. Realized Volatility (5-минутные интервалы аппроксимация)
        features["realized_volatility"] = returns.rolling(window=20).std() * np.sqrt(
            252
        )

        # 5. Order Flow Imbalance прокси
        features["order_flow_imbalance"] = (data["Close"] - data["Open"]) / (
            data["High"] - data["Low"] + 0.0001
        )

        # 6. VPIN (Volume-synchronized Probability of Informed Trading)
        features["vpin"] = self._calculate_vpin(data)

        return features

    def _calculate_vpin(self, data: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
        """Расчет VPIN метрики"""
        # Упрощенная версия VPIN
        price_changes = data["Close"].pct_change()
        volume = data["Volume"]

        # Классификация объема как buy/sell
        buy_volume = volume.where(price_changes > 0, 0)
        sell_volume = volume.where(price_changes < 0, 0)

        # Расчет дисбаланса в скользящем окне
        buy_sum = buy_volume.rolling(window=bucket_size).sum()
        sell_sum = sell_volume.rolling(window=bucket_size).sum()
        total_sum = buy_sum + sell_sum

        vpin = abs(buy_sum - sell_sum) / (total_sum + 1)

        return vpin

    def create_order_book_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков на основе стакана заявок (приближение)"""
        features = pd.DataFrame(index=data.index)

        # 1. Глубина рынка прокси
        features["market_depth"] = (
            data["Volume"] / data["Volume"].rolling(window=20).mean()
        )

        # 2. Прессинг покупателей/продавцов
        high_close_ratio = (data["High"] - data["Close"]) / (
            data["High"] - data["Low"] + 0.0001
        )
        low_close_ratio = (data["Close"] - data["Low"]) / (
            data["High"] - data["Low"] + 0.0001
        )
        features["buying_pressure"] = low_close_ratio
        features["selling_pressure"] = high_close_ratio

        # 3. Микроструктурный шум
        features["microstructure_noise"] = (
            data["Close"].rolling(window=5).std()
            / data["Close"].rolling(window=20).std()
        )

        return features

    def create_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков на основе настроений рынка"""
        features = pd.DataFrame(index=data.index)

        # 1. Fear & Greed Index компоненты
        # Momentum
        features["momentum_score"] = data["Close"] / data["Close"].shift(10) - 1

        # Volume
        features["volume_score"] = (
            data["Volume"] / data["Volume"].rolling(window=30).mean()
        )

        # Volatility (inverse)
        volatility = data["Close"].pct_change().rolling(window=30).std()
        features["volatility_score"] = 1 / (1 + volatility)

        # Market dominance (прокси через объем)
        features["dominance_score"] = (
            data["Volume"].rolling(window=7).mean()
            / data["Volume"].rolling(window=30).mean()
        )

        # 2. Накопление/Распределение
        mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (
            data["High"] - data["Low"] + 0.0001
        )
        features["accumulation_distribution"] = (mfm * data["Volume"]).cumsum()

        # 3. Smart Money Index прокси
        opening_30min_return = (data["Close"] - data["Open"]) / data["Open"]
        features["smart_money_index"] = opening_30min_return.rolling(window=20).mean()

        return features

    def create_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание циклических признаков"""
        features = pd.DataFrame(index=data.index)

        # 1. Фурье-преобразование для выявления циклов
        close_fft = np.fft.fft(data["Close"].values)
        n = len(close_fft)
        freq = np.fft.fftfreq(n)

        # Выбираем топ-5 частот
        power = np.abs(close_fft)
        top_freq_idx = np.argsort(power)[-6:-1]  # Исключаем DC компонент

        for i, idx in enumerate(top_freq_idx):
            period = 1 / abs(freq[idx]) if freq[idx] != 0 else n
            features[f"cycle_{i}_sin"] = np.sin(2 * np.pi * np.arange(n) / period)
            features[f"cycle_{i}_cos"] = np.cos(2 * np.pi * np.arange(n) / period)

        # 2. Сезонная декомпозиция (если достаточно данных)
        if len(data) > 365:
            try:
                decomposition = seasonal_decompose(
                    data["Close"], model="multiplicative", period=365
                )
                features["seasonal"] = decomposition.seasonal
                features["trend"] = decomposition.trend
            except:
                logger.warning("Не удалось выполнить сезонную декомпозицию")

        return features

    def create_whale_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Признаки для обнаружения активности крупных игроков"""
        features = pd.DataFrame(index=data.index)

        # 1. Аномальный объем
        volume_zscore = (
            data["Volume"] - data["Volume"].rolling(window=30).mean()
        ) / data["Volume"].rolling(window=30).std()
        features["whale_volume"] = volume_zscore > 2

        # 2. Большие ценовые движения с низким объемом (манипуляция)
        price_change = data["Close"].pct_change().abs()
        features["price_manipulation"] = (
            price_change > price_change.quantile(0.95)
        ) & (volume_zscore < 0)

        # 3. Накопление китов (постепенный рост объема)
        features["whale_accumulation"] = (
            data["Volume"].rolling(window=10).mean()
            / data["Volume"].rolling(window=50).mean()
        )

        # 4. Следы stop-loss hunting
        high_low_range = (data["High"] - data["Low"]) / data["Close"]
        features["stop_hunting"] = (high_low_range > high_low_range.quantile(0.95)) & (
            data["Close"].pct_change().abs() < 0.01
        )

        return features

    def create_network_features(
        self, data: pd.DataFrame, other_cryptos: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Создание сетевых признаков (корреляции с другими криптовалютами)"""
        features = pd.DataFrame(index=data.index)

        close_returns = data["Close"].pct_change()

        for crypto_name, crypto_data in other_cryptos.items():
            if len(crypto_data) == len(data):
                other_returns = crypto_data["Close"].pct_change()

                # Скользящая корреляция
                features[f"corr_{crypto_name}_30d"] = close_returns.rolling(
                    window=30
                ).corr(other_returns)

                # Beta относительно другой криптовалюты
                covariance = close_returns.rolling(window=30).cov(other_returns)
                variance = other_returns.rolling(window=30).var()
                features[f"beta_{crypto_name}"] = covariance / (variance + 0.0001)

        return features

    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Определение рыночных режимов"""
        features = pd.DataFrame(index=data.index)

        returns = data["Close"].pct_change()

        # 1. Режимы волатильности (низкая/средняя/высокая)
        volatility = returns.rolling(window=20).std()
        vol_percentiles = volatility.rolling(window=252).rank(pct=True)
        features["low_vol_regime"] = vol_percentiles < 0.33
        features["medium_vol_regime"] = (vol_percentiles >= 0.33) & (
            vol_percentiles < 0.67
        )
        features["high_vol_regime"] = vol_percentiles >= 0.67

        # 2. Трендовые режимы
        sma_50 = data["Close"].rolling(window=50).mean()
        sma_200 = data["Close"].rolling(window=200).mean()
        features["uptrend"] = (data["Close"] > sma_50) & (sma_50 > sma_200)
        features["downtrend"] = (data["Close"] < sma_50) & (sma_50 < sma_200)
        features["sideways"] = ~(features["uptrend"] | features["downtrend"])

        # 3. Режимы ликвидности
        volume_ma = data["Volume"].rolling(window=20).mean()
        volume_percentile = data["Volume"].rolling(window=252).rank(pct=True)
        features["high_liquidity"] = volume_percentile > 0.7
        features["low_liquidity"] = volume_percentile < 0.3

        return features

    def apply_feature_selection(
        self, features_df: pd.DataFrame, target: pd.Series, n_features: int = 50
    ) -> pd.DataFrame:
        """Отбор наиболее важных признаков"""
        # Удаляем NaN
        valid_idx = ~(features_df.isnull().any(axis=1) | target.isnull())
        features_clean = features_df[valid_idx]
        target_clean = target[valid_idx]

        # SelectKBest
        selector = SelectKBest(
            score_func=f_regression, k=min(n_features, features_clean.shape[1])
        )
        selected_features = selector.fit_transform(features_clean, target_clean)

        # Получаем имена выбранных признаков
        selected_indices = selector.get_support(indices=True)
        selected_columns = features_df.columns[selected_indices]

        # Сохраняем важность признаков
        self.feature_importance = dict(zip(features_df.columns, selector.scores_))

        logger.info(
            f"Выбрано {len(selected_columns)} признаков из {features_df.shape[1]}"
        )

        return features_df[selected_columns]

    def apply_pca_transformation(
        self, features_df: pd.DataFrame, n_components: float = 0.95
    ) -> pd.DataFrame:
        """Применение PCA для снижения размерности"""
        # Стандартизация
        features_scaled = (features_df - features_df.mean()) / features_df.std()
        features_scaled = features_scaled.fillna(0)

        # PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features_scaled)

        # Создание DataFrame с PCA компонентами
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f"pca_{i}" for i in range(pca_features.shape[1])],
            index=features_df.index,
        )

        self.pca_components = pca

        logger.info(
            f"PCA: {pca_features.shape[1]} компонент объясняют "
            f"{pca.explained_variance_ratio_.sum():.2%} дисперсии"
        )

        return pca_df

    def create_interaction_features(
        self, features_df: pd.DataFrame, max_interactions: int = 10
    ) -> pd.DataFrame:
        """Создание признаков взаимодействия"""
        interaction_features = pd.DataFrame(index=features_df.index)

        # Выбираем топ признаки для взаимодействия
        if hasattr(self, "feature_importance") and self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:max_interactions]
            top_feature_names = [f[0] for f in top_features]
        else:
            # Если важность не рассчитана, берем первые признаки
            top_feature_names = features_df.columns[:max_interactions]

        # Создаем взаимодействия
        for i, feat1 in enumerate(top_feature_names):
            for feat2 in top_feature_names[i + 1 :]:
                if feat1 in features_df.columns and feat2 in features_df.columns:
                    # Мультипликативное взаимодействие
                    interaction_features[f"{feat1}_x_{feat2}"] = (
                        features_df[feat1] * features_df[feat2]
                    )

                    # Соотношение
                    interaction_features[f"{feat1}_div_{feat2}"] = features_df[
                        feat1
                    ] / (features_df[feat2] + 0.0001)

        return interaction_features

    def create_lag_features(
        self, features_df: pd.DataFrame, lags: List[int] = [1, 3, 7, 14, 30]
    ) -> pd.DataFrame:
        """Создание лаговых признаков"""
        lag_features = pd.DataFrame(index=features_df.index)

        # Выбираем ключевые признаки для лагов
        key_features = (
            ["Close", "Volume", "RSI", "MACD"]
            if "RSI" in features_df.columns
            else ["Close", "Volume"]
        )

        for feature in key_features:
            if feature in features_df.columns:
                for lag in lags:
                    # Лаг
                    lag_features[f"{feature}_lag_{lag}"] = features_df[feature].shift(
                        lag
                    )

                    # Изменение относительно лага
                    lag_features[f"{feature}_change_{lag}"] = (
                        features_df[feature] / features_df[feature].shift(lag) - 1
                    )

                    # Скользящее среднее за период
                    lag_features[f"{feature}_ma_{lag}"] = (
                        features_df[feature].rolling(window=lag).mean()
                    )

        return lag_features

    def create_all_features(
        self, data: pd.DataFrame, other_cryptos: Dict[str, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Создание всех продвинутых признаков"""
        all_features = []

        # 1. Микроструктурные признаки
        logger.info("Создание микроструктурных признаков...")
        all_features.append(self.create_market_microstructure_features(data))

        # 2. Признаки стакана заявок
        logger.info("Создание признаков стакана заявок...")
        all_features.append(self.create_order_book_features(data))

        # 3. Признаки настроений
        logger.info("Создание признаков настроений...")
        all_features.append(self.create_sentiment_features(data))

        # 4. Циклические признаки
        logger.info("Создание циклических признаков...")
        all_features.append(self.create_cyclical_features(data))

        # 5. Признаки китов
        logger.info("Создание признаков китов...")
        all_features.append(self.create_whale_detection_features(data))

        # 6. Сетевые признаки (если есть данные других криптовалют)
        if other_cryptos:
            logger.info("Создание сетевых признаков...")
            all_features.append(self.create_network_features(data, other_cryptos))

        # 7. Режимные признаки
        logger.info("Создание режимных признаков...")
        all_features.append(self.create_regime_features(data))

        # Объединение всех признаков
        combined_features = pd.concat(all_features, axis=1)

        # Добавление лаговых признаков
        logger.info("Создание лаговых признаков...")
        lag_features = self.create_lag_features(combined_features)
        combined_features = pd.concat([combined_features, lag_features], axis=1)

        # Очистка от NaN и inf
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
        combined_features = combined_features.fillna(method="ffill").fillna(0)

        logger.info(f"Создано {combined_features.shape[1]} признаков")

        return combined_features
