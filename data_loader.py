"""
Модуль для загрузки и подготовки данных с улучшенными техническими индикаторами
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from config import CRYPTO_SYMBOLS, MACRO_INDICATORS, DIRS
from feature_engineering import AdvancedFeatureEngineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и подготовки данных"""

    def __init__(self):
        self.crypto_data = {}
        self.macro_data = {}
        self.features_data = {}

    def fetch_crypto_data(self, symbols: List[str], period: str = "2y") -> Dict:
        """Загрузка данных криптовалют"""
        logger.info(f"Загрузка данных криптовалют за период: {period}")

        crypto_data = {}
        for name in symbols:
            if name not in CRYPTO_SYMBOLS:
                logger.warning(f"Криптовалюта {name} не найдена в конфигурации")
                continue

            symbol = CRYPTO_SYMBOLS[name]
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if not data.empty:
                    # Очистка данных
                    data = self._clean_crypto_data(data)
                    data = self._filter_data_until_yesterday(data)
                    crypto_data[name] = data
                    logger.info(f"✓ {name}: {len(data)} записей (до вчера)")
                else:
                    logger.warning(f"✗ Нет данных для {name}")

            except Exception as e:
                logger.error(f"Ошибка загрузки {name}: {e}")

        self.crypto_data = crypto_data
        return crypto_data

    def _filter_data_until_yesterday(self, data: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация данных до вчерашнего дня включительно"""
        # Конвертируем индекс в UTC и убираем timezone info для корректного сравнения
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index).tz_localize(None)
        # Получаем вчерашнюю дату без timezone
        yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        yesterday = pd.Timestamp(yesterday.date())

        return data_copy[data_copy.index <= yesterday]

    def _clean_crypto_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных от выбросов и аномалий"""
        # Удаление дней с нулевым объемом
        data = data[data["Volume"] > 0]

        # Заполнение пропущенных значений
        data = data.fillna(method="ffill")

        # Удаление дубликатов
        data = data[~data.index.duplicated(keep="first")]

        return data

    def _clean_macro_data(
        self, data: pd.DataFrame, indicator_name: str
    ) -> pd.DataFrame:
        """Очистка данных макропоказателей с учетом отсутствующего volume"""
        # Проверяем наличие volume
        if "Volume" in data.columns:
            # Если volume есть, но равен 0, используем синтетический volume
            if (data["Volume"] == 0).all():
                # Создаем синтетический volume на основе волатильности цены
                price_changes = data["Close"].pct_change().abs()
                synthetic_volume = price_changes.rolling(window=20).mean() * 1e9
                data["Volume"] = synthetic_volume.fillna(method="bfill")
                logger.info(f"Создан синтетический volume для {indicator_name}")
        else:
            # Если volume отсутствует, добавляем его
            data["Volume"] = 1e6  # Константное значение
            logger.info(f"Добавлен фиктивный volume для {indicator_name}")

        return data

    def fetch_macro_data(self, indicators: List[str], period: str = "2y") -> Dict:
        """Загрузка макропоказателей"""
        logger.info(f"Загрузка макропоказателей за период: {period}")

        macro_data = {}
        for name in indicators:
            if name not in MACRO_INDICATORS:
                logger.warning(f"Макропоказатель {name} не найден в конфигурации")
                continue

            symbol = MACRO_INDICATORS[name]
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if not data.empty:
                    data = self._clean_macro_data(data, name)
                    data = self._filter_data_until_yesterday(data)
                    macro_data[name] = data["Close"]
                    logger.info(f"✓ {name}: {len(data)} записей")
                else:
                    logger.warning(f"✗ Нет данных для {name}")

            except Exception as e:
                logger.error(f"Ошибка загрузки {name}: {e}")

        self.macro_data = macro_data
        return macro_data

    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Расчет улучшенных технических индикаторов"""
        df = pd.DataFrame(index=price_data.index)

        close = price_data["Close"]
        high = price_data["High"]
        low = price_data["Low"]
        volume = price_data["Volume"]

        # Скользящие средние
        df["MA_7"] = close.rolling(window=7).mean()
        df["MA_14"] = close.rolling(window=14).mean()
        df["MA_21"] = close.rolling(window=21).mean()

        # Экспоненциальные скользящие средние
        df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        # RSI
        df["RSI"] = self._calculate_rsi(close, window=14)

        # MACD
        df["MACD"], df["MACD_Signal"] = self._calculate_macd(close)

        # Bollinger Bands
        df["BB_Upper"], df["BB_Lower"], df["BB_Width"] = (
            self._calculate_bollinger_bands(close)
        )

        # Волатильность
        df["Volatility"] = close.rolling(window=20).std()

        # ATR (Average True Range)
        df["ATR"] = self._calculate_atr(high, low, close)

        # Stochastic Oscillator
        df["Stochastic_K"], df["Stochastic_D"] = self._calculate_stochastic(
            high, low, close
        )

        # Объемные индикаторы
        df["Volume_MA"] = volume.rolling(window=20).mean()
        df["Volume_Ratio"] = volume / df["Volume_MA"]

        # Изменения цены
        df["Price_Change"] = close.pct_change()
        df["Price_Change_3d"] = close.pct_change(periods=3)
        df["Price_Change_7d"] = close.pct_change(periods=7)

        # Логарифмические доходности
        df["Log_Return"] = np.log(close / close.shift(1))

        # Ценовые паттерны
        df["High_Low_Ratio"] = high / low
        df["Close_Open_Ratio"] = close / price_data["Open"]

        return df

    def calculate_advanced_technical_indicators(
        self, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Расширенный набор технических индикаторов"""
        df = pd.DataFrame(index=price_data.index)

        close = price_data["Close"]
        high = price_data["High"]
        low = price_data["Low"]
        volume = price_data["Volume"]

        # Скользящие средние
        df["MA_7"] = close.rolling(window=7).mean()
        df["MA_14"] = close.rolling(window=14).mean()
        df["MA_21"] = close.rolling(window=21).mean()

        # Экспоненциальные скользящие средние
        df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

        # RSI
        df["RSI"] = self._calculate_rsi(close, window=14)

        # MACD
        df["MACD"], df["MACD_Signal"] = self._calculate_macd(close)

        # Bollinger Bands
        df["BB_Upper"], df["BB_Lower"], df["BB_Width"] = (
            self._calculate_bollinger_bands(close)
        )

        # Волатильность
        df["Volatility"] = close.rolling(window=20).std()

        # ATR (Average True Range)
        df["ATR"] = self._calculate_atr(high, low, close)

        # Stochastic Oscillator
        df["Stochastic_K"], df["Stochastic_D"] = self._calculate_stochastic(
            high, low, close
        )

        # Объемные индикаторы
        df["Volume_MA"] = volume.rolling(window=20).mean()
        df["Volume_Ratio"] = volume / df["Volume_MA"]

        # Изменения цены
        df["Price_Change"] = close.pct_change()
        df["Price_Change_3d"] = close.pct_change(periods=3)
        df["Price_Change_7d"] = close.pct_change(periods=7)

        # Логарифмические доходности
        df["Log_Return"] = np.log(close / close.shift(1))

        # Ценовые паттерны
        df["High_Low_Ratio"] = high / low
        df["Close_Open_Ratio"] = close / price_data["Open"]

        # 1. Commodity Channel Ind)ex (CCI
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        df["CCI"] = (typical_price - sma_tp) / (0.015 * mad)

        # 2. Williams %R
        highest_high = high.rolling(window=14).max()
        lowest_low = low.rolling(window=14).min()
        df["Williams_R"] = -100 * ((highest_high - close) / (highest_high - lowest_low))

        # 3. Money Flow Index (MFI)
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        positive_flow = pd.Series(0, index=price_data.index)
        negative_flow = pd.Series(0, index=price_data.index)

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        df["MFI"] = mfi

        # 4. Ichimoku Cloud компоненты
        # Tenkan-sen (Conversion Line)
        high_9 = high.rolling(window=9).max()
        low_9 = low.rolling(window=9).min()
        df["Ichimoku_Tenkan"] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line)
        high_26 = high.rolling(window=26).max()
        low_26 = low.rolling(window=26).min()
        df["Ichimoku_Kijun"] = (high_26 + low_26) / 2

        # 5. Parabolic SAR (упрощенная версия)
        df["PSAR"] = self._calculate_psar(high, low, close)

        # 6. Накопительные индикаторы
        df["Cumulative_Volume"] = volume.cumsum()
        df["VWAP"] = (close * volume).cumsum() / volume.cumsum()

        # 7. Индикаторы тренда
        df["ADX"] = self._calculate_adx(high, low, close)

        # 8. Осцилляторы момента
        df["ROC"] = close.pct_change(periods=10) * 100  # Rate of Change
        df["Momentum"] = close - close.shift(10)

        return df

    def _calculate_psar(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        initial_af: float = 0.02,
        max_af: float = 0.2,
    ) -> pd.Series:
        """Расчет Parabolic SAR"""
        psar = close.copy()
        bull = True
        af = initial_af
        ep = high.iloc[0]
        hp = high.iloc[0]
        lp = low.iloc[0]

        for i in range(1, len(close)):
            if bull:
                psar.iloc[i] = psar.iloc[i - 1] + af * (hp - psar.iloc[i - 1])
            else:
                psar.iloc[i] = psar.iloc[i - 1] + af * (lp - psar.iloc[i - 1])

        reverse = False

        if bull:
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                reverse = True
                psar.iloc[i] = hp
                lp = low.iloc[i]
                af = initial_af
        else:
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                reverse = True
                psar.iloc[i] = lp
                hp = high.iloc[i]
                af = initial_af

        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + initial_af, max_af)
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + initial_af, max_af)

        return psar

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Расчет Average Directional Index"""
        # Расчет True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Расчет направленного движения
        up_move = high - high.shift()
        down_move = low.shift() - low
        pos_dm = pd.Series(0, index=high.index)
        neg_dm = pd.Series(0, index=high.index)
        pos_dm[up_move > down_move] = up_move[up_move > down_move]
        neg_dm[down_move > up_move] = down_move[down_move > up_move]

        pos_dm[pos_dm < 0] = 0
        neg_dm[neg_dm < 0] = 0

        # Сглаживание
        atr = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Расчет MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет полос Боллинджера"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        width = upper - lower
        return upper, lower, width

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Расчет Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Расчет Stochastic Oscillator"""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=smooth).mean()

        return k_percent, d_percent

    def prepare_features(
        self, crypto_name: str, include_macro: bool = True
    ) -> pd.DataFrame:
        """Подготовка признаков для модели с правильным масштабированием"""
        if crypto_name not in self.crypto_data:
            raise ValueError(f"Данные для {crypto_name} не загружены")

        # Основные данные
        main_data = self.crypto_data[crypto_name].copy()

        # Создание базового датафрейма
        features_df = pd.DataFrame(index=main_data.index)

        # ВАЖНО: Close должен быть первым столбцом для правильной работы модели
        features_df["Close"] = main_data["Close"]

        # Логарифмическое преобразование для стабилизации дисперсии
        features_df["Log_Close"] = np.log(main_data["Close"])
        features_df["Log_Volume"] = np.log(
            main_data["Volume"] + 1
        )  # +1 для избежания log(0)

        # Основные ценовые признаки
        features_df["High"] = main_data["High"]
        features_df["Low"] = main_data["Low"]
        features_df["Open"] = main_data["Open"]
        features_df["Volume"] = main_data["Volume"]

        # Нормализованные ценовые отношения
        features_df["HL_Ratio"] = (main_data["High"] - main_data["Low"]) / main_data[
            "Close"
        ]
        features_df["OC_Ratio"] = (main_data["Open"] - main_data["Close"]) / main_data[
            "Close"
        ]

        # Технические индикаторы
        tech_indicators = self.calculate_technical_indicators(main_data)

        # Выбираем только самые важные индикаторы
        important_indicators = [
            "MA_7",
            "MA_14",
            "EMA_12",
            "RSI",
            "MACD",
            "BB_Width",
            "ATR",
            "Volume_Ratio",
            "Price_Change",
            "Price_Change_3d",
            "Log_Return",
        ]

        for indicator in important_indicators:
            if indicator in tech_indicators.columns:
                features_df[indicator] = tech_indicators[indicator]
        # Макропоказатели (опционально)
        if include_macro:
            features_df = self._process_macro_indicators(features_df)

        # Временные признаки
        features_df["Day_of_Week"] = features_df.index.dayofweek
        features_df["Day_of_Month"] = features_df.index.day
        features_df["Month"] = features_df.index.month

        # Циклическое кодирование временных признаков
        features_df["Day_Sin"] = np.sin(2 * np.pi * features_df["Day_of_Week"] / 7)
        features_df["Day_Cos"] = np.cos(2 * np.pi * features_df["Day_of_Week"] / 7)
        features_df["Month_Sin"] = np.sin(2 * np.pi * features_df["Month"] / 12)
        features_df["Month_Cos"] = np.cos(2 * np.pi * features_df["Month"] / 12)

        # Удаление временных колонок
        features_df = features_df.drop(["Day_of_Week", "Day_of_Month", "Month"], axis=1)

        # Удаление NaN и inf
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()

        # Проверка корреляции и удаление высококоррелированных признаков
        features_df = self._remove_highly_correlated_features(features_df)

        self.features_data[crypto_name] = features_df

        # Сохранение в CSV
        csv_path = DIRS["datasets"] / f"{crypto_name}_features.csv"
        features_df.to_csv(csv_path)
        logger.info(f"Датасет сохранен: {csv_path}")
        logger.info(f"Форма данных: {features_df.shape}")
        logger.info(f"Признаки: {list(features_df.columns)}")

        return features_df

    def _process_macro_indicators(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Улучшенная обработка макропоказателей"""
        if not self.macro_data:
            return features_df
        for macro_name, macro_series in self.macro_data.items():
            # Ресемплинг на дневные данные
            macro_resampled = macro_series.reindex(features_df.index, method="ffill")
            # Вычисление изменений
            macro_change = macro_resampled.pct_change()
            # Обработка выходных и праздничных дней
            # 1. Находим дни без изменений (выходные/праздники)
            no_change_mask = (macro_change == 0) | macro_change.isna()
            # 2. Для таких дней используем интерполяцию
            if no_change_mask.any():
                # Линейная интерполяция между торговыми днями
                macro_change_clean = macro_change.copy()
                macro_change_clean[no_change_mask] = np.nan
                macro_change_clean = macro_change_clean.interpolate(
                    method="linear", limit_direction="both"
                )
                # Если остались NaN в начале или конце, заполняем средним
                if macro_change_clean.isna().any():
                    mean_change = macro_change_clean.mean()
                    macro_change_clean = macro_change_clean.fillna(
                        mean_change if not np.isnan(mean_change) else 0
                    )
                features_df[f"Macro_{macro_name}_Change"] = macro_change_clean
            else:
                features_df[f"Macro_{macro_name}_Change"] = macro_change
            # Добавляем сглаженную версию для уменьшения шума
            features_df[f"Macro_{macro_name}_Change_MA5"] = (
                features_df[f"Macro_{macro_name}_Change"]
                .rolling(window=5, min_periods=1)
                .mean()
            )
        return features_df

    def _remove_highly_correlated_features(
        self, df: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Удаление высококоррелированных признаков"""
        # Не удаляем Close - это наша целевая переменная
        corr_matrix = df.drop("Close", axis=1).corr().abs()

        # Выбираем верхний треугольник матрицы корреляции
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Находим признаки с корреляцией выше порога
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        if to_drop:
            logger.info(f"Удалены высококоррелированные признаки: {to_drop}")
            df = df.drop(to_drop, axis=1)

        return df

    def create_sequences(
        self, data: np.ndarray, lookback: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для RNN"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback : i])
            y.append(data[i, 0])  # Предсказываем цену закрытия
        return np.array(X), np.array(y)

    def get_data_info(self, crypto_name: str) -> Dict:
        """Получение информации о данных"""
        if crypto_name not in self.features_data:
            return {}

        data = self.features_data[crypto_name]

        info = {
            "shape": data.shape,
            "date_range": (data.index.min(), data.index.max()),
            "missing_values": data.isnull().sum().sum(),
            "features": list(data.columns),
            "price_stats": {
                "min": data["Close"].min(),
                "max": data["Close"].max(),
                "mean": data["Close"].mean(),
                "std": data["Close"].std(),
                "current": data["Close"].iloc[-1],
            },
            "correlations": {
                "top_positive": data.corr()["Close"]
                .sort_values(ascending=False)[1:6]
                .to_dict(),
                "top_negative": data.corr()["Close"].sort_values()[0:5].to_dict(),
            },
        }

        return info
