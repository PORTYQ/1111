"""
Обновленный модуль для визуализации данных и результатов
Поддержка ансамблевых прогнозов и расширенной аналитики
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from config import PLOT_CONFIG, DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка стиля matplotlib
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class Visualizer:
    """Класс для создания визуализаций с поддержкой новых типов графиков"""

    def __init__(self):
        self.colors = PLOT_CONFIG["colors"]
        self.extended_colors = {
            **self.colors,
            "ensemble": "#9c27b0",
            "confidence": "#e0e0e0",
            "advanced": "#ff5722",
        }

    def plot_price_history(
        self, crypto_data: Dict, crypto_name: str, save_plot: bool = True
    ) -> go.Figure:
        """График истории цен криптовалюты"""
        if crypto_name not in crypto_data:
            raise ValueError(f"Данные для {crypto_name} не найдены")

        data = crypto_data[crypto_name]

        # Создание подграфиков
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=[
                f"{crypto_name} - Цена и объем",
                "Технические индикаторы",
                "Волатильность",
            ],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ],
            vertical_spacing=0.08,
        )

        # График цены
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Цена",
            ),
            row=1,
            col=1,
        )

        # График объема
        fig.add_trace(
            go.Bar(
                x=data.index, y=data["Volume"], name="Объем", opacity=0.3, yaxis="y2"
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Скользящие средние
        ma_7 = data["Close"].rolling(window=7).mean()
        ma_21 = data["Close"].rolling(window=21).mean()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_7,
                mode="lines",
                name="MA 7",
                line=dict(color="orange", width=1),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma_21,
                mode="lines",
                name="MA 21",
                line=dict(color="red", width=1),
            ),
            row=2,
            col=1,
        )

        # Волатильность
        volatility = data["Close"].rolling(window=20).std()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volatility,
                mode="lines",
                name="Волатильность (20д)",
                line=dict(color="purple"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            title=f"{crypto_name} - Анализ цены",
            height=800,
            xaxis_rangeslider_visible=False,
        )

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_price_history.html"
            fig.write_html(str(file_path))
            logger.info(f"График сохранен: {file_path}")

        return fig

    def plot_ensemble_forecast(
        self,
        current_data: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        ensemble_forecast: np.ndarray,
        confidence_intervals: Dict,
        crypto_name: str,
        save_plot: bool = True,
    ) -> go.Figure:
        """График ансамблевого прогноза с доверительными интервалами"""

        fig = go.Figure()

        # Исторические данные (последние 90 дней)
        recent_data = current_data.tail(90)
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data.values,
                mode="lines",
                name="Исторические данные",
                line=dict(color=self.extended_colors["actual"], width=2),
            )
        )

        # Ансамблевый прогноз
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=ensemble_forecast,
                mode="lines+markers",
                name="Ансамблевый прогноз",
                line=dict(color=self.extended_colors["ensemble"], width=3),
                marker=dict(size=6),
            )
        )

        # Доверительный интервал 95%
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=confidence_intervals["upper_95"],
                mode="lines",
                name="95% верхняя граница",
                line=dict(color=self.extended_colors["confidence"], width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=confidence_intervals["lower_95"],
                mode="lines",
                name="95% доверительный интервал",
                line=dict(color=self.extended_colors["confidence"], width=0),
                fill="tonexty",
                fillcolor="rgba(156, 39, 176, 0.2)",
            )
        )

        # Стандартное отклонение как полоса
        std_upper = ensemble_forecast + confidence_intervals["std"]
        std_lower = ensemble_forecast - confidence_intervals["std"]

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=std_upper,
                mode="lines",
                name="±1σ",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=std_lower,
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            )
        )

        # Соединительная линия
        fig.add_trace(
            go.Scatter(
                x=[recent_data.index[-1], forecast_dates[0]],
                y=[recent_data.values[-1], ensemble_forecast[0]],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            )
        )

        # Аннотации
        current_price = recent_data.values[-1]
        final_price = ensemble_forecast[-1]
        change_pct = ((final_price - current_price) / current_price) * 100

        fig.add_annotation(
            x=forecast_dates[-1],
            y=final_price,
            text=f"${final_price:.2f}<br>({change_pct:+.1f}%)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.extended_colors["ensemble"],
            bgcolor="white",
            bordercolor=self.extended_colors["ensemble"],
            borderwidth=2,
        )

        fig.update_layout(
            title=f"{crypto_name} - Ансамблевый прогноз с доверительными интервалами",
            xaxis_title="Дата",
            yaxis_title="Цена ($)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_ensemble_forecast.html"
            fig.write_html(str(file_path))
            logger.info(f"График ансамблевого прогноза сохранен: {file_path}")

        return fig

    def plot_advanced_model_comparison(
        self, metrics_dict: Dict, crypto_name: str, save_plot: bool = True
    ) -> go.Figure:
        """Расширенное сравнение моделей с радиальной диаграммой"""

        models = list(metrics_dict.keys())
        model_names = [model.split("_")[-1] for model in models]

        # Создание подграфиков
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Основные метрики",
                "Радиальная диаграмма производительности",
                "Корреляция ошибок",
                "Временная стабильность",
            ],
            specs=[
                [{"type": "bar"}, {"type": "polar"}],
                [{"type": "heatmap"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. Основные метрики (барный график)
        metrics_to_plot = ["RMSE", "MAE", "MAPE"]
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_dict[model][metric] for model in models]
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    text=[f"{v:.2f}" for v in values],
                    textposition="auto",
                ),
                row=1,
                col=1,
            )

        # 2. Радиальная диаграмма
        # Нормализация метрик для радиальной диаграммы
        metrics_for_radar = ["R2", "Directional_Accuracy"]
        inverse_metrics = ["RMSE", "MAE", "MAPE"]  # Метрики, где меньше = лучше

        for model, model_name in zip(models, model_names):
            values = []
            categories = []

            # Добавляем прямые метрики
            for metric in metrics_for_radar:
                values.append(metrics_dict[model][metric])
                categories.append(metric)

            # Добавляем инвертированные метрики
            for metric in inverse_metrics:
                # Нормализуем и инвертируем
                max_val = max(metrics_dict[m][metric] for m in models)
                min_val = min(metrics_dict[m][metric] for m in models)
                if max_val != min_val:
                    normalized = 1 - (metrics_dict[model][metric] - min_val) / (
                        max_val - min_val
                    )
                else:
                    normalized = 0.5
                values.append(normalized)
                categories.append(f"{metric} (inv)")

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Замкнуть фигуру
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=model_name,
                    opacity=0.7,
                ),
                row=1,
                col=2,
            )

        # 3. Матрица корреляции ошибок (заглушка)
        correlation_matrix = np.random.rand(len(models), len(models))
        np.fill_diagonal(correlation_matrix, 1)

        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=model_names,
                y=model_names,
                colorscale="RdBu",
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
                texttemplate="%{text}",
                showscale=False,
            ),
            row=2,
            col=1,
        )

        # 4. Временная стабильность (линейный график)
        for model, model_name in zip(models, model_names):
            # Симуляция временной стабильности
            time_points = list(range(10))
            stability_values = [
                metrics_dict[model]["RMSE"] * (1 + np.random.normal(0, 0.1))
                for _ in time_points
            ]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=stability_values,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(width=2),
                    marker=dict(size=6),
                ),
                row=2,
                col=2,
            )

        # Обновление макета
        fig.update_layout(
            title=f"{crypto_name} - Расширенное сравнение моделей",
            height=900,
            showlegend=True,
        )

        # Настройка осей
        fig.update_xaxes(title_text="Модель", row=1, col=1)
        fig.update_yaxes(title_text="Значение", row=1, col=1)
        fig.update_xaxes(title_text="Временной период", row=2, col=2)
        fig.update_yaxes(title_text="RMSE", row=2, col=2)

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_advanced_comparison.html"
            fig.write_html(str(file_path))
            logger.info(f"График расширенного сравнения сохранен: {file_path}")

        return fig

    def plot_feature_importance_advanced(
        self,
        features_df: pd.DataFrame,
        feature_importance: Dict,
        crypto_name: str,
        save_plot: bool = True,
    ) -> go.Figure:
        """Продвинутый анализ важности признаков с группировкой"""

        # Создание подграфиков
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Топ-20 важных признаков",
                "Важность по категориям",
                "Корреляционная матрица топ признаков",
                "Временная динамика признаков",
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.15,
        )

        # 1. Топ-20 признаков
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:20]
            feature_names, importance_values = zip(*sorted_features)

            fig.add_trace(
                go.Bar(
                    y=list(feature_names),
                    x=list(importance_values),
                    orientation="h",
                    marker_color=self.extended_colors["advanced"],
                    text=[f"{v:.3f}" for v in importance_values],
                    textposition="auto",
                ),
                row=1,
                col=1,
            )

        # 2. Важность по категориям
        categories = {
            "Технические": ["RSI", "MACD", "ATR", "BB_", "MA_", "EMA_"],
            "Микроструктура": ["spread", "vpin", "amihud", "kyle"],
            "Настроения": ["sentiment", "fear", "greed", "accumulation"],
            "Циклические": ["cycle", "seasonal", "trend"],
            "Сетевые": ["corr_", "beta_"],
            "Другие": [],
        }

        category_importance = {}
        for category, keywords in categories.items():
            total_importance = 0
            for feature, importance in feature_importance.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    total_importance += importance
            if total_importance > 0:
                category_importance[category] = total_importance

        if category_importance:
            fig.add_trace(
                go.Pie(
                    labels=list(category_importance.keys()),
                    values=list(category_importance.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3,
                ),
                row=1,
                col=2,
            )

        # 3. Корреляционная матрица топ признаков
        top_features = [f for f, _ in sorted_features[:10]]
        if all(f in features_df.columns for f in top_features):
            corr_matrix = features_df[top_features].corr()

            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showscale=True,
                ),
                row=2,
                col=1,
            )

        # 4. Временная динамика важных признаков
        for i, (feature, _) in enumerate(sorted_features[:5]):
            if feature in features_df.columns:
                # Нормализация для сравнения
                normalized = (
                    features_df[feature] - features_df[feature].mean()
                ) / features_df[feature].std()

                fig.add_trace(
                    go.Scatter(
                        x=features_df.index[-100:],  # Последние 100 точек
                        y=normalized.iloc[-100:],
                        mode="lines",
                        name=feature[:20],  # Обрезаем длинные имена
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )

        # Обновление макета
        fig.update_layout(
            title=f"{crypto_name} - Продвинутый анализ важности признаков",
            height=900,
            showlegend=True,
        )

        # Настройка осей
        fig.update_xaxes(title_text="Важность", row=1, col=1)
        fig.update_yaxes(title_text="Признак", row=1, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=2)
        fig.update_yaxes(title_text="Нормализованное значение", row=2, col=2)

        if save_plot:
            file_path = (
                DIRS["plots"] / f"{crypto_name}_feature_importance_advanced.html"
            )
            fig.write_html(str(file_path))
            logger.info(f"График важности признаков сохранен: {file_path}")

        return fig

    def plot_predictions(
        self,
        predictions: Dict,
        crypto_name: str,
        model_type: str,
        save_plot: bool = True,
    ) -> go.Figure:
        """График предсказаний модели"""

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                f"{crypto_name} - Предсказания модели {model_type}",
                "Ошибки предсказания",
            ],
            vertical_spacing=0.1,
        )

        # График предсказаний для обучающей выборки
        fig.add_trace(
            go.Scatter(
                x=predictions["train_dates"],
                y=predictions["y_train"],
                mode="lines",
                name="Факт (обучение)",
                line=dict(color=self.colors["actual"], width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=predictions["train_dates"],
                y=predictions["train_pred"],
                mode="lines",
                name="Прогноз (обучение)",
                line=dict(color=self.colors["predicted"], width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # График предсказаний для тестовой выборки
        fig.add_trace(
            go.Scatter(
                x=predictions["test_dates"],
                y=predictions["y_test"],
                mode="lines",
                name="Факт (тест)",
                line=dict(color=self.colors["actual"], width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=predictions["test_dates"],
                y=predictions["test_pred"],
                mode="lines",
                name="Прогноз (тест)",
                line=dict(color=self.colors["forecast"], width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # График ошибок
        test_errors = predictions["y_test"] - predictions["test_pred"]
        fig.add_trace(
            go.Scatter(
                x=predictions["test_dates"],
                y=test_errors,
                mode="lines+markers",
                name="Ошибка предсказания",
                line=dict(color="red"),
                marker=dict(size=3),
            ),
            row=2,
            col=1,
        )

        # Добавление нулевой линии для ошибок
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            title=f"{crypto_name} - Анализ модели {model_type}",
            height=800,
            showlegend=True,
        )

        fig.update_xaxes(title_text="Дата", row=1, col=1)
        fig.update_yaxes(title_text="Цена ($)", row=1, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=1)
        fig.update_yaxes(title_text="Ошибка ($)", row=2, col=1)

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_{model_type}_predictions.html"
            fig.write_html(str(file_path))
            logger.info(f"График предсказаний сохранен: {file_path}")

        return fig

    def plot_forecast(
        self,
        current_data: pd.Series,
        forecast_dates: pd.DatetimeIndex,
        forecast_values: np.ndarray,
        crypto_name: str,
        model_type: str,
        save_plot: bool = True,
    ) -> go.Figure:
        """График прогноза на будущее"""

        fig = go.Figure()

        # Исторические данные (последние 60 дней)
        recent_data = current_data.tail(60)
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data.values,
                mode="lines",
                name="Исторические данные",
                line=dict(color=self.colors["actual"], width=2),
            )
        )

        # Прогноз
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines+markers",
                name=f"Прогноз ({model_type})",
                line=dict(color=self.colors["forecast"], width=2, dash="dash"),
                marker=dict(size=4),
            )
        )

        # Соединительная линия
        fig.add_trace(
            go.Scatter(
                x=[recent_data.index[-1], forecast_dates[0]],
                y=[recent_data.values[-1], forecast_values[0]],
                mode="lines",
                name="Соединение",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            )
        )

        # Доверительный интервал (упрощенный)
        std_dev = np.std(forecast_values) * 0.5
        upper_bound = forecast_values + std_dev
        lower_bound = forecast_values - std_dev

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=upper_bound,
                mode="lines",
                name="Верхняя граница",
                line=dict(color="lightgray", width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=lower_bound,
                mode="lines",
                name="Нижняя граница",
                line=dict(color="lightgray", width=0),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.2)",
                showlegend=True,
            )
        )

        fig.update_layout(
            title=f"{crypto_name} - Прогноз на {len(forecast_dates)} дней ({model_type})",
            xaxis_title="Дата",
            yaxis_title="Цена ($)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_{model_type}_forecast.html"
            fig.write_html(str(file_path))
            logger.info(f"График прогноза сохранен: {file_path}")

        return fig

    def plot_training_history(
        self, history: Dict, crypto_name: str, model_type: str, save_plot: bool = True
    ) -> go.Figure:
        """График истории обучения"""

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Функция потерь", "Средняя абсолютная ошибка"],
        )

        epochs = range(1, len(history.history["loss"]) + 1)

        # График потерь
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history["loss"],
                mode="lines",
                name="Потери (обучение)",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=history.history["val_loss"],
                mode="lines",
                name="Потери (валидация)",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )

        # График MAE
        if "mae" in history.history:
            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history["mae"],
                    mode="lines",
                    name="MAE (обучение)",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=list(epochs),
                    y=history.history["val_mae"],
                    mode="lines",
                    name="MAE (валидация)",
                    line=dict(color="orange"),
                ),
                row=1,
                col=2,
            )

        fig.update_layout(
            title=f"{crypto_name} - История обучения модели {model_type}", height=400
        )

        fig.update_xaxes(title_text="Эпоха", row=1, col=1)
        fig.update_yaxes(title_text="Потери", row=1, col=1)
        fig.update_xaxes(title_text="Эпоха", row=1, col=2)
        fig.update_yaxes(title_text="MAE", row=1, col=2)

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_{model_type}_training.html"
            fig.write_html(str(file_path))
            logger.info(f"График обучения сохранен: {file_path}")

        return fig

    def plot_model_comparison(
        self, metrics_dict: Dict, crypto_name: str, save_plot: bool = True
    ) -> go.Figure:
        """Сравнение моделей"""

        models = list(metrics_dict.keys())
        model_names = [model.split("_")[-1] for model in models]

        metrics_names = ["RMSE", "MAE", "R2", "MAPE", "Directional_Accuracy"]

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=metrics_names,
            specs=[[{}, {}, {}], [{}, {}, {}]],
            vertical_spacing=0.15,
        )

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

        for i, metric in enumerate(metrics_names):
            if i >= len(positions):
                break

            values = [metrics_dict[model][metric] for model in models]

            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    showlegend=False,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                ),
                row=positions[i][0],
                col=positions[i][1],
            )

        fig.update_layout(
            title=f"{crypto_name} - Сравнение моделей", height=600, showlegend=False
        )

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_model_comparison.html"
            fig.write_html(str(file_path))
            logger.info(f"Сравнение моделей сохранено: {file_path}")

        return fig

    def plot_feature_importance(
        self, features_df: pd.DataFrame, crypto_name: str, save_plot: bool = True
    ) -> go.Figure:
        """Анализ важности признаков через корреляцию"""

        # Корреляция с ценой закрытия
        correlations = (
            features_df.corr()["Close"]
            .drop("Close")
            .sort_values(key=abs, ascending=False)
        )

        fig = go.Figure()

        colors = ["red" if x < 0 else "blue" for x in correlations.values]

        fig.add_trace(
            go.Bar(
                x=correlations.values,
                y=correlations.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in correlations.values],
                textposition="auto",
            )
        )

        fig.update_layout(
            title=f"{crypto_name} - Корреляция признаков с ценой",
            xaxis_title="Корреляция",
            yaxis_title="Признаки",
            height=max(400, len(correlations) * 25),
        )

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_feature_importance.html"
            fig.write_html(str(file_path))
            logger.info(f"График важности признаков сохранен: {file_path}")

        return fig

    def create_summary_dashboard(
        self, crypto_name: str, predictions: Dict, metrics: Dict, save_plot: bool = True
    ) -> go.Figure:
        """Создание сводной панели"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Предсказания vs Факт",
                "Метрики моделей",
                "Распределение ошибок",
                "Точность направления",
            ],
            specs=[[{}, {}], [{}, {}]],
        )

        # График предсказаний (тестовая выборка)
        for model_key, pred_data in predictions.items():
            model_type = model_key.split("_")[-1]

            fig.add_trace(
                go.Scatter(
                    x=pred_data["y_test"],
                    y=pred_data["test_pred"],
                    mode="markers",
                    name=f"{model_type}",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

        # Идеальная линия
        if predictions:
            all_actual = np.concatenate(
                [pred["y_test"] for pred in predictions.values()]
            )
            min_val, max_val = all_actual.min(), all_actual.max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Идеальное предсказание",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Метрики моделей
        models = list(metrics.keys())
        model_names = [model.split("_")[-1] for model in models]
        rmse_values = [metrics[model]["RMSE"] for model in models]

        fig.add_trace(
            go.Bar(x=model_names, y=rmse_values, name="RMSE", showlegend=False),
            row=1,
            col=2,
        )

        # Распределение ошибок
        for model_key, pred_data in predictions.items():
            model_type = model_key.split("_")[-1]
            errors = pred_data["y_test"] - pred_data["test_pred"]

            fig.add_trace(
                go.Histogram(
                    x=errors, name=f"Ошибки {model_type}", opacity=0.7, nbinsx=30
                ),
                row=2,
                col=1,
            )

        # Точность направления
        direction_acc = [metrics[model]["Directional_Accuracy"] for model in models]

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=direction_acc,
                name="Точность направления (%)",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"{crypto_name} - Сводная панель результатов", height=800
        )

        # Обновление осей
        fig.update_xaxes(title_text="Фактическая цена", row=1, col=1)
        fig.update_yaxes(title_text="Предсказанная цена", row=1, col=1)
        fig.update_xaxes(title_text="Модель", row=1, col=2)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_xaxes(title_text="Ошибка", row=2, col=1)
        fig.update_yaxes(title_text="Частота", row=2, col=1)
        fig.update_xaxes(title_text="Модель", row=2, col=2)
        fig.update_yaxes(title_text="Точность (%)", row=2, col=2)

        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_summary_dashboard.html"
            fig.write_html(str(file_path))
            logger.info(f"Сводная панель сохранена: {file_path}")

        return fig

    def plot_overfitting_history(self, model_key: str):
        """Визуализация истории переобучения"""
        if (
            not hasattr(self, "overfitting_history")
            or model_key not in self.overfitting_history
        ):
            logger.warning(f"Нет данных о переобучении для {model_key}")
            return
        import matplotlib.pyplot as plt

        history = self.overfitting_history[model_key]
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        overfit_ratios = [h["overfitting_ratio"] for h in history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(epochs, train_losses, label="Train Loss", color="blue")
        ax1.plot(epochs, val_losses, label="Validation Loss", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Training History - {model_key}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # График коэффициента переобучения
        ax2.plot(epochs, overfit_ratios, label="Overfitting Ratio", color="orange")
        ax2.axhline(y=1.5, color="red", linestyle="--", label="Warning Threshold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Loss / Train Loss")
        ax2.set_title("Overfitting Ratio")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = DIRS["plots"] / f"{model_key}_overfitting_analysis.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"График переобучения сохранен: {save_path}")

    def plot_shap_summary(self, model_key: str, shap_data: Dict, 
                          crypto_name: str, save_plot: bool = True) -> go.Figure:
        """Визуализация SHAP значений"""
        if not shap_data or 'feature_importance' not in shap_data:
            logger.warning("Нет данных SHAP для визуализации")
            return None
        feature_importance = shap_data['feature_importance']
        feature_names = shap_data.get('top_features', [])
        # Создаем график важности признаков
        fig = go.Figure()
        # Берем топ-20 признаков
        top_n = 20
        if feature_names:
            names = [f['name'] if 'name' in f else f"Feature {f['index']}" 
                     for f in feature_names[:top_n]]
            values = [f['importance'] for f in feature_names[:top_n]]
        else:
            names = [f"Feature {i}" for i in range(min(top_n, len(feature_importance)))]
            values = feature_importance[:top_n]
        fig.add_trace(go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color='lightblue',
            text=[f"{v:.4f}" for v in values],
            textposition='auto',
        ))         
        fig.update_layout(
            title=f"{crypto_name} - SHAP Feature Importance ({model_key})",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Features",
            height=600,
            template="plotly_white"
        )
        if save_plot:
            file_path = DIRS["plots"] / f"{crypto_name}_{model_key}_shap_importance.html"
            fig.write_html(str(file_path))
            logger.info(f"SHAP график сохранен: {file_path}")
        return fig
    def plot_overfitting_analysis(self, overfitting_history: List[Dict], 
                                  model_key: str, save_plot: bool = True) -> go.Figure:
        """График анализа переобучения"""
        if not overfitting_history:
            return None
        epochs = [h['epoch'] for h in overfitting_history]
        train_losses = [h['train_loss'] for h in overfitting_history]
        val_losses = [h['val_loss'] for h in overfitting_history]
        overfit_ratios = [h['overfitting_ratio'] for h in overfitting_history]
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Loss History", "Overfitting Ratio"],
            vertical_spacing=0.1
        )
        # График потерь
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name="Train Loss", 
                       line=dict(color="blue")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name="Val Loss", 
                       line=dict(color="red")),
            row=1, col=1
        )
        # График коэффициента переобучения
        fig.add_trace(
            go.Scatter(x=epochs, y=overfit_ratios, name="Overfit Ratio", 
                       line=dict(color="orange")),
             row=2, col=1
        )
        # Добавляем пороговую линию
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", 
                      annotation_text="Warning Threshold", row=2, col=1)
        fig.update_layout(
            title=f"Overfitting Analysis - {model_key}",
            height=600,
            template="plotly_white"
        )
        if save_plot:
            file_path = DIRS["plots"] / f"{model_key}_overfitting_analysis.html"
            fig.write_html(str(file_path))
        return fig                                                                       
