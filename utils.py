"""
Утилиты и вспомогательные функции
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from config import DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Класс для валидации данных"""

    @staticmethod
    def validate_crypto_data(data: pd.DataFrame, crypto_name: str) -> bool:
        """Валидация данных криптовалюты"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Проверка наличия колонок
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Отсутствуют колонки для {crypto_name}: {missing_columns}")
            return False

        # Проверка на пустые данные
        if data.empty:
            logger.error(f"Пустые данные для {crypto_name}")
            return False

        # Проверка на отрицательные цены
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.warning(
                    f"Найдены отрицательные или нулевые цены в {col} для {crypto_name}"
                )

        # Проверка логики High >= Low
        if (data["High"] < data["Low"]).any():
            logger.error(f"Нарушена логика High >= Low для {crypto_name}")
            return False

        # Проверка на выбросы (цены, отличающиеся более чем в 10 раз)
        price_ratios = data["Close"].pct_change().abs()
        outliers = price_ratios > 5.0  # 500% изменение за день
        if outliers.any():
            logger.warning(
                f"Найдены потенциальные выбросы в данных {crypto_name}: "
                f"{outliers.sum()} точек"
            )

        return True

    @staticmethod
    def validate_features(features_df: pd.DataFrame) -> bool:
        """Валидация признаков для модели"""
        # Проверка на NaN
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Найдено {nan_count} NaN значений в признаках")

        # Проверка на бесконечные значения
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.error(f"Найдено {inf_count} бесконечных значений")
            return False

        # Проверка размера данных
        if len(features_df) < 100:
            logger.warning(f"Мало данных для обучения: {len(features_df)} записей")

        return True


class ModelUtils:
    """Утилиты для работы с моделями"""

    @staticmethod
    def save_model_info(
        model, crypto_name: str, model_type: str, metrics: Dict, parameters: Dict
    ):
        """Сохранение информации о модели"""
        model_info = {
            "crypto_name": crypto_name,
            "model_type": model_type,
            "metrics": metrics,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "model_summary": ModelUtils._get_model_summary(model),
        }

        info_file = DIRS["models"] / f"{crypto_name}_{model_type}_info.json"
        with open(info_file, "w") as f:
            json.dump(model_info, f, indent=4, default=str)

        logger.info(f"Информация о модели сохранена: {info_file}")

    @staticmethod
    def _get_model_summary(model) -> Dict:
        """Получение сводки о модели"""
        return {
            "total_params": model.count_params(),
            "trainable_params": int(
                np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            ),
            "layers_count": len(model.layers),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        }

    @staticmethod
    def load_model_info(crypto_name: str, model_type: str) -> Optional[Dict]:
        """Загрузка информации о модели"""
        info_file = DIRS["models"] / f"{crypto_name}_{model_type}_info.json"

        if info_file.exists():
            with open(info_file, "r") as f:
                return json.load(f)
        return None

    @staticmethod
    def calculate_model_confidence(predictions: np.ndarray, actual: np.ndarray) -> Dict:
        """Расчет доверительных интервалов"""
        errors = predictions - actual

        # Стандартное отклонение ошибок
        error_std = np.std(errors)

        # Доверительные интервалы (95%)
        confidence_95 = 1.96 * error_std

        # Процент предсказаний в доверительном интервале
        within_confidence = np.abs(errors) <= confidence_95
        confidence_accuracy = np.mean(within_confidence) * 100

        return {
            "error_std": float(error_std),
            "confidence_95": float(confidence_95),
            "confidence_accuracy": float(confidence_accuracy),
        }


class FileManager:
    """Управление файлами проекта"""

    @staticmethod
    def cleanup_old_files(days_old: int = 30):
        """Очистка старых файлов"""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        for directory in DIRS.values():
            if not directory.exists():
                continue

            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        try:
                            file_path.unlink()
                            logger.info(f"Удален старый файл: {file_path}")
                        except Exception as e:
                            logger.error(f"Ошибка удаления {file_path}: {e}")

    @staticmethod
    def get_project_size() -> Dict:
        """Получение размера проекта по папкам"""
        sizes = {}

        for name, directory in DIRS.items():
            if directory.exists():
                total_size = sum(
                    f.stat().st_size for f in directory.rglob("*") if f.is_file()
                )
                sizes[name] = {
                    "bytes": total_size,
                    "mb": round(total_size / (1024 * 1024), 2),
                    "files_count": len(list(directory.rglob("*"))),
                }
            else:
                sizes[name] = {"bytes": 0, "mb": 0, "files_count": 0}

        return sizes

    @staticmethod
    def export_results_to_excel(session_results: Dict, filename: str = None):
        """Экспорт результатов в Excel"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_analysis_results_{timestamp}.xlsx"

        file_path = DIRS["metrics"] / filename

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Сводная таблица метрик
            metrics_data = []
            for crypto_name, results in session_results.items():
                for model_key, metrics in results.get("metrics", {}).items():
                    model_type = model_key.split("_")[-1]
                    metrics_data.append(
                        {"Криптовалюта": crypto_name, "Модель": model_type, **metrics}
                    )

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name="Метрики", index=False)

            # Детальные данные по каждой криптовалюте
            for crypto_name, results in session_results.items():
                if "features_df" in results:
                    features_df = results["features_df"]
                    sheet_name = crypto_name[
                        :31
                    ]  # Ограничение Excel на длину имени листа
                    features_df.to_excel(writer, sheet_name=sheet_name)

        logger.info(f"Результаты экспортированы в Excel: {file_path}")
        return file_path


class MetricsCalculator:
    """Расширенные метрики для оценки моделей"""

    @staticmethod
    def calculate_advanced_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray = None
    ) -> Dict:
        """Расчет продвинутых метрик"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {}

        # Базовые метрики
        metrics["MSE"] = float(mean_squared_error(y_true, y_pred))
        metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))
        metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
        metrics["R2"] = float(r2_score(y_true, y_pred))

        # MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["MAPE"] = float(mape) if np.isfinite(mape) else float("inf")

        # Symmetric MAPE
        smape = (
            np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
            * 100
        )
        metrics["SMAPE"] = float(smape)

        # Directional Accuracy
        if len(y_true) > 1:
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics["Directional_Accuracy"] = float(
                np.mean(actual_direction == pred_direction) * 100
            )

        # Theil's U статистика
        if len(y_true) > 1:
            theil_u = MetricsCalculator._calculate_theil_u(y_true, y_pred)
            metrics["Theil_U"] = float(theil_u)

        # Максимальная ошибка
        metrics["Max_Error"] = float(np.max(np.abs(y_true - y_pred)))

        # Процент предсказаний в пределах определенной точности
        error_5pct = np.abs((y_true - y_pred) / y_true) <= 0.05
        error_10pct = np.abs((y_true - y_pred) / y_true) <= 0.10

        metrics["Accuracy_5pct"] = float(np.mean(error_5pct) * 100)
        metrics["Accuracy_10pct"] = float(np.mean(error_10pct) * 100)

        return metrics

    @staticmethod
    def _calculate_theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Расчет Theil's U статистики"""
        actual_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)

        numerator = np.sqrt(np.mean((actual_changes - pred_changes) ** 2))
        denominator = np.sqrt(np.mean(actual_changes**2)) + np.sqrt(
            np.mean(pred_changes**2)
        )

        return numerator / denominator if denominator != 0 else 0


class ConfigManager:
    """Управление конфигурацией"""

    @staticmethod
    def save_user_preferences(preferences: Dict):
        """Сохранение пользовательских настроек"""
        config_file = DIRS["metrics"] / "user_preferences.json"

        with open(config_file, "w") as f:
            json.dump(preferences, f, indent=4)

        logger.info("Пользовательские настройки сохранены")

    @staticmethod
    def load_user_preferences() -> Dict:
        """Загрузка пользовательских настроек"""
        config_file = DIRS["metrics"] / "user_preferences.json"

        if config_file.exists():
            with open(config_file, "r") as f:
                return json.load(f)

        # Настройки по умолчанию
        return {
            "default_period": "2y",
            "default_epochs": 50,
            "default_models": ["LSTM"],
            "use_macro": True,
            "auto_save_plots": True,
            "verbose": False,
        }


class Logger:
    """Класс для управления логированием"""

    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: str = None):
        """Настройка логирования"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        if log_file is None:
            log_file = (
                DIRS["logs"]
                / f"crypto_predictor_{datetime.now().strftime('%Y%m%d')}.log"
            )

        # Настройка файлового обработчика
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))

        # Настройка консольного обработчика
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))

        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        logger.info(f"Логирование настроено. Файл: {log_file}")


def create_project_structure():
    """Создание структуры проекта"""
    print("📁 Создание структуры проекта...")

    for name, directory in DIRS.items():
        if not directory.exists():
            directory.mkdir(exist_ok=True, parents=True)
            print(f"✓ Создана папка: {directory}")
        else:
            print(f"  Папка уже существует: {directory}")

    # Создание requirements.txt
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        requirements = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "scikit-learn>=0.24.0",
            "tensorflow>=2.6.0",
            "yfinance>=0.1.70",
            "plotly>=5.0.0",
            "openpyxl>=3.0.0",
            "joblib>=1.0.0",
        ]

        with open(requirements_path, "w") as f:
            f.write("\n".join(requirements))

        print(f"✓ Создан файл: {requirements_path}")

    print("\n✅ Структура проекта создана успешно!")


def format_price(price: float, currency: str = "$") -> str:
    """Форматирование цены"""
    return f"{currency}{price:,.2f}"


def format_percentage(value: float) -> str:
    """Форматирование процентов"""
    return f"{value:+.2f}%"


def calculate_profit_loss(
    initial_price: float, final_price: float, amount: float = 1.0
) -> Dict:
    """Расчет прибыли/убытка"""
    profit_loss = (final_price - initial_price) * amount
    profit_loss_pct = ((final_price - initial_price) / initial_price) * 100

    return {
        "initial_value": initial_price * amount,
        "final_value": final_price * amount,
        "profit_loss": profit_loss,
        "profit_loss_pct": profit_loss_pct,
        "is_profit": profit_loss > 0,
    }


def generate_report_summary(session_results: Dict) -> str:
    """Генерация текстового отчета"""
    report = []
    report.append("=" * 60)
    report.append("📊 ОТЧЕТ ПО АНАЛИЗУ КРИПТОВАЛЮТ")
    report.append("=" * 60)
    report.append(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    report.append("")

    for crypto_name, results in session_results.items():
        report.append(f"\n🪙 {crypto_name}")
        report.append("-" * 40)

        # Метрики моделей
        for model_key, metrics in results.get("metrics", {}).items():
            model_type = model_key.split("_")[-1]
            report.append(f"\nМодель: {model_type}")
            report.append(f"  RMSE: ${metrics['RMSE']:.2f}")
            report.append(f"  MAE: ${metrics['MAE']:.2f}")
            report.append(f"  R²: {metrics['R2']:.4f}")
            report.append(f"  MAPE: {metrics['MAPE']:.2f}%")
            report.append(
                f"  Точность направления: {metrics['Directional_Accuracy']:.1f}%"
            )

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def save_report(report_text: str, filename: str = None):
    """Сохранение отчета в файл"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crypto_report_{timestamp}.txt"

    report_path = DIRS["metrics"] / filename

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Отчет сохранен: {report_path}")
    return report_path
