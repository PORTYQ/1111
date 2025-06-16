"""
Обновленный конфигурационный файл для проекта прогнозирования криптовалют
С поддержкой новых моделей и расширенных параметров
"""

import os
from pathlib import Path

# Базовые настройки проекта
PROJECT_NAME = "CryptoPricePredictor v2.0"
BASE_DIR = Path(__file__).parent

# Директории для сохранения данных
DIRS = {
    "models": BASE_DIR / "models",
    "plots": BASE_DIR / "plots",
    "metrics": BASE_DIR / "metrics",
    "datasets": BASE_DIR / "datasets",
    "logs": BASE_DIR / "logs",
    "cache": BASE_DIR / "cache",  # Для кэширования признаков
}

# Создание директорий
for dir_path in DIRS.values():
    dir_path.mkdir(exist_ok=True)

# Конфигурация криптовалют
CRYPTO_SYMBOLS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Binance Coin": "BNB-USD",
    "Cardano": "ADA-USD",
    "Solana": "SOL-USD",
    "Polygon": "MATIC-USD",
    "Ripple": "XRP-USD",  # Добавлено
    "Avalanche": "AVAX-USD",  # Добавлено
}

# Конфигурация макропоказателей
MACRO_INDICATORS = {
    "DXY": "DX-Y.NYB",  # Индекс доллара США
    "Gold": "GC=F",  # Золото
    "VIX": "^VIX",  # Индекс волатильности
    "TNX": "^TNX",  # 10-летние облигации США
    "Oil": "CL=F",  # Нефть WTI - добавлено
    "SP500": "^GSPC",  # S&P 500 - добавлено
}

# Параметры модели по умолчанию (оптимизированные для v2.0)
MODEL_CONFIG = {
    "lookback_window": 40,  # Увеличено с 30
    "train_test_split": 0.85,
    "batch_size": 32,  # Увеличено с 16
    "epochs": 150,  # Увеличено со 100
    "patience": 30,  # Увеличено с 20
    "validation_split": 0.15,
    "learning_rate": 0.0001,  # Добавлено
    "early_stopping_min_delta": 0.0001,  # Добавлено
}

# Параметры для разных типов моделей
MODEL_SPECIFIC_CONFIG = {
    "LSTM": {
        "units": [50, 30, 20],
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "epochs": 100,
    },
    "GRU": {
        "units": [50, 30, 20],
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "epochs": 100,
    },
    "LSTM_ADVANCED": {
        "units": [128, 64],
        "dropout": 0.3,
        "attention_heads": 4,
        "epochs": 150,
        "patience": 40,
    },
    "TRANSFORMER": {
        "num_heads": 4,
        "key_dim": 64,
        "ff_dim": 256,
        "num_blocks": 2,
        "epochs": 200,
        "patience": 50,
    },
    "CNN_LSTM": {
        "cnn_filters": [64, 128],
        "kernel_size": 3,
        "lstm_units": [100, 50],
        "dropout": 0.3,
        "epochs": 120,
    },
}

# Параметры LSTM/GRU (базовые)
LSTM_CONFIG = {
    "units": [50, 30, 20],
    "dropout": 0.2,
    "recurrent_dropout": 0.0,
}

GRU_CONFIG = {
    "units": [50, 30, 20],
    "dropout": 0.2,
    "recurrent_dropout": 0.0,
}

# Настройки визуализации
PLOT_CONFIG = {
    "figsize": (15, 10),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "colors": {
        "actual": "#1f77b4",
        "predicted": "#ff7f0e",
        "forecast": "#d62728",
        "macro": "#2ca02c",
    },
    "theme": "plotly_white",  # Добавлено
}

# Периоды загрузки данных
DATA_PERIODS = {
    "3 месяца": "3mo",  # Добавлено
    "6 месяцев": "6mo",  # Добавлено
    "1 год": "1y",
    "2 года": "2y",
    "3 года": "3y",
    "5 лет": "5y",
    "Макс": "max",  # Добавлено
}

# Горизонты прогнозирования
FORECAST_HORIZONS = {
    "1 день": 1,  # Добавлено
    "3 дня": 3,  # Добавлено
    "1 неделя": 7,
    "2 недели": 14,
    "1 месяц": 30,
    "2 месяца": 60,  # Добавлено
    "3 месяца": 90,
    "6 месяцев": 180,  # Добавлено
}

# Параметры для продвинутой инженерии признаков
FEATURE_ENGINEERING_CONFIG = {
    "use_microstructure": True,
    "use_sentiment": True,
    "use_whale_detection": True,
    "use_regime_detection": True,
    "use_network_features": True,
    "use_cyclical": True,
    "max_features": 150,  # Увеличено со 100
    "pca_variance_threshold": 0.95,
    "correlation_threshold": 0.95,
}

# Параметры для ансамблевых моделей
ENSEMBLE_CONFIG = {
    "min_models": 2,
    "weight_optimization": "performance",  # 'equal', 'performance', 'optimization'
    "confidence_level": 0.95,
    "bootstrap_samples": 100,
}

# Параметры для оптимизации гиперпараметров
HYPEROPT_CONFIG = {
    "max_trials": 20,
    "objective": "val_loss",
    "directory": "hyperopt_results",
    "overwrite": True,
}

# Настройки для мониторинга и логирования
MONITORING_CONFIG = {
    "log_level": "INFO",
    "tensorboard": True,
    "mlflow_tracking": False,
    "save_checkpoints": True,
    "checkpoint_frequency": 10,  # эпох
}

# Лимиты и ограничения
LIMITS = {
    "max_features": 200,
    "max_lookback": 100,
    "min_data_points": 500,
    "max_epochs": 500,
    "max_forecast_days": 365,
}

# Параметры валидации и тестирования
VALIDATION_CONFIG = {
    "walk_forward_splits": 5,
    "test_size": 0.15,
    "gap_days": 0,  # Дни между train и test для избежания утечки
    "purge_days": 2,  # Дни для очистки между фолдами
}

# Настройки производительности
PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "mixed_precision": True,
    "cache_features": True,
    "parallel_models": False,
    "num_workers": 4,
}

# Пороги для оценки качества моделей
QUALITY_THRESHOLDS = {
    "excellent": {"R2": 0.85, "directional_accuracy": 0.70},
    "good": {"R2": 0.70, "directional_accuracy": 0.60},
    "acceptable": {"R2": 0.50, "directional_accuracy": 0.55},
    "poor": {"R2": 0.30, "directional_accuracy": 0.50},
}

# Настройки для торговых сигналов
TRADING_CONFIG = {
    "signal_threshold": 0.05,  # 5% изменение для сигнала
    "stop_loss": 0.03,  # 3% стоп-лосс
    "take_profit": 0.10,  # 10% тейк-профит
    "position_size": 0.1,  # 10% от капитала на позицию
    "max_positions": 3,
}

# Веб-интерфейс и API настройки
WEB_CONFIG = {
    "host": "127.0.0.1",
    "port": 8080,
    "debug": False,
    "auto_reload": True,
}

# Настройки базы данных (если используется)
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": BASE_DIR / "crypto_predictions.db",
    "echo": False,
}

# Настройки уведомлений
NOTIFICATION_CONFIG = {
    "email_enabled": False,
    "email_smtp": "smtp.gmail.com",
    "email_port": 587,
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
}

# Экспортные настройки
EXPORT_CONFIG = {
    "formats": ["csv", "excel", "json", "html"],
    "compress": True,
    "include_metadata": True,
    "timestamp_format": "%Y%m%d_%H%M%S",
}

# Настройки безопасности
SECURITY_CONFIG = {
    "api_key_required": False,
    "rate_limit": 100,  # запросов в минуту
    "allowed_origins": ["*"],
    "ssl_enabled": False,
}

# Предустановленные стратегии
PRESET_STRATEGIES = {
    "conservative": {
        "models": ["LSTM", "GRU"],
        "epochs": 100,
        "features": ["basic", "technical"],
        "forecast_horizon": 7,
    },
    "balanced": {
        "models": ["LSTM_ADVANCED", "GRU", "CNN_LSTM"],
        "epochs": 150,
        "features": ["all"],
        "forecast_horizon": 30,
    },
    "aggressive": {
        "models": ["TRANSFORMER", "LSTM_ADVANCED", "CNN_LSTM"],
        "epochs": 200,
        "features": ["all_advanced"],
        "forecast_horizon": 90,
    },
}

# Словарь описаний для UI
UI_DESCRIPTIONS = {
    "models": {
        "LSTM": "Long Short-Term Memory - классическая архитектура для временных рядов",
        "GRU": "Gated Recurrent Unit - эффективная альтернатива LSTM",
        "LSTM_ADVANCED": "Bidirectional LSTM с механизмом внимания",
        "TRANSFORMER": "Современная архитектура с self-attention",
        "CNN_LSTM": "Гибридная модель для захвата локальных и временных паттернов",
    },
    "features": {
        "microstructure": "Анализ микроструктуры рынка (spread, ликвидность)",
        "sentiment": "Индикаторы рыночных настроений",
        "whale": "Обнаружение активности крупных игроков",
        "regime": "Определение рыночных режимов",
        "network": "Корреляции между криптовалютами",
        "cyclical": "Циклические и сезонные паттерны",
    },
}

# Версия конфигурации
CONFIG_VERSION = "2.0.0"


# Функция для получения оптимальных параметров модели
def get_model_config(model_type: str) -> dict:
    """Получить конфигурацию для конкретного типа модели"""
    base_config = MODEL_CONFIG.copy()
    if model_type in MODEL_SPECIFIC_CONFIG:
        base_config.update(MODEL_SPECIFIC_CONFIG[model_type])
    return base_config


# Функция для валидации конфигурации
def validate_config() -> bool:
    """Проверить корректность конфигурации"""
    try:
        # Проверка существования директорий
        for name, path in DIRS.items():
            if not path.exists():
                print(f"Создание директории: {path}")
                path.mkdir(parents=True, exist_ok=True)

        # Проверка параметров
        assert MODEL_CONFIG["lookback_window"] > 0
        assert MODEL_CONFIG["epochs"] > 0
        assert 0 < MODEL_CONFIG["train_test_split"] < 1

        return True
    except Exception as e:
        print(f"Ошибка валидации конфигурации: {e}")
        return False


# Автоматическая валидация при импорте
if __name__ != "__main__":
    validate_config()
