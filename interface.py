"""
Обновленный пользовательский интерфейс для системы прогнозирования криптовалют
С поддержкой новых моделей и расширенных функций
"""
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from data_loader import DataLoader
from models import ModelBuilder
from visualizer import Visualizer
from feature_engineering import AdvancedFeatureEngineering
from config import (
    CRYPTO_SYMBOLS,
    MACRO_INDICATORS,
    DATA_PERIODS,
    FORECAST_HORIZONS,
    DIRS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoPredictorInterface:
    """Основной интерфейс системы прогнозирования с поддержкой новых возможностей"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.model_builder = ModelBuilder()
        self.visualizer = Visualizer()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.session_results = {}
        self.available_models = ["LSTM", "GRU", "LSTM_ADVANCED", "TRANSFORMER", "CNN_LSTM"]

    def display_welcome_message(self):
        """Приветственное сообщение"""
        print("=" * 60)
        print("🚀 СИСТЕМА ПРОГНОЗИРОВАНИЯ КРИПТОВАЛЮТ V2.0 🚀")
        print("=" * 60)
        print("Добро пожаловать в улучшенную систему машинного обучения")
        print("для прогнозирования курсов криптовалют!")
        print("\nНовые возможности:")
        print("✨ Продвинутые архитектуры моделей (Transformer, CNN-LSTM)")
        print("✨ Расширенная инженерия признаков (100+ индикаторов)")
        print("✨ Ансамблевые прогнозы с доверительными интервалами")
        print("✨ Автоматическая оптимизация гиперпараметров")
        print("=" * 60)
        print()

    def select_models(self) -> List[str]:
        """Выбор моделей для обучения"""
        print("\n🤖 ВЫБОР МОДЕЛЕЙ")
        print("-" * 30)

        print("Доступные модели:")
        model_descriptions = {
            "LSTM": "Базовая LSTM - быстрая, надежная",
            "GRU": "Базовая GRU - эффективная альтернатива LSTM",
            "LSTM_ADVANCED": "Продвинутая LSTM с Attention - высокая точность",
            "TRANSFORMER": "Transformer - передовая архитектура",
            "CNN_LSTM": "CNN-LSTM гибрид - захват локальных паттернов"
        }
        
        for i, (model, desc) in enumerate(model_descriptions.items(), 1):
            print(f"{i}. {model} - {desc}")
        
        print(f"{len(model_descriptions) + 1}. Все базовые модели (LSTM, GRU)")
        print(f"{len(model_descriptions) + 2}. Все продвинутые модели")
        print(f"{len(model_descriptions) + 3}. Все модели")

        while True:
            try:
                choice = input("\nВыберите модели (номер или несколько через запятую): ")
                
                if choice.strip() == str(len(model_descriptions) + 1):
                    return ["LSTM", "GRU"]
                elif choice.strip() == str(len(model_descriptions) + 2):
                    return ["LSTM_ADVANCED", "TRANSFORMER", "CNN_LSTM"]
                elif choice.strip() == str(len(model_descriptions) + 3):
                    return list(model_descriptions.keys())
                
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [
                    list(model_descriptions.keys())[i] 
                    for i in indices 
                    if 0 <= i < len(model_descriptions)
                ]
                
                if selected:
                    print(f"Выбраны: {', '.join(selected)}")
                    return selected
                else:
                    print("❌ Неверный выбор! Попробуйте снова.")

            except (ValueError, IndexError):
                print("❌ Неверный формат! Введите номера через запятую.")

    def select_feature_engineering_options(self) -> Dict:
        """Выбор опций для инженерии признаков"""
        print("\n🛠️ НАСТРОЙКА ИНЖЕНЕРИИ ПРИЗНАКОВ")
        print("-" * 30)
        
        options = {}
        
        # Использование продвинутых признаков
        print("\nДоступные группы признаков:")
        feature_groups = {
            "microstructure": "Микроструктура рынка (spread, VPIN, ликвидность)",
            "sentiment": "Индикаторы настроений (Fear & Greed компоненты)",
            "whale": "Детекция китов и манипуляций",
            "regime": "Определение рыночных режимов",
            "network": "Корреляции с другими криптовалютами",
            "cyclical": "Циклические паттерны (Фурье анализ)"
        }
        
        for key, desc in feature_groups.items():
            use = input(f"Использовать {desc}? (y/n): ").lower() == 'y'
            options[f"use_{key}"] = use
        
        # PCA
        use_pca = input("\nПрименить PCA для снижения размерности? (y/n): ").lower() == 'y'
        if use_pca:
            n_components = input("Процент объясненной дисперсии (0.95 по умолчанию): ")
            options["pca_components"] = float(n_components) if n_components else 0.95
        else:
            options["pca_components"] = None
        
        # Отбор признаков
        use_selection = input("Применить отбор признаков? (y/n): ").lower() == 'y'
        if use_selection:
            n_features = input("Количество признаков (100 по умолчанию): ")
            options["n_features"] = int(n_features) if n_features else 100
        else:
            options["n_features"] = None
        
        return options

    def select_cryptocurrencies(self) -> List[str]:
        """Выбор криптовалют для анализа"""
        print("\n📈 ВЫБОР КРИПТОВАЛЮТ")
        print("-" * 30)

        crypto_list = list(CRYPTO_SYMBOLS.keys())

        print("Доступные криптовалюты:")
        for i, crypto in enumerate(crypto_list, 1):
            print(f"{i:2d}. {crypto}")

        print(f"{len(crypto_list) + 1:2d}. Все криптовалюты")

        while True:
            try:
                choice = input(
                    "\nВыберите криптовалюту (номер или несколько через запятую): "
                )

                if choice.strip() == str(len(crypto_list) + 1):
                    return crypto_list

                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [
                    crypto_list[i] for i in indices if 0 <= i < len(crypto_list)
                ]

                if selected:
                    print(f"Выбраны: {', '.join(selected)}")
                    return selected
                else:
                    print("❌ Неверный выбор! Попробуйте снова.")

            except (ValueError, IndexError):
                print("❌ Неверный формат! Введите номера через запятую.")

    def run_analysis(
        self,
        cryptocurrencies: List[str],
        models: List[str],
        use_macro: bool,
        macro_indicators: List[str],
        parameters: Dict,
        feature_options: Dict = None
    ):
        """Запуск анализа с поддержкой новых возможностей"""
        print("\n🔄 ЗАПУСК АНАЛИЗА")
        print("=" * 50)

        # Загрузка данных
        print("📥 Загрузка данных...")
        crypto_data = self.data_loader.fetch_crypto_data(
            cryptocurrencies, parameters["data_period"]
        )

        if use_macro:
            macro_data = self.data_loader.fetch_macro_data(
                macro_indicators, parameters["data_period"]
            )
        else:
            macro_data = {}

        # Анализ каждой криптовалюты
        for crypto_name in cryptocurrencies:
            if crypto_name not in crypto_data:
                print(f"⚠️ Пропуск {crypto_name} - данные не загружены")
                continue

            print(f"\n🔍 Анализ {crypto_name}")
            print("-" * 40)

            try:
                # Базовая подготовка признаков
                features_df = self.data_loader.prepare_features(crypto_name, use_macro)
                
                # Продвинутая инженерия признаков
                if feature_options:
                    print("🛠️ Применение продвинутой инженерии признаков...")
                    features_df = self._apply_advanced_features(
                        crypto_name, features_df, crypto_data, feature_options
                    )

                print(
                    f"📊 Подготовлено {len(features_df)} записей с {len(features_df.columns)} признаками"
                )

                # Визуализация
                self._create_analysis_visualizations(crypto_name, crypto_data, features_df)

                crypto_results = {
                    "features_df": features_df,
                    "models": {},
                    "predictions": {},
                    "metrics": {},
                }

                # Обучение моделей
                for model_type in models:
                    print(f"\n🚀 Обучение модели {model_type}...")

                    result = self.model_builder.train_model(
                        crypto_name,
                        model_type,
                        features_df,
                        epochs=parameters["epochs"],
                    )

                    model_key = f"{crypto_name}_{model_type}"
                    crypto_results["models"][model_type] = result["model"]
                    crypto_results["predictions"][model_key] = result["predictions"]
                    crypto_results["metrics"][model_key] = result["metrics"]

                    # Визуализация результатов
                    self._create_model_visualizations(
                        crypto_name, model_type, result, crypto_data
                    )

                    # Вывод метрик
                    self._print_metrics(result["metrics"], crypto_name, model_type)

                # Ансамблевый прогноз если моделей больше одной
                if len(models) > 1:
                    print(f"\n🎯 Создание ансамблевого прогноза для {crypto_name}...")
                    self._create_ensemble_forecast(
                        crypto_name, features_df, models, parameters["forecast_days"]
                    )

                # Сравнение моделей
                if len(models) > 1:
                    print(f"\n📊 Сравнение моделей для {crypto_name}...")
                    self._compare_and_visualize_models(crypto_name, crypto_results)

                self.session_results[crypto_name] = crypto_results

            except Exception as e:
                logger.error(f"Ошибка при анализе {crypto_name}: {e}")
                print(f"❌ Ошибка при анализе {crypto_name}: {e}")

        # Сохранение сессии
        self._save_session_results(parameters, feature_options)

        print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 50)
        self._print_session_summary()

    def _apply_advanced_features(
        self, crypto_name: str, features_df: pd.DataFrame, 
        crypto_data: Dict, feature_options: Dict
    ) -> pd.DataFrame:
        """Применение продвинутой инженерии признаков"""
        
        # Создание продвинутых признаков
        advanced_features = []
        
        if feature_options.get("use_microstructure", False):
            print("  - Добавление микроструктурных признаков...")
            microstructure = self.feature_engineer.create_market_microstructure_features(
                crypto_data[crypto_name]
            )
            advanced_features.append(microstructure)
        
        if feature_options.get("use_sentiment", False):
            print("  - Добавление признаков настроений...")
            sentiment = self.feature_engineer.create_sentiment_features(
                crypto_data[crypto_name]
            )
            advanced_features.append(sentiment)
        
        if feature_options.get("use_whale", False):
            print("  - Добавление признаков детекции китов...")
            whale = self.feature_engineer.create_whale_detection_features(
                crypto_data[crypto_name]
            )
            advanced_features.append(whale)
        
        if feature_options.get("use_regime", False):
            print("  - Добавление режимных признаков...")
            regime = self.feature_engineer.create_regime_features(
                crypto_data[crypto_name]
            )
            advanced_features.append(regime)
        
        if feature_options.get("use_network", False) and len(crypto_data) > 1:
            print("  - Добавление сетевых признаков...")
            other_cryptos = {k: v for k, v in crypto_data.items() if k != crypto_name}
            network = self.feature_engineer.create_network_features(
                crypto_data[crypto_name], other_cryptos
            )
            advanced_features.append(network)
        
        if feature_options.get("use_cyclical", False):
            print("  - Добавление циклических признаков...")
            cyclical = self.feature_engineer.create_cyclical_features(
                crypto_data[crypto_name]
            )
            advanced_features.append(cyclical)
        
        # Объединение признаков
        if advanced_features:
            all_features = pd.concat([features_df] + advanced_features, axis=1)
            
            # Очистка от NaN
            all_features = all_features.replace([np.inf, -np.inf], np.nan)
            all_features = all_features.fillna(method='ffill').fillna(0)
            
            # Отбор признаков
            if feature_options.get("n_features"):
                print(f"  - Отбор {feature_options['n_features']} лучших признаков...")
                all_features = self.feature_engineer.apply_feature_selection(
                    all_features, all_features['Close'], feature_options['n_features']
                )
            
            # PCA
            if feature_options.get("pca_components"):
                print(f"  - Применение PCA...")
                pca_features = self.feature_engineer.apply_pca_transformation(
                    all_features.drop('Close', axis=1), 
                    feature_options['pca_components']
                )
                all_features = pd.concat([all_features[['Close']], pca_features], axis=1)
            
            return all_features
        
        return features_df

    def _create_analysis_visualizations(
        self, crypto_name: str, crypto_data: Dict, features_df: pd.DataFrame
    ):
        """Создание визуализаций для анализа"""
        # График истории цен
        price_fig = self.visualizer.plot_price_history(crypto_data, crypto_name)
        
        # Анализ важности признаков
        importance_fig = self.visualizer.plot_feature_importance(features_df, crypto_name)

    def _create_model_visualizations(
        self, crypto_name: str, model_type: str, result: Dict, crypto_data: Dict
    ):
        """Создание визуализаций для модели"""
        # График предсказаний
        pred_fig = self.visualizer.plot_predictions(
            result["predictions"], crypto_name, model_type
        )
        
        # График истории обучения
        training_fig = self.visualizer.plot_training_history(
            result["history"], crypto_name, model_type
        )

    def _create_ensemble_forecast(
        self, crypto_name: str, features_df: pd.DataFrame, 
        models: List[str], days_ahead: int
    ):
        """Создание ансамблевого прогноза"""
        try:
            dates, ensemble_forecast, ensemble_info = self.model_builder.ensemble_predict(
                crypto_name, features_df, days_ahead, models
            )
            
            # Визуализация ансамблевого прогноза
            fig = self.visualizer.plot_ensemble_forecast(
                features_df['Close'],
                dates,
                ensemble_forecast,
                ensemble_info['confidence_intervals'],
                crypto_name
            )
            
            # Вывод информации
            print(f"\n📊 Ансамблевый прогноз:")
            print(f"   Использованные модели: {', '.join(models)}")
            print(f"   Веса моделей:")
            for model, weight in ensemble_info['weights'].items():
                print(f"     - {model}: {weight:.3f}")
            
            current_price = features_df['Close'].iloc[-1]
            final_price = ensemble_forecast[-1]
            change_percent = ((final_price - current_price) / current_price) * 100
            
            print(f"   Прогноз на {days_ahead} дней:")
            print(f"     Текущая цена: ${current_price:.2f}")
            print(f"     Прогнозная цена: ${final_price:.2f}")
            print(f"     Изменение: {change_percent:+.2f}%")
            print(f"     95% доверительный интервал: "
                  f"${ensemble_info['confidence_intervals']['lower_95'][-1]:.2f} - "
                  f"${ensemble_info['confidence_intervals']['upper_95'][-1]:.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка создания ансамблевого прогноза: {e}")

    def _compare_and_visualize_models(self, crypto_name: str, crypto_results: Dict):
        """Сравнение и визуализация моделей"""
        comparison_fig = self.visualizer.plot_model_comparison(
            crypto_results["metrics"], crypto_name
        )
        
        # Создание сводной панели
        summary_fig = self.visualizer.create_summary_dashboard(
            crypto_name,
            crypto_results["predictions"],
            crypto_results["metrics"],
        )
        
        self._print_model_comparison(crypto_results["metrics"], crypto_name)

    def _print_metrics(self, metrics: Dict, crypto_name: str, model_type: str):
        """Вывод метрик модели"""
        print(f"\n📈 Метрики модели {model_type} для {crypto_name}:")
        print(f"   RMSE: ${metrics['RMSE']:.2f}")
        print(f"   MAE:  ${metrics['MAE']:.2f}")
        print(f"   R²:   {metrics['R2']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        print(f"   Точность направления: {metrics['Directional_Accuracy']:.1f}%")

    def _print_model_comparison(self, metrics_dict: Dict, crypto_name: str):
        """Вывод сравнения моделей"""
        print(f"\n🔍 Сравнение моделей для {crypto_name}:")
        print("-" * 80)
        print(
            f"{'Модель':<15} {'RMSE':<10} {'MAE':<10} {'R²':<8} {'MAPE':<8} {'Точность':<10}"
        )
        print("-" * 80)

        # Сортировка по RMSE
        sorted_models = sorted(metrics_dict.items(), key=lambda x: x[1]['RMSE'])
        
        for model_key, metrics in sorted_models:
            model_type = model_key.split("_", 1)[1]
            print(
                f"{model_type:<15} "
                f"${metrics['RMSE']:<9.2f} "
                f"${metrics['MAE']:<9.2f} "
                f"{metrics['R2']:<7.4f} "
                f"{metrics['MAPE']:<7.1f}% "
                f"{metrics['Directional_Accuracy']:<9.1f}%"
            )

        # Определение лучшей модели
        best_model = sorted_models[0]
        best_type = best_model[0].split("_", 1)[1]
        print(f"\n🏆 Лучшая модель по RMSE: {best_type}")
        
        # Рекомендации
        best_r2 = best_model[1]['R2']
        if best_r2 > 0.85:
            recommendation = "Отличное качество! Модель готова к использованию."
        elif best_r2 > 0.70:
            recommendation = "Хорошее качество. Рекомендуется дополнительная валидация."
        elif best_r2 > 0.50:
            recommendation = "Удовлетворительное качество. Рассмотрите улучшение признаков."
        else:
            recommendation = "Требуется улучшение. Попробуйте другие архитектуры или больше данных."
        
        print(f"💡 Рекомендация: {recommendation}")

    def _save_session_results(self, parameters: Dict, feature_options: Dict = None):
        """Сохранение результатов сессии"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Подготовка данных для сохранения
        session_data = {
            "timestamp": timestamp, 
            "parameters": parameters,
            "feature_options": feature_options,
            "results": {}
        }

        for crypto_name, results in self.session_results.items():
            session_data["results"][crypto_name] = {
                "metrics": results["metrics"],
                "data_info": self.data_loader.get_data_info(crypto_name),
                "models_used": list(results["models"].keys())
            }

        # Сохранение в JSON
        session_file = DIRS["metrics"] / f"session_{timestamp}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=4, ensure_ascii=False, default=str)

        print(f"\n💾 Результаты сессии сохранены: {session_file}")

    def _print_session_summary(self):
        """Вывод итогов сессии"""
        print(f"\n📋 ИТОГИ СЕССИИ")
        print("-" * 30)
        print(f"Проанализировано криптовалют: {len(self.session_results)}")

        total_models = sum(
            len(results["models"]) for results in self.session_results.values()
        )
        print(f"Обучено моделей: {total_models}")

        # Лучшая модель по всем криптовалютам
        best_overall = None
        best_rmse = float('inf')
        
        for crypto_name, results in self.session_results.items():
            for model_key, metrics in results["metrics"].items():
                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    best_overall = (crypto_name, model_key.split("_", 1)[1])
        
        if best_overall:
            print(f"\n🏆 Лучшая модель overall: {best_overall[1]} для {best_overall[0]}")
            print(f"   RMSE: ${best_rmse:.2f}")

        print(f"\n📁 Файлы сохранены в:")
        print(f"   📊 Модели: {DIRS['models']}")
        print(f"   📈 Графики: {DIRS['plots']}")
        print(f"   📋 Метрики: {DIRS['metrics']}")
        print(f"   📄 Датасеты: {DIRS['datasets']}")

    def select_macro_indicators(self) -> Tuple[bool, List[str]]:
        """Выбор макропоказателей"""
        print("\n📊 МАКРОПОКАЗАТЕЛИ")
        print("-" * 30)

        use_macro = input("Включить макропоказатели в анализ? (y/n): ").lower() == "y"

        if not use_macro:
            return False, []

        macro_list = list(MACRO_INDICATORS.keys())

        print("\nДоступные макропоказатели:")
        for i, indicator in enumerate(macro_list, 1):
            description = {
                "DXY": "Индекс доллара США",
                "Gold": "Золото",
                "VIX": "Индекс волатильности",
                "TNX": "10-летние облигации США",
            }
            print(f"{i:2d}. {indicator} - {description.get(indicator, '')}")

        print(f"{len(macro_list) + 1:2d}. Все показатели")

        while True:
            try:
                choice = input(
                    "\nВыберите показатели (номер или несколько через запятую): "
                )

                if choice.strip() == str(len(macro_list) + 1):
                    return True, macro_list

                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [macro_list[i] for i in indices if 0 <= i < len(macro_list)]

                if selected:
                    print(f"Выбраны: {', '.join(selected)}")
                    return True, selected
                else:
                    print("❌ Неверный выбор! Попробуйте снова.")

            except (ValueError, IndexError):
                print("❌ Неверный формат! Введите номера через запятую.")

    def select_parameters(self) -> Dict:
        """Выбор параметров обучения"""
        print("\n⚙️ ПАРАМЕТРЫ ОБУЧЕНИЯ")
        print("-" * 30)

        # Период данных
        print("Периоды загрузки данных:")
        periods = list(DATA_PERIODS.keys())
        for i, period in enumerate(periods, 1):
            print(f"{i}. {period}")

        while True:
            try:
                choice = int(input("Выберите период: ")) - 1
                if 0 <= choice < len(periods):
                    data_period = DATA_PERIODS[periods[choice]]
                    break
                else:
                    print("❌ Неверный выбор!")
            except ValueError:
                print("❌ Введите число!")

        # Горизонт прогнозирования
        print("\nГоризонт прогнозирования:")
        horizons = list(FORECAST_HORIZONS.keys())
        for i, horizon in enumerate(horizons, 1):
            print(f"{i}. {horizon}")

        while True:
            try:
                choice = int(input("Выберите горизонт: ")) - 1
                if 0 <= choice < len(horizons):
                    forecast_days = FORECAST_HORIZONS[horizons[choice]]
                    break
                else:
                    print("❌ Неверный выбор!")
            except ValueError:
                print("❌ Введите число!")

        # Количество эпох
        while True:
            try:
                epochs = int(
                    input("Количество эпох обучения (по умолчанию 100): ") or "100"
                )
                if epochs > 0:
                    break
                else:
                    print("❌ Количество эпох должно быть положительным!")
            except ValueError:
                print("❌ Введите число!")

        return {
            "data_period": data_period,
            "forecast_days": forecast_days,
            "epochs": epochs,
        }

    def run(self):
        """Главная функция запуска интерфейса"""
        self.display_welcome_message()

        while True:
            print("\n🎯 ГЛАВНОЕ МЕНЮ")
            print("-" * 20)
            print("1. 🚀 Новый анализ")
            print("2. 🔄 Загрузить предыдущую сессию")
            print("3. 🔮 Интерактивный прогноз")
            print("4. 📊 Просмотр результатов")
            print("5. 🎨 Создать кастомный прогноз")
            print("6. ❌ Выход")

            choice = input("\nВыберите действие: ")

            if choice == "1":
                self._run_new_analysis()
            elif choice == "2":
                self.load_previous_session()
            elif choice == "3":
                self.interactive_forecast()
            elif choice == "4":
                self._view_results()
            elif choice == "5":
                self._create_custom_forecast()
            elif choice == "6":
                print("\n👋 До свидания!")
                break
            else:
                print("❌ Неверный выбор!")

    def _run_new_analysis(self):
        """Запуск нового анализа"""
        cryptocurrencies = self.select_cryptocurrencies()
        use_macro, macro_indicators = self.select_macro_indicators()
        models = self.select_models()
        parameters = self.select_parameters()
        feature_options = self.select_feature_engineering_options()

        self.run_analysis(
            cryptocurrencies, models, use_macro, macro_indicators, 
            parameters, feature_options
        )

    def _create_custom_forecast(self):
        """Создание кастомного прогноза с выбором параметров"""
        print("\n🎨 КАСТОМНЫЙ ПРОГНОЗ")
        print("-" * 40)
        
        if not self.model_builder.models:
            print("❌ Нет обученных моделей. Сначала запустите обучение.")
            return
        
        # Выбор криптовалюты
        available_cryptos = list(set(key.split('_')[0] for key in self.model_builder.models.keys()))
        print("Доступные криптовалюты:")
        for i, crypto in enumerate(available_cryptos, 1):
            print(f"{i}. {crypto}")
        
        while True:
            try:
                choice = int(input("\nВыберите криптовалюту: ")) - 1
                if 0 <= choice < len(available_cryptos):
                    crypto_name = available_cryptos[choice]
                    break
            except ValueError:
                pass
        
        # Выбор типа прогноза
        print("\nТип прогноза:")
        print("1. Одна модель")
        print("2. Ансамблевый прогноз")
        
        forecast_type = input("Выберите тип (1/2): ")
        
        # Горизонт прогноза
        days = int(input("Количество дней для прогноза (1-365): "))
        days = max(1, min(365, days))
        
        # Загрузка свежих данных
        print("\n📥 Загрузка свежих данных...")
        crypto_data = self.data_loader.fetch_crypto_data([crypto_name], "1y")
        features_df = self.data_loader.prepare_features(crypto_name)
        
        if forecast_type == "1":
            # Выбор модели
            available_models = [key.split('_', 1)[1] for key in self.model_builder.models.keys() 
                              if key.startswith(crypto_name)]
            print("\nДоступные модели:")
            for i, model in enumerate(available_models, 1):
                print(f"{i}. {model}")
            
            choice = int(input("Выберите модель: ")) - 1
            model_type = available_models[choice]
            
            # Создание прогноза
            dates, values = self.model_builder.create_forecast(
                crypto_name, model_type, features_df, days
            )
            
            # Визуализация
            fig = self.visualizer.plot_forecast(
                features_df['Close'], dates, values, crypto_name, model_type
            )
            
        else:
            # Ансамблевый прогноз
            dates, values, info = self.model_builder.ensemble_predict(
                crypto_name, features_df, days
            )
            
            # Визуализация с доверительными интервалами
            fig = self.visualizer.plot_ensemble_forecast(
                features_df['Close'], dates, values, 
                info['confidence_intervals'], crypto_name
            )
        
        # Статистика прогноза
        self._print_forecast_statistics(dates, values, features_df['Close'].iloc[-1])

    def _print_forecast_statistics(self, dates, values, current_price):
        """Вывод статистики прогноза"""
        print("\n📊 СТАТИСТИКА ПРОГНОЗА")
        print("-" * 40)
        
        # Основные метрики
        final_price = values[-1]
        max_price = np.max(values)
        min_price = np.min(values)
        avg_price = np.mean(values)
        
        change_pct = ((final_price - current_price) / current_price) * 100
        max_gain = ((max_price - current_price) / current_price) * 100
        max_loss = ((min_price - current_price) / current_price) * 100
        
        print(f"Текущая цена: ${current_price:.2f}")
        print(f"Прогноз на {dates[-1].strftime('%d.%m.%Y')}: ${final_price:.2f}")
        print(f"Изменение: {change_pct:+.2f}%")
        print(f"\nДиапазон прогноза:")
        print(f"  Максимум: ${max_price:.2f} ({max_gain:+.2f}%)")
        print(f"  Минимум: ${min_price:.2f} ({max_loss:+.2f}%)")
        print(f"  Среднее: ${avg_price:.2f}")
        
        # Анализ тренда
        mid_point = len(values) // 2
        first_half_avg = np.mean(values[:mid_point])
        second_half_avg = np.mean(values[mid_point:])
        
        if second_half_avg > first_half_avg * 1.02:
            trend = "📈 Восходящий тренд"
        elif second_half_avg < first_half_avg * 0.98:
            trend = "📉 Нисходящий тренд"
        else:
            trend = "➡️ Боковой тренд"
        
        print(f"\nОбщий тренд: {trend}")
        
        # Волатильность прогноза
        volatility = np.std(values) / np.mean(values) * 100
        print(f"Волатильность прогноза: {volatility:.1f}%")
        
        # Рекомендации
        print("\n💡 РЕКОМЕНДАЦИИ:")
        if change_pct > 10:
            print("✅ Сильный восходящий тренд. Рассмотрите возможность покупки.")
        elif change_pct > 5:
            print("✅ Умеренный рост. Подходит для долгосрочных инвестиций.")
        elif change_pct > -5:
            print("⚠️ Нейтральный прогноз. Рекомендуется выжидательная позиция.")
        else:
            print("❌ Негативный прогноз. Рассмотрите возможность продажи или хеджирования.")
        
        if volatility > 20:
            print("⚠️ Высокая волатильность. Используйте стоп-лоссы.")

    def interactive_forecast(self):
        """Интерактивное создание прогнозов"""
        print("\n🔮 ИНТЕРАКТИВНЫЙ ПРОГНОЗ")
        print("-" * 40)

        # Проверка наличия обученных моделей
        if not self.model_builder.models:
            print("❌ Нет обученных моделей. Сначала запустите обучение.")
            return

        available_models = list(self.model_builder.models.keys())
        print("Доступные модели:")
        for i, model_key in enumerate(available_models, 1):
            crypto_name, model_type = model_key.split("_", 1)
            print(f"{i}. {crypto_name} - {model_type}")

        while True:
            try:
                choice = int(input("\nВыберите модель (номер): ")) - 1
                if 0 <= choice < len(available_models):
                    selected_model = available_models[choice]
                    crypto_name, model_type = selected_model.split("_", 1)
                    break
                else:
                    print("❌ Неверный выбор!")
            except ValueError:
                print("❌ Введите число!")

        # Выбор горизонта прогноза
        horizons = list(FORECAST_HORIZONS.keys())
        print("\nГоризонт прогнозирования:")
        for i, horizon in enumerate(horizons, 1):
            print(f"{i}. {horizon}")

        while True:
            try:
                choice = int(input("Выберите горизонт: ")) - 1
                if 0 <= choice < len(horizons):
                    forecast_days = FORECAST_HORIZONS[horizons[choice]]
                    break
                else:
                    print("❌ Неверный выбор!")
            except ValueError:
                print("❌ Введите число!")

        # Создание прогноза
        if crypto_name in self.session_results:
            features_df = self.session_results[crypto_name]["features_df"]
        else:
            # Загрузка свежих данных
            crypto_data = self.data_loader.fetch_crypto_data([crypto_name])
            features_df = self.data_loader.prepare_features(crypto_name)

        try:
            print(f"\n🔮 Создание прогноза {crypto_name} на {forecast_days} дней...")
            forecast_dates, forecast_values = self.model_builder.create_forecast(
                crypto_name, model_type, features_df, forecast_days
            )

            # Визуализация прогноза
            forecast_fig = self.visualizer.plot_forecast(
                features_df["Close"],
                forecast_dates,
                forecast_values,
                crypto_name,
                model_type,
            )

            # Вывод прогноза
            self._print_forecast_summary(forecast_dates, forecast_values, crypto_name)

        except Exception as e:
            print(f"❌ Ошибка создания прогноза: {e}")

    def _print_forecast_summary(self, forecast_dates, forecast_values, crypto_name):
        """Вывод сводки прогноза"""
        current_price = forecast_values[0]
        final_price = forecast_values[-1]
        change_percent = ((final_price - current_price) / current_price) * 100

        print(f"\n📊 Сводка прогноза для {crypto_name}:")
        print(f"   Начальная цена: ${current_price:.2f}")
        print(f"   Финальная цена: ${final_price:.2f}")
        print(f"   Изменение: {change_percent:+.2f}%")
        print(f"   Максимум: ${max(forecast_values):.2f}")
        print(f"   Минимум: ${min(forecast_values):.2f}")

        trend = "📈 Восходящий" if change_percent > 0 else "📉 Нисходящий"
        print(f"   Тренд: {trend}")

    def _view_results(self):
        """Просмотр результатов"""
        if not self.session_results:
            print("❌ Нет результатов для отображения. Запустите анализ.")
            return

        print(f"\n📊 РЕЗУЛЬТАТЫ ТЕКУЩЕЙ СЕССИИ")
        print("-" * 40)

        for crypto_name, results in self.session_results.items():
            print(f"\n🪙 {crypto_name}:")

            for model_key, metrics in results["metrics"].items():
                model_type = model_key.split("_", 1)[1]
                print(
                    f"   {model_type}: RMSE=${metrics['RMSE']:.2f}, "
                    f"R²={metrics['R2']:.3f}, "
                    f"Точность={metrics['Directional_Accuracy']:.1f}%"
                )

        input("\nНажмите Enter для продолжения...")

    def load_previous_session(self) -> bool:
        """Загрузка предыдущей сессии"""
        print("\n🔄 ЗАГРУЗКА ПРЕДЫДУЩЕЙ СЕССИИ")
        print("-" * 40)

        # Поиск файлов сессий
        session_files = list(DIRS["metrics"].glob("session_*.json"))

        if not session_files:
            print("❌ Предыдущие сессии не найдены")
            return False

        # Сортировка по времени создания
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print("Доступные сессии:")
        for i, session_file in enumerate(
            session_files[:10], 1
        ):  # Показываем последние 10
            timestamp = session_file.stem.replace("session_", "")
            formatted_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime(
                "%d.%m.%Y %H:%M"
            )
            print(f"{i}. {formatted_time}")

        while True:
            try:
                choice = input(
                    "\nВыберите сессию для загрузки (номер) или 'n' для отмены: "
                )

                if choice.lower() == "n":
                    return False

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(session_files[:10]):
                    selected_file = session_files[choice_idx]

                    # Загрузка данных сессии
                    with open(selected_file, "r", encoding="utf-8") as f:
                        session_data = json.load(f)

                    print(f"✅ Сессия загружена: {selected_file.name}")
                    self._display_session_info(session_data)
                    return True
                else:
                    print("❌ Неверный выбор!")

            except ValueError:
                print("❌ Введите число или 'n'!")

    def _display_session_info(self, session_data: Dict):
        """Отображение информации о сессии"""
        print(f"\n📋 Информация о сессии:")
        print(f"   Время: {session_data['timestamp']}")
        print(f"   Период данных: {session_data['parameters']['data_period']}")
        print(
            f"   Горизонт прогноза: {session_data['parameters']['forecast_days']} дней"
        )
        print(f"   Эпохи обучения: {session_data['parameters']['epochs']}")

        if 'feature_options' in session_data:
            print(f"\n   Использованные признаки:")
            for key, value in session_data['feature_options'].items():
                if key.startswith('use_') and value:
                    feature_name = key.replace('use_', '').capitalize()
                    print(f"     ✓ {feature_name}")

        print(f"\n   Результаты:")
        for crypto_name, results in session_data["results"].items():
            models_count = len(results["metrics"])
            models_list = results.get("models_used", [])
            print(f"     {crypto_name}: {models_count} модель(ей)")
            if models_list:
                print(f"       Модели: {', '.join(models_list)}")


def main():
    """Главная функция"""
    interface = CryptoPredictorInterface()
    interface.run()


if __name__ == "__main__":
    main()