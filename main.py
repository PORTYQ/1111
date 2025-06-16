"""
Главный файл с обновленным графическим интерфейсом для системы прогнозирования криптовалют
Версия 2.0 с поддержкой новых моделей и функций
"""

import sys
import os
import warnings
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
from datetime import datetime
import json
import numpy as np

# Подавление предупреждений
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

try:
    from interface import CryptoPredictorInterface
    from config import (
        CRYPTO_SYMBOLS, MACRO_INDICATORS, DATA_PERIODS, 
        FORECAST_HORIZONS, UI_DESCRIPTIONS, PRESET_STRATEGIES,
        MODEL_SPECIFIC_CONFIG
    )
    from utils import Logger, create_project_structure
    from feature_engineering import AdvancedFeatureEngineering
except ImportError as e:
    messagebox.showerror("Ошибка импорта", f"Не удалось импортировать модули:\n{e}")
    sys.exit(1)


class CryptoGUI:
    """Обновленный графический интерфейс для системы прогнозирования"""

    def __init__(self, root):
        self.root = root
        self.root.title("Система прогнозирования курса криптовалют v2.0")
        self.root.geometry("1200x800")
        
        # Иконка приложения (если есть)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # Стили
        self.setup_styles()

        # Переменные
        self.selected_cryptos = []
        self.selected_models = []
        self.selected_macros = []
        self.feature_options = {}
        self.interface = CryptoPredictorInterface()
        self.is_running = False
        self.current_strategy = None

        # Создание интерфейса
        self.create_widgets()
        
        # Загрузка последних настроек
        self.load_last_settings()

    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        style.theme_use("clam")

        # Цвета
        bg_color = "#f0f0f0"
        accent_color = "#2196F3"
        success_color = "#4CAF50"
        warning_color = "#FF9800"
        error_color = "#F44336"

        self.root.configure(bg=bg_color)

        # Стиль для кнопок
        style.configure(
            "Accent.TButton",
            background=accent_color,
            foreground="white",
            borderwidth=0,
            focuscolor="none",
            font=("Arial", 10, "bold"),
        )
        style.map("Accent.TButton", background=[("active", "#1976D2")])
        
        style.configure(
            "Success.TButton",
            background=success_color,
            foreground="white",
            borderwidth=0,
            font=("Arial", 10, "bold"),
        )
        
        style.configure(
            "Warning.TButton",
            background=warning_color,
            foreground="white",
            borderwidth=0,
            font=("Arial", 10, "bold"),
        )

    def create_widgets(self):
        """Создание виджетов"""
        # Создание notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка основного анализа
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="🏠 Главная")
        self.create_main_tab()
        
        # Вкладка настроек моделей
        self.models_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.models_tab, text="🤖 Модели")
        self.create_models_tab()
        
        # Вкладка инженерии признаков
        self.features_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.features_tab, text="🛠️ Признаки")
        self.create_features_tab()
        
        # Вкладка результатов
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📊 Результаты")
        self.create_results_tab()
        
        # Вкладка прогнозов
        self.forecast_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.forecast_tab, text="🔮 Прогнозы")
        self.create_forecast_tab()
        
        # Статус бар
        self.create_status_bar()

    def create_main_tab(self):
        """Создание главной вкладки"""
        # Заголовок
        header = tk.Label(
            self.main_tab,
            text="🚀 Система прогнозирования криптовалют v2.0",
            font=("Arial", 18, "bold"),
            bg="#2196F3",
            fg="white",
            pady=15,
        )
        header.pack(fill=tk.X)
        
        # Основной контейнер
        main_frame = ttk.Frame(self.main_tab, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Три колонки
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Левая колонка - выбор криптовалют
        self.create_crypto_section(left_frame)
        
        # Средняя колонка - макропоказатели и параметры
        self.create_macro_section(middle_frame)
        self.create_params_section(middle_frame)
        
        # Правая колонка - стратегии и управление
        self.create_strategy_section(right_frame)
        self.create_control_buttons(right_frame)
        
        # Консоль внизу
        self.create_console(main_frame)

    def create_crypto_section(self, parent):
        """Секция выбора криптовалют"""
        frame = ttk.LabelFrame(parent, text="📈 Криптовалюты", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Поиск
        search_frame = ttk.Frame(frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(search_frame, text="Поиск:").pack(side=tk.LEFT)
        self.crypto_search = ttk.Entry(search_frame)
        self.crypto_search.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.crypto_search.bind("<KeyRelease>", self.filter_cryptos)

        # Список с чекбоксами в scrollable frame
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.crypto_vars = {}
        self.crypto_checkboxes = {}
        for crypto in CRYPTO_SYMBOLS.keys():
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(scrollable_frame, text=crypto, variable=var)
            cb.pack(anchor=tk.W, pady=2)
            self.crypto_vars[crypto] = var
            self.crypto_checkboxes[crypto] = cb
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Кнопки выбора
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Выбрать все",
            command=lambda: self.select_all(self.crypto_vars),
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Снять все",
            command=lambda: self.deselect_all(self.crypto_vars),
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Топ-5",
            command=self.select_top_cryptos,
        ).pack(side=tk.LEFT, padx=2)

    def create_models_tab(self):
        """Создание вкладки настроек моделей"""
        main_frame = ttk.Frame(self.models_tab, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Выбор моделей
        models_frame = ttk.LabelFrame(main_frame, text="🤖 Выбор моделей", padding="10")
        models_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.model_vars = {}
        
        # Создаем frame для каждой модели с описанием
        for model, description in UI_DESCRIPTIONS["models"].items():
            model_frame = ttk.Frame(models_frame)
            model_frame.pack(fill=tk.X, pady=5)
            
            var = tk.BooleanVar(value=model in ["LSTM", "GRU"])
            self.model_vars[model] = var
            
            cb = ttk.Checkbutton(
                model_frame,
                text=model,
                variable=var,
                command=lambda m=model: self.update_model_info(m)
            )
            cb.pack(side=tk.LEFT)
            
            ttk.Label(
                model_frame,
                text=f" - {description}",
                font=("Arial", 9),
                foreground="gray"
            ).pack(side=tk.LEFT, padx=(10, 0))
            
            # Кнопка настроек для модели
            ttk.Button(
                model_frame,
                text="⚙️",
                width=3,
                command=lambda m=model: self.configure_model(m)
            ).pack(side=tk.RIGHT)
        
        # Информация о выбранных моделях
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Информация", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame,
            height=10,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.model_info_text.pack(fill=tk.BOTH, expand=True)
        
        self.update_model_info()

    def create_features_tab(self):
        """Создание вкладки инженерии признаков"""
        main_frame = ttk.Frame(self.features_tab, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Основные признаки
        basic_frame = ttk.LabelFrame(main_frame, text="📊 Базовые признаки", padding="10")
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(basic_frame, text="Технические индикаторы включены по умолчанию").pack()
        
        # Продвинутые признаки
        advanced_frame = ttk.LabelFrame(main_frame, text="🚀 Продвинутые признаки", padding="10")
        advanced_frame.pack(fill=tk.BOTH, expand=True)
        
        self.feature_vars = {}
        
        for feature, description in UI_DESCRIPTIONS["features"].items():
            feature_frame = ttk.Frame(advanced_frame)
            feature_frame.pack(fill=tk.X, pady=5)
            
            var = tk.BooleanVar(value=True)
            self.feature_vars[f"use_{feature}"] = var
            
            cb = ttk.Checkbutton(
                feature_frame,
                text=description,
                variable=var
            )
            cb.pack(side=tk.LEFT)
        
        # Настройки обработки признаков
        processing_frame = ttk.LabelFrame(main_frame, text="⚙️ Обработка признаков", padding="10")
        processing_frame.pack(fill=tk.X, pady=(10, 0))
        
        # PCA
        pca_frame = ttk.Frame(processing_frame)
        pca_frame.pack(fill=tk.X, pady=5)
        
        self.use_pca = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pca_frame,
            text="Использовать PCA",
            variable=self.use_pca
        ).pack(side=tk.LEFT)
        
        ttk.Label(pca_frame, text="Дисперсия:").pack(side=tk.LEFT, padx=(20, 5))
        self.pca_variance = tk.DoubleVar(value=0.95)
        pca_spin = ttk.Spinbox(
            pca_frame,
            from_=0.8,
            to=0.99,
            increment=0.01,
            textvariable=self.pca_variance,
            width=10
        )
        pca_spin.pack(side=tk.LEFT)
        
        # Отбор признаков
        selection_frame = ttk.Frame(processing_frame)
        selection_frame.pack(fill=tk.X, pady=5)
        
        self.use_selection = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            selection_frame,
            text="Отбор признаков",
            variable=self.use_selection
        ).pack(side=tk.LEFT)
        
        ttk.Label(selection_frame, text="Количество:").pack(side=tk.LEFT, padx=(20, 5))
        self.n_features = tk.IntVar(value=100)
        selection_spin = ttk.Spinbox(
            selection_frame,
            from_=50,
            to=200,
            increment=10,
            textvariable=self.n_features,
            width=10
        )
        selection_spin.pack(side=tk.LEFT)

    def create_results_tab(self):
        """Создание вкладки результатов"""
        main_frame = ttk.Frame(self.results_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель управления
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            control_frame,
            text="🔄 Обновить",
            command=self.refresh_results
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="📊 Детальный анализ",
            command=self.show_detailed_analysis
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="💾 Экспорт",
            command=self.export_results
        ).pack(side=tk.LEFT, padx=5)
        
        # Таблица результатов
        columns = ("Криптовалюта", "Модель", "RMSE", "MAE", "R²", "Точность направления")
        self.results_tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Сводка внизу
        summary_frame = ttk.LabelFrame(main_frame, text="📋 Сводка", padding="10")
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.results_summary = tk.Text(summary_frame, height=5, wrap=tk.WORD)
        self.results_summary.pack(fill=tk.BOTH, expand=True)

    def create_forecast_tab(self):
        """Создание вкладки прогнозов"""
        main_frame = ttk.Frame(self.forecast_tab, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Выбор для прогноза
        selection_frame = ttk.LabelFrame(main_frame, text="🎯 Параметры прогноза", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Криптовалюта
        ttk.Label(selection_frame, text="Криптовалюта:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.forecast_crypto = ttk.Combobox(selection_frame, state="readonly", width=20)
        self.forecast_crypto.grid(row=0, column=1, pady=5, padx=(10, 0))
        
        # Тип прогноза
        ttk.Label(selection_frame, text="Тип прогноза:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.forecast_type = ttk.Combobox(
            selection_frame,
            values=["Одна модель", "Ансамблевый"],
            state="readonly",
            width=20
        )
        self.forecast_type.set("Ансамблевый")
        self.forecast_type.grid(row=1, column=1, pady=5, padx=(10, 0))
        
        # Горизонт
        ttk.Label(selection_frame, text="Горизонт (дни):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.forecast_days = tk.IntVar(value=30)
        days_spin = ttk.Spinbox(
            selection_frame,
            from_=1,
            to=365,
            textvariable=self.forecast_days,
            width=20
        )
        days_spin.grid(row=2, column=1, pady=5, padx=(10, 0))
        
        # Кнопка создания прогноза
        ttk.Button(
            selection_frame,
            text="🔮 Создать прогноз",
            command=self.create_forecast,
            style="Accent.TButton"
        ).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Результаты прогноза
        results_frame = ttk.LabelFrame(main_frame, text="📊 Результаты прогноза", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.forecast_text = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.forecast_text.pack(fill=tk.BOTH, expand=True)

    def create_strategy_section(self, parent):
        """Секция выбора стратегий"""
        frame = ttk.LabelFrame(parent, text="🎯 Стратегии", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        
        strategies = list(PRESET_STRATEGIES.keys())
        for i, strategy in enumerate(strategies):
            btn = ttk.Button(
                frame,
                text=strategy.capitalize(),
                command=lambda s=strategy: self.apply_strategy(s),
                style="Warning.TButton" if strategy == "conservative" else "Accent.TButton"
            )
            btn.pack(fill=tk.X, pady=2)
        
        # Кастомная стратегия
        ttk.Button(
            frame,
            text="⚙️ Настроить",
            command=self.custom_strategy
        ).pack(fill=tk.X, pady=2)

    def create_macro_section(self, parent):
        """Секция макропоказателей"""
        frame = ttk.LabelFrame(parent, text="📊 Макропоказатели", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.use_macro = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Использовать макропоказатели",
            variable=self.use_macro,
            command=self.toggle_macro,
        ).pack(anchor=tk.W, pady=(0, 5))

        # Список макропоказателей
        self.macro_frame = ttk.Frame(frame)
        self.macro_frame.pack(fill=tk.BOTH, expand=True)

        self.macro_vars = {}
        descriptions = {
            "DXY": "Индекс доллара США",
            "Gold": "Золото",
            "VIX": "Индекс волатильности",
            "TNX": "10-летние облигации США",
            "Oil": "Нефть WTI",
            "SP500": "S&P 500",
        }

        for macro, desc in descriptions.items():
            if macro in MACRO_INDICATORS:
                var = tk.BooleanVar(value=True)
                cb = ttk.Checkbutton(
                    self.macro_frame, text=f"{macro} - {desc}", variable=var
                )
                cb.pack(anchor=tk.W, pady=2)
                self.macro_vars[macro] = var
    def create_params_section(self, parent):
        """Секция параметров"""
        frame = ttk.LabelFrame(parent, text="⚙️ Параметры", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))

        # Период данных
        ttk.Label(frame, text="Период данных:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.period_var = tk.StringVar(value="2y")
        period_combo = ttk.Combobox(
            frame,
            textvariable=self.period_var,
            values=list(DATA_PERIODS.values()),
            state="readonly",
            width=15,
        )
        period_combo.grid(row=0, column=1, pady=5, padx=(10, 0))

        # Горизонт прогноза
        ttk.Label(frame, text="Прогноз (дни):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.forecast_var = tk.IntVar(value=30)
        forecast_spin = ttk.Spinbox(
            frame, from_=7, to=90, textvariable=self.forecast_var, width=15
        )
        forecast_spin.grid(row=1, column=1, pady=5, padx=(10, 0))

        # Эпохи обучения
        ttk.Label(frame, text="Эпохи обучения:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.epochs_var = tk.IntVar(value=100)
        epochs_spin = ttk.Spinbox(
            frame, from_=10, to=300, textvariable=self.epochs_var, width=15
        )
        epochs_spin.grid(row=2, column=1, pady=5, padx=(10, 0))

    def create_control_buttons(self, parent):
        """Кнопки управления"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.start_btn = ttk.Button(
            frame,
            text="🚀 Начать анализ",
            command=self.start_analysis,
            style="Accent.TButton",
        )
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(
            frame, text="⏹ Остановить", command=self.stop_analysis, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=2)

        ttk.Button(
            frame, 
            text="📂 Загрузить сессию", 
            command=self.load_session
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            frame, 
            text="💾 Сохранить настройки", 
            command=self.save_settings
        ).pack(fill=tk.X, pady=2)

    def create_console(self, parent):
        """Консоль вывода"""
        console_frame = ttk.LabelFrame(parent, text="📋 Консоль", padding="5")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Панель инструментов консоли
        toolbar = ttk.Frame(console_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            toolbar,
            text="🗑️ Очистить",
            command=lambda: self.console.delete(1.0, tk.END)
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar,
            text="💾 Сохранить лог",
            command=self.save_console_log
        ).pack(side=tk.LEFT, padx=2)

        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=10,
            wrap=tk.WORD,
            bg="black",
            fg="#00FF00",
            font=("Consolas", 9),
        )
        self.console.pack(fill=tk.BOTH, expand=True)

        # Перенаправление вывода
        sys.stdout = ConsoleRedirector(self.console)

    def create_status_bar(self):
        """Создание статус бара"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Статус
        self.status_label = ttk.Label(
            self.status_bar,
            text="Готов к работе",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Прогресс
        self.progress = ttk.Progressbar(
            self.status_bar,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side=tk.RIGHT, padx=5)

    def toggle_macro(self):
        """Переключение макропоказателей"""
        state = "normal" if self.use_macro.get() else "disabled"
        for widget in self.macro_frame.winfo_children():
            widget.configure(state=state)

    def select_all(self, vars_dict):
        """Выбрать все"""
        for var in vars_dict.values():
            var.set(True)

    def deselect_all(self, vars_dict):
        """Снять все"""
        for var in vars_dict.values():
            var.set(False)

    def filter_cryptos(self, event):
        """Фильтрация криптовалют по поиску"""
        search_text = self.crypto_search.get().lower()
        
        for crypto, checkbox in self.crypto_checkboxes.items():
            if search_text in crypto.lower():
                checkbox.pack(anchor=tk.W, pady=2)
            else:
                checkbox.pack_forget()

    def select_top_cryptos(self):
        """Выбор топ-5 криптовалют"""
        top_cryptos = ["Bitcoin", "Ethereum", "Binance Coin", "Solana", "Cardano"]
        
        self.deselect_all(self.crypto_vars)
        for crypto in top_cryptos:
            if crypto in self.crypto_vars:
                self.crypto_vars[crypto].set(True)

    def apply_strategy(self, strategy_name):
        """Применение предустановленной стратегии"""
        strategy = PRESET_STRATEGIES[strategy_name]
        self.current_strategy = strategy_name
        
        # Обновление моделей
        self.deselect_all(self.model_vars)
        for model in strategy["models"]:
            if model in self.model_vars:
                self.model_vars[model].set(True)
        
        # Обновление эпох
        self.epochs_var.set(strategy["epochs"])
        
        # Обновление горизонта прогноза
        self.forecast_var.set(strategy["forecast_horizon"])
        
        # Обновление признаков
        if strategy["features"] == "basic":
            for var in self.feature_vars.values():
                var.set(False)
        elif strategy["features"] == "all":
            for var in self.feature_vars.values():
                var.set(True)
        
        self.update_status(f"Применена стратегия: {strategy_name}")
        messagebox.showinfo("Стратегия", f"Применена стратегия '{strategy_name}'")

    def custom_strategy(self):
        """Настройка кастомной стратегии"""
        # Переключение на вкладку моделей
        self.notebook.select(1)
        messagebox.showinfo("Кастомная стратегия", 
                          "Настройте параметры на вкладках 'Модели' и 'Признаки'")

    def update_model_info(self, model_name=None):
        """Обновление информации о моделях"""
        self.model_info_text.delete(1.0, tk.END)
        
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        
        if not selected_models:
            self.model_info_text.insert(tk.END, "Выберите модели для отображения информации")
            return
        
        info = f"Выбрано моделей: {len(selected_models)}\n\n"
        
        for model in selected_models:
            info += f"{'='*50}\n"
            info += f"Модель: {model}\n"
            info += f"Описание: {UI_DESCRIPTIONS['models'].get(model, 'Нет описания')}\n"
            
            if model in MODEL_SPECIFIC_CONFIG:
                config = MODEL_SPECIFIC_CONFIG[model]
                info += f"Параметры:\n"
                for key, value in config.items():
                    info += f"  - {key}: {value}\n"
            
            info += "\n"
        
        self.model_info_text.insert(tk.END, info)

    def configure_model(self, model_name):
        """Настройка параметров конкретной модели"""
        config_window = tk.Toplevel(self.root)
        config_window.title(f"Настройка {model_name}")
        config_window.geometry("400x300")
        
        # Загрузка текущих параметров
        current_config = MODEL_SPECIFIC_CONFIG.get(model_name, {})
        
        # Создание полей для каждого параметра
        row = 0
        param_vars = {}
        
        for param, value in current_config.items():
            ttk.Label(config_window, text=f"{param}:").grid(
                row=row, column=0, sticky=tk.W, padx=10, pady=5
            )
            
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(config_window, variable=var).grid(
                    row=row, column=1, padx=10, pady=5
                )
            elif isinstance(value, int):
                var = tk.IntVar(value=value)
                ttk.Spinbox(
                    config_window, 
                    from_=1, 
                    to=1000, 
                    textvariable=var,
                    width=20
                ).grid(row=row, column=1, padx=10, pady=5)
            elif isinstance(value, float):
                var = tk.DoubleVar(value=value)
                ttk.Spinbox(
                    config_window,
                    from_=0.0,
                    to=1.0,
                    increment=0.01,
                    textvariable=var,
                    width=20
                ).grid(row=row, column=1, padx=10, pady=5)
            else:
                var = tk.StringVar(value=str(value))
                ttk.Entry(config_window, textvariable=var, width=20).grid(
                    row=row, column=1, padx=10, pady=5
                )
            
            param_vars[param] = var
            row += 1
        
        # Кнопки
        btn_frame = ttk.Frame(config_window)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=20)
        
        def save_config():
            # Сохранение новых параметров
            for param, var in param_vars.items():
                MODEL_SPECIFIC_CONFIG[model_name][param] = var.get()
            messagebox.showinfo("Успех", f"Параметры {model_name} сохранены")
            config_window.destroy()
        
        ttk.Button(btn_frame, text="Сохранить", command=save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=config_window.destroy).pack(side=tk.LEFT, padx=5)

    def start_analysis(self):
        """Запуск анализа"""
        # Сбор выбранных параметров
        self.selected_cryptos = [
            name for name, var in self.crypto_vars.items() if var.get()
        ]
        self.selected_models = [
            name for name, var in self.model_vars.items() if var.get()
        ]
        self.selected_macros = [
            name for name, var in self.macro_vars.items() if var.get()
        ]

        # Проверки
        if not self.selected_cryptos:
            messagebox.showwarning("Внимание", "Выберите хотя бы одну криптовалюту!")
            return

        if not self.selected_models:
            messagebox.showwarning("Внимание", "Выберите хотя бы одну модель!")
            return

        # Блокировка интерфейса
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        self.update_status("Выполняется анализ...")

        # Очистка консоли
        self.console.delete(1.0, tk.END)

        # Запуск в отдельном потоке
        thread = threading.Thread(target=self.run_analysis_thread)
        thread.daemon = True
        thread.start()

    def run_analysis_thread(self):
        """Поток для анализа"""
        try:
            print("🚀 Запуск анализа...")
            print(f"📈 Криптовалюты: {', '.join(self.selected_cryptos)}")
            print(f"🤖 Модели: {', '.join(self.selected_models)}")
            print(
                f"📊 Макропоказатели: {', '.join(self.selected_macros) if self.use_macro.get() else 'Отключены'}"
            )
            print("=" * 60)

            parameters = {
                "data_period": self.period_var.get(),
                "forecast_days": self.forecast_var.get(),
                "epochs": self.epochs_var.get(),
            }

            # Сбор опций для признаков
            feature_options = {}
            for key, var in self.feature_vars.items():
                feature_options[key] = var.get()
            
            if self.use_pca.get():
                feature_options["pca_components"] = self.pca_variance.get()
            else:
                feature_options["pca_components"] = None
            
            if self.use_selection.get():
                feature_options["n_features"] = self.n_features.get()
            else:
                feature_options["n_features"] = None

            self.interface.run_analysis(
                self.selected_cryptos,
                self.selected_models,
                self.use_macro.get(),
                self.selected_macros,
                parameters,
                feature_options
            )

            print("\n✅ Анализ завершен успешно!")
            self.update_status("Анализ завершен")
            
            # Обновление результатов
            self.root.after(0, self.refresh_results)
            
            # Обновление списка для прогнозов
            self.root.after(0, self.update_forecast_cryptos)
            
            messagebox.showinfo("Успех", "Анализ завершен успешно!")

        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            self.update_status(f"Ошибка: {str(e)[:50]}...")
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{e}")

        finally:
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.progress.stop()

    def stop_analysis(self):
        """Остановка анализа"""
        if messagebox.askyesno("Подтверждение", "Остановить анализ?"):
            self.is_running = False
            print("\n⏹ Анализ остановлен пользователем")
            self.update_status("Анализ остановлен")

    def refresh_results(self):
        """Обновление таблицы результатов"""
        # Очистка таблицы
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Заполнение новыми данными
        if self.interface.session_results:
            for crypto_name, results in self.interface.session_results.items():
                for model_key, metrics in results.get("metrics", {}).items():
                    model_type = model_key.split("_", 1)[1]
                    
                    self.results_tree.insert("", "end", values=(
                        crypto_name,
                        model_type,
                        f"{metrics['RMSE']:.2f}",
                        f"{metrics['MAE']:.2f}",
                        f"{metrics['R2']:.4f}",
                        f"{metrics['Directional_Accuracy']:.1f}%"
                    ))
            
            # Обновление сводки
            self.update_results_summary()

    def update_results_summary(self):
        """Обновление сводки результатов"""
        self.results_summary.delete(1.0, tk.END)
        
        if not self.interface.session_results:
            return
        
        # Подсчет статистики
        total_models = sum(
            len(results["metrics"]) for results in self.interface.session_results.values()
        )
        
        # Поиск лучшей модели
        best_model = None
        best_rmse = float('inf')
        
        for crypto_name, results in self.interface.session_results.items():
            for model_key, metrics in results["metrics"].items():
                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    best_model = (crypto_name, model_key.split("_", 1)[1])
        
        # Формирование текста сводки
        summary = f"Всего обучено моделей: {total_models}\n"
        summary += f"Проанализировано криптовалют: {len(self.interface.session_results)}\n"
        
        if best_model:
            summary += f"\nЛучшая модель: {best_model[1]} для {best_model[0]}\n"
            summary += f"RMSE: ${best_rmse:.2f}\n"
        
        if self.current_strategy:
            summary += f"\nИспользована стратегия: {self.current_strategy}"
        
        self.results_summary.insert(tk.END, summary)

    def show_detailed_analysis(self):
        """Показать детальный анализ"""
        selected = self.results_tree.selection()
        
        if not selected:
            messagebox.showwarning("Внимание", "Выберите модель для анализа")
            return
        
        # Получение данных выбранной строки
        values = self.results_tree.item(selected[0])['values']
        crypto_name = values[0]
        model_type = values[1]
        
        # Создание окна детального анализа
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Детальный анализ: {crypto_name} - {model_type}")
        detail_window.geometry("800x600")
        
        # Текстовое поле для анализа
        text = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Формирование детального отчета
        model_key = f"{crypto_name}_{model_type}"
        
        if crypto_name in self.interface.session_results:
            results = self.interface.session_results[crypto_name]
            metrics = results["metrics"].get(model_key, {})
            
            report = f"ДЕТАЛЬНЫЙ АНАЛИЗ\n{'='*50}\n\n"
            report += f"Криптовалюта: {crypto_name}\n"
            report += f"Модель: {model_type}\n\n"
            
            report += f"МЕТРИКИ КАЧЕСТВА:\n{'-'*30}\n"
            for metric, value in metrics.items():
                report += f"{metric}: {value:.4f}\n"
            
            # Добавление дополнительной информации
            if "predictions" in results and model_key in results["predictions"]:
                pred_data = results["predictions"][model_key]
                report += f"\n\nСТАТИСТИКА ПРЕДСКАЗАНИЙ:\n{'-'*30}\n"
                report += f"Размер обучающей выборки: {len(pred_data.get('train_pred', []))}\n"
                report += f"Размер тестовой выборки: {len(pred_data.get('test_pred', []))}\n"
            
            text.insert(tk.END, report)
        
        # Кнопка закрытия
        ttk.Button(
            detail_window,
            text="Закрыть",
            command=detail_window.destroy
        ).pack(pady=10)

    def update_forecast_cryptos(self):
        """Обновление списка криптовалют для прогноза"""
        if self.interface.model_builder.models:
            cryptos = list(set(
                key.split('_')[0] for key in self.interface.model_builder.models.keys()
            ))
            self.forecast_crypto['values'] = cryptos
            if cryptos:
                self.forecast_crypto.set(cryptos[0])

    def create_forecast(self):
        """Создание прогноза"""
        crypto = self.forecast_crypto.get()
        if not crypto:
            messagebox.showwarning("Внимание", "Выберите криптовалюту")
            return
        
        self.forecast_text.delete(1.0, tk.END)
        self.forecast_text.insert(tk.END, "Создание прогноза...\n")
        
        # Запуск в отдельном потоке
        thread = threading.Thread(
            target=self.create_forecast_thread,
            args=(crypto,)
        )
        thread.daemon = True
        thread.start()

    def create_forecast_thread(self, crypto_name):
        """Поток для создания прогноза"""
        try:
            forecast_type = self.forecast_type.get()
            days = self.forecast_days.get()
            
            # Загрузка данных
            self.update_forecast_status("Загрузка данных...")
            crypto_data = self.interface.data_loader.fetch_crypto_data([crypto_name], "1y")
            features_df = self.interface.data_loader.prepare_features(crypto_name)
            
            if forecast_type == "Ансамблевый":
                self.update_forecast_status("Создание ансамблевого прогноза...")
                dates, values, info = self.interface.model_builder.ensemble_predict(
                    crypto_name, features_df, days
                )
                
                # Формирование отчета
                report = self.format_ensemble_forecast_report(
                    crypto_name, dates, values, info, features_df['Close'].iloc[-1]
                )
            else:
                # Выбор лучшей модели
                available_models = [
                    key.split('_', 1)[1] for key in self.interface.model_builder.models.keys()
                    if key.startswith(crypto_name)
                ]
                
                if not available_models:
                    raise ValueError("Нет доступных моделей")
                
                # Используем модель с лучшим RMSE
                best_model = min(
                    available_models,
                    key=lambda m: self.interface.session_results[crypto_name]["metrics"][f"{crypto_name}_{m}"]["RMSE"]
                )
                
                self.update_forecast_status(f"Создание прогноза с помощью {best_model}...")
                dates, values = self.interface.model_builder.create_forecast(
                    crypto_name, best_model, features_df, days
                )
                
                report = self.format_single_forecast_report(
                    crypto_name, best_model, dates, values, features_df['Close'].iloc[-1]
                )
            
            self.update_forecast_status(report)
            
        except Exception as e:
            self.update_forecast_status(f"Ошибка создания прогноза: {e}")

    def format_ensemble_forecast_report(self, crypto_name, dates, values, info, current_price):
        """Форматирование отчета ансамблевого прогноза"""
        report = f"АНСАМБЛЕВЫЙ ПРОГНОЗ ДЛЯ {crypto_name}\n{'='*60}\n\n"
        
        # Основная информация
        final_price = values[-1]
        change_pct = ((final_price - current_price) / current_price) * 100
        
        report += f"Текущая цена: ${current_price:.2f}\n"
        report += f"Прогноз на {dates[-1].strftime('%d.%m.%Y')}: ${final_price:.2f}\n"
        report += f"Изменение: {change_pct:+.2f}%\n\n"
        
        # Использованные модели и веса
        report += "МОДЕЛИ И ВЕСА:\n"
        for model, weight in info['weights'].items():
            report += f"  - {model}: {weight:.3f}\n"
        
        # Доверительные интервалы
        report += f"\n95% ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ:\n"
        report += f"  Нижняя граница: ${info['confidence_intervals']['lower_95'][-1]:.2f}\n"
        report += f"  Верхняя граница: ${info['confidence_intervals']['upper_95'][-1]:.2f}\n"
        
        # Статистика
        report += f"\nСТАТИСТИКА ПРОГНОЗА:\n"
        report += f"  Максимум: ${np.max(values):.2f}\n"
        report += f"  Минимум: ${np.min(values):.2f}\n"
        report += f"  Среднее: ${np.mean(values):.2f}\n"
        report += f"  Волатильность: {np.std(values)/np.mean(values)*100:.1f}%\n"
        
        # Рекомендации
        report += "\nРЕКОМЕНДАЦИИ:\n"
        if change_pct > 10:
            report += "✅ Сильный восходящий тренд. Рассмотрите возможность покупки.\n"
        elif change_pct > 5:
            report += "✅ Умеренный рост. Подходит для долгосрочных инвестиций.\n"
        elif change_pct > -5:
            report += "⚠️ Нейтральный прогноз. Рекомендуется выжидательная позиция.\n"
        else:
            report += "❌ Негативный прогноз. Рассмотрите возможность продажи.\n"
        
        return report

    def format_single_forecast_report(self, crypto_name, model_type, dates, values, current_price):
        """Форматирование отчета одиночного прогноза"""
        report = f"ПРОГНОЗ {model_type} ДЛЯ {crypto_name}\n{'='*60}\n\n"
        
        final_price = values[-1]
        change_pct = ((final_price - current_price) / current_price) * 100
        
        report += f"Модель: {model_type}\n"
        report += f"Текущая цена: ${current_price:.2f}\n"
        report += f"Прогноз на {dates[-1].strftime('%d.%m.%Y')}: ${final_price:.2f}\n"
        report += f"Изменение: {change_pct:+.2f}%\n\n"
        
        report += f"ДИАПАЗОН ПРОГНОЗА:\n"
        report += f"  Максимум: ${np.max(values):.2f}\n"
        report += f"  Минимум: ${np.min(values):.2f}\n"
        report += f"  Среднее: ${np.mean(values):.2f}\n"
        
        return report

    def update_forecast_status(self, message):
        """Обновление статуса прогноза"""
        self.root.after(0, lambda: self.forecast_text.insert(tk.END, f"\n{message}"))

    def export_results(self):
        """Экспорт результатов"""
        if not self.interface.session_results:
            messagebox.showinfo("Информация", "Нет результатов для экспорта")
            return

        try:
            from utils import FileManager
            
            # Диалог сохранения файла
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel файлы", "*.xlsx"),
                    ("CSV файлы", "*.csv"),
                    ("JSON файлы", "*.json")
                ]
            )
            
            if file_path:
                if file_path.endswith('.xlsx'):
                    FileManager.export_results_to_excel(
                        self.interface.session_results,
                        file_path
                    )
                messagebox.showinfo("Успех", f"Результаты экспортированы в:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта:\n{e}")

    def save_console_log(self):
        """Сохранение лога консоли"""
        log_text = self.console.get(1.0, tk.END)
        
        if not log_text.strip():
            messagebox.showinfo("Информация", "Консоль пуста")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_text)
                messagebox.showinfo("Успех", f"Лог сохранен в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения лога:\n{e}")

    def load_session(self):
        """Загрузка предыдущей сессии"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Обработка загруженных данных
                self.process_loaded_session(session_data)
                messagebox.showinfo("Успех", "Сессия загружена успешно")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки сессии:\n{e}")

    def process_loaded_session(self, session_data):
        """Обработка загруженной сессии"""
        # Обновление параметров
        if 'parameters' in session_data:
            params = session_data['parameters']
            if 'data_period' in params:
                self.period_var.set(params['data_period'])
            if 'forecast_days' in params:
                self.forecast_var.set(params['forecast_days'])
            if 'epochs' in params:
                self.epochs_var.set(params['epochs'])
        
        # Отображение результатов
        if 'results' in session_data:
            print(f"\nЗагружена сессия от {session_data.get('timestamp', 'неизвестно')}")
            print(f"Результаты для {len(session_data['results'])} криптовалют")

    def save_settings(self):
        """Сохранение текущих настроек"""
        settings = {
            'cryptos': [name for name, var in self.crypto_vars.items() if var.get()],
            'models': [name for name, var in self.model_vars.items() if var.get()],
            'macros': [name for name, var in self.macro_vars.items() if var.get()],
            'use_macro': self.use_macro.get(),
            'period': self.period_var.get(),
            'forecast_days': self.forecast_var.get(),
            'epochs': self.epochs_var.get(),
            'features': {key: var.get() for key, var in self.feature_vars.items()},
            'use_pca': self.use_pca.get(),
            'pca_variance': self.pca_variance.get(),
            'use_selection': self.use_selection.get(),
            'n_features': self.n_features.get(),
            'current_strategy': self.current_strategy
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Успех", f"Настройки сохранены в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения настроек:\n{e}")

    def load_last_settings(self):
        """Загрузка последних настроек при запуске"""
        settings_file = Path("last_settings.json")
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Применение настроек
                if 'cryptos' in settings:
                    for crypto in settings['cryptos']:
                        if crypto in self.crypto_vars:
                            self.crypto_vars[crypto].set(True)
                
                if 'models' in settings:
                    for model in settings['models']:
                        if model in self.model_vars:
                            self.model_vars[model].set(True)
                
                if 'period' in settings:
                    self.period_var.set(settings['period'])
                
                if 'epochs' in settings:
                    self.epochs_var.set(settings['epochs'])
                
                print("Загружены последние настройки")
            except Exception as e:
                print(f"Не удалось загрузить последние настройки: {e}")

    def update_status(self, message):
        """Обновление статус бара"""
        self.status_label.config(text=message)
        self.root.update()

    def on_closing(self):
        """Действия при закрытии приложения"""
        # Сохранение текущих настроек
        try:
            settings = {
                'cryptos': [name for name, var in self.crypto_vars.items() if var.get()],
                'models': [name for name, var in self.model_vars.items() if var.get()],
                'period': self.period_var.get(),
                'epochs': self.epochs_var.get(),
            }
            
            with open("last_settings.json", 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except:
            pass
        
        self.root.destroy()


class ConsoleRedirector:
    """Перенаправление вывода в текстовое поле"""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.update()

    def flush(self):
        pass


def main():
    """Главная функция"""
    # Проверка и создание структуры проекта
    create_project_structure()

    # Создание GUI
    root = tk.Tk()
    app = CryptoGUI(root)
    
    # Обработчик закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()            