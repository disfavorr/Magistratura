# Predicting the work of IT software development teams using machine learning models

Магистерская диссертация: прогнозирование сроков выполнения задач 
ИТ-командами разработки ПО с использованием моделей машинного обучения.

## Данные
Полный датасет TAWOS: https://github.com/SOLAR-group/TAWOS
Выгруженный и использованный в данной работе находится в `raw/tawos_clean.csv`


## Запуск
Скрипты запускаются по порядку:
1. `pipeline_1_eda.py` — разведочный анализ
2. `pipeline_2_prepare.py` — очистка
3. `pipeline_3a_manual_features.py` — ручные признаки
4. `pipeline_3b_auto_features.py` — авто признаки
5. `pipeline_4_baselines.py` — baseline
6. `pipeline_5_gridsearch.py` — GridSearch
7. `pipeline_6_optuna.py` — Optuna NSGA-II
8. `pipeline_7_tabnet.py` — TabNet
9. `pipeline_8_analysis.py` — Wilcoxon, bootstrap, SHAP

## Главные результаты
- XGBoost manual MAPE 171% на опытных исполнителях
- XGBoost manual MAPE 50,58% при длительности задачи 30-90 дней
- Классификация задержек F1=0.545, ROC-AUC=0.70
- Гибридная схема: ML при наличии истории, иначе SP-baseline