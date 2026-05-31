"""
Дополнительные SHAP-графики: драйверы целевых переменных
для XGBoost manual (лидер обоих треков).
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap

warnings.filterwarnings('ignore')

ROOT = r'D:\blablabla\Magistr'
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
MODELS_DIR = os.path.join(ROOT, 'Dataset', 'models', 'core')
SHAP_PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'shap_plots')

os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_POSITIVE = '#d7191c'
COLOR_NEGATIVE = '#3a76b5'
COLOR_BAR = '#7fbc41'

RANDOM_STATE = 42


# Загрузка моделей и данных

print('Загрузка моделей и данных')

with open(os.path.join(MODELS_DIR, 'grid_xgb_reg_manual.pkl'), 'rb') as f:
    xgb_reg = pickle.load(f)['model']

with open(os.path.join(MODELS_DIR, 'grid_xgb_clf_manual.pkl'), 'rb') as f:
    xgb_clf = pickle.load(f)['model']

X_test = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'X_test.csv'))
y_reg_test_days = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'y_reg_test_days.csv')).iloc[:, 0]
y_clf_test = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'y_clf_test.csv')).iloc[:, 0]

# Выборка для SHAP (1000 для скорости)
rng = np.random.default_rng(RANDOM_STATE)
sample_n = min(1000, len(X_test))
sample_idx = rng.choice(len(X_test), sample_n, replace=False)
X_sample = X_test.iloc[sample_idx].reset_index(drop=True)

# Вычисление SHAP-значений

print('Вычисление SHAP-значений (регрессия)')
explainer_reg = shap.TreeExplainer(xgb_reg)
shap_values_reg = explainer_reg.shap_values(X_sample)

print('Вычисление SHAP-значений (классификация)')
explainer_clf = shap.TreeExplainer(xgb_clf)
shap_values_clf = explainer_clf.shap_values(X_sample)
if isinstance(shap_values_clf, list) and len(shap_values_clf) == 2:
    shap_values_clf = shap_values_clf[1]


# 32. Bar plot регрессии: топ-15 драйверов

print('\nПостроение 32_shap_bar_reg.png')
mean_abs_reg = np.abs(shap_values_reg).mean(axis=0)
top15_reg = np.argsort(mean_abs_reg)[::-1][:15]
top15_reg_features = X_sample.columns[top15_reg].tolist()
top15_reg_values = mean_abs_reg[top15_reg]

fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(top15_reg_features))
ax.barh(y_pos, top15_reg_values[::-1], color=COLOR_BAR, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(top15_reg_features[::-1])
ax.set_xlabel('Средняя абсолютная важность SHAP')
ax.set_title('Драйверы целевой переменной resolution_days (XGBoost регрессия)')
plt.tight_layout()
plt.savefig(os.path.join(SHAP_PLOTS_DIR, '32_shap_bar_reg.png'))
plt.close()
print('  Сохранён 32_shap_bar_reg.png')


# 33. Bar plot классификации

print('Построение 33_shap_bar_clf.png')
mean_abs_clf = np.abs(shap_values_clf).mean(axis=0)
top15_clf = np.argsort(mean_abs_clf)[::-1][:15]
top15_clf_features = X_sample.columns[top15_clf].tolist()
top15_clf_values = mean_abs_clf[top15_clf]

fig, ax = plt.subplots(figsize=(9, 8))
y_pos = np.arange(len(top15_clf_features))
ax.barh(y_pos, top15_clf_values[::-1], color=COLOR_BAR, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(top15_clf_features[::-1])
ax.set_xlabel('Средняя абсолютная важность SHAP')
ax.set_title('Драйверы целевой переменной target_is_delayed (XGBoost классификация)')
plt.tight_layout()
plt.savefig(os.path.join(SHAP_PLOTS_DIR, '33_shap_bar_clf.png'))
plt.close()
print('  Сохранён 33_shap_bar_clf.png')


# 34. Direcional plot регрессии (раздельно положительные и отрицательные вклады)

print('Построение 34_shap_directional_reg.png')

# Для каждого признака: средний положительный вклад vs средний отрицательный
mean_positive_reg = np.where(shap_values_reg > 0, shap_values_reg, 0).mean(axis=0)
mean_negative_reg = np.where(shap_values_reg < 0, shap_values_reg, 0).mean(axis=0)

# Сортировка по сумме абсолютных значений
combined_reg = mean_positive_reg - mean_negative_reg
top15_dir_idx = np.argsort(combined_reg)[::-1][:15]
features_dir = X_sample.columns[top15_dir_idx].tolist()
pos_dir = mean_positive_reg[top15_dir_idx]
neg_dir = mean_negative_reg[top15_dir_idx]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(features_dir))
ax.barh(y_pos, pos_dir[::-1], color=COLOR_POSITIVE, edgecolor='black',
        label='Удлиняет прогноз (push up)', alpha=0.85)
ax.barh(y_pos, neg_dir[::-1], color=COLOR_NEGATIVE, edgecolor='black',
        label='Укорачивает прогноз (push down)', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(features_dir[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Средний SHAP-вклад в прогноз resolution_days (log-шкала)')
ax.set_title('Драйверы регрессии: направление влияния на длительность')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(SHAP_PLOTS_DIR, '34_shap_directional_reg.png'))
plt.close()
print('  Сохранён 34_shap_directional_reg.png')


# 35. Directional plot классификации

print('Построение 35_shap_directional_clf.png')

mean_positive_clf = np.where(shap_values_clf > 0, shap_values_clf, 0).mean(axis=0)
mean_negative_clf = np.where(shap_values_clf < 0, shap_values_clf, 0).mean(axis=0)

combined_clf = mean_positive_clf - mean_negative_clf
top15_dir_idx = np.argsort(combined_clf)[::-1][:15]
features_dir_clf = X_sample.columns[top15_dir_idx].tolist()
pos_dir_clf = mean_positive_clf[top15_dir_idx]
neg_dir_clf = mean_negative_clf[top15_dir_idx]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(features_dir_clf))
ax.barh(y_pos, pos_dir_clf[::-1], color=COLOR_POSITIVE, edgecolor='black',
        label='Увеличивает риск задержки', alpha=0.85)
ax.barh(y_pos, neg_dir_clf[::-1], color=COLOR_NEGATIVE, edgecolor='black',
        label='Снижает риск задержки', alpha=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(features_dir_clf[::-1])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Средний SHAP-вклад в логит прогноза target_is_delayed')
ax.set_title('Драйверы классификации: направление влияния на риск задержки')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(SHAP_PLOTS_DIR, '35_shap_directional_clf.png'))
plt.close()
print('  Сохранён 35_shap_directional_clf.png')


# 36. Waterfall plot - примеры конкретных прогнозов (один долгий, один короткий)

print('Построение 36_shap_waterfall_examples.png')

y_sample_days = y_reg_test_days.iloc[sample_idx].reset_index(drop=True)

# Самая короткая фактически и самая длинная фактически
idx_short = y_sample_days.idxmin()
idx_long = y_sample_days.idxmax()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Короткая задача
shap_short = shap_values_reg[idx_short]
top_short_idx = np.argsort(np.abs(shap_short))[::-1][:10]
ax = axes[0]
y_pos = np.arange(len(top_short_idx))
values = shap_short[top_short_idx][::-1]
features = X_sample.columns[top_short_idx].tolist()[::-1]
colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in values]
ax.barh(y_pos, values, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('SHAP-вклад (log_days)')
ax.set_title(f'Короткая задача (факт = {y_sample_days.iloc[idx_short]:.1f} дней)\nТоп-10 признаков по вкладу')

# Долгая задача
shap_long = shap_values_reg[idx_long]
top_long_idx = np.argsort(np.abs(shap_long))[::-1][:10]
ax = axes[1]
y_pos = np.arange(len(top_long_idx))
values = shap_long[top_long_idx][::-1]
features = X_sample.columns[top_long_idx].tolist()[::-1]
colors = [COLOR_POSITIVE if v > 0 else COLOR_NEGATIVE for v in values]
ax.barh(y_pos, values, color=colors, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('SHAP-вклад (log_days)')
ax.set_title(f'Долгая задача (факт = {y_sample_days.iloc[idx_long]:.1f} дней)\nТоп-10 признаков по вкладу')

plt.suptitle('Примеры конкретных прогнозов: разложение по признакам',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_PLOTS_DIR, '36_shap_waterfall_examples.png'))
plt.close()
print('  Сохранён 36_shap_waterfall_examples.png')

print('\nГотово. Создано 5 графиков в Dataset/shap_plots/')