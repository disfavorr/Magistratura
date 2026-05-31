import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Пути
ROOT = r'D:\blablabla\Magistr'
MANUAL_PATH = os.path.join(ROOT, 'Dataset', 'features_manual', 'tawos_features_manual.csv')
PREPARED_PATH = os.path.join(ROOT, 'Dataset', 'prepared', 'tawos_prepared.csv')
FEATURES_AUTO_DIR = os.path.join(ROOT, 'Dataset', 'features_auto')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'eda_plots')

os.makedirs(FEATURES_AUTO_DIR, exist_ok=True)

# Возникали проблемы с красивыми подписями и визуализацией, далее это будет прописываться явно
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'

MAIN_COLOR = '#3a76b5'
ACCENT_COLOR = '#a83232'
RANDOM_STATE = 42
CORR_LIMIT = 0.7
IMPORTANCE_CUMSUM_THRESHOLD = 0.90

# Загрузка ручного набора + исходных колонок для генерации
print('Загрузка ручного набора признаков')
df_manual = pd.read_csv(MANUAL_PATH)
print(f'Размерность ручного набора: {df_manual.shape}')

# Нам нужны и базовые числовые поля для генерации новых признаков

base_features = [
    'story_point', 'description_length', 'description_code_length',
    'issue_links_count', 'blocking_links_count', 'duplicate_links_count',
    'components_count', 'comments_count', 'is_in_sprint',
    'day_of_week', 'month', 'quarter', 'tasks_created_same_day',
    'assignee_mean_duration', 'assignee_median_duration', 'assignee_std_duration',
    'assignee_delay_rate', 'assignee_completed_count',
    'assignee_open_tasks_count', 'assignee_days_since_first_task', 'has_history',
    'team_mean_duration', 'team_median_duration', 'team_completed_tasks_count',
    'team_unique_assignees', 'team_delay_rate',
]

df = df_manual.copy()
train_mask = df['_split'] == 'train'

# Генерация кандидатов
print('\nГенерация кандидатов')
generated_features = []

# Агрегаты по проекту
print('  Группа 1: Агрегаты по проекту')
project_agg_features = ['story_point', 'description_length', 'comments_count', 'issue_links_count']
project_agg = df[train_mask].groupby('project_id')[project_agg_features].agg(['mean', 'std']).reset_index()
project_agg.columns = ['project_id'] + [f'gen_project_{stat}_{col}' for col in project_agg_features for stat in ['mean', 'std']]
df = df.merge(project_agg, on='project_id', how='left')
generated_features.extend([c for c in project_agg.columns if c.startswith('gen_')])
print(f'    Создано: {len([c for c in project_agg.columns if c.startswith("gen_")])}')

# Агрегаты по исполнителю
print('  Группа 2: Агрегаты по исполнителю')
assignee_agg_features = ['description_length', 'comments_count', 'issue_links_count', 'story_point']
assignee_agg = (df[train_mask].dropna(subset=['assignee_id'])
                .groupby('assignee_id')[assignee_agg_features].agg(['mean', 'std']).reset_index())
assignee_agg.columns = ['assignee_id'] + [f'gen_assignee_{stat}_{col}' for col in assignee_agg_features for stat in ['mean', 'std']]
df = df.merge(assignee_agg, on='assignee_id', how='left')
new_assignee_cols = [c for c in assignee_agg.columns if c.startswith('gen_')]
df[new_assignee_cols] = df[new_assignee_cols].fillna(0)
generated_features.extend(new_assignee_cols)
print(f'    Создано: {len(new_assignee_cols)}')

# Попарные взаимодействия
print('  Группа 3: Попарные взаимодействия')
key_numeric = ['story_point', 'description_length', 'issue_links_count',
               'comments_count', 'assignee_mean_duration', 'assignee_completed_count']
pairwise_count = 0
for i, a in enumerate(key_numeric):
    for b in key_numeric[i+1:]:
        # Произведение
        df[f'gen_mul_{a}_{b}'] = df[a] * df[b]
        generated_features.append(f'gen_mul_{a}_{b}')
        # Отношение 
        df[f'gen_div_{a}_{b}'] = df[a] / (df[b] + 1) # (защита от деления на ноль)
        generated_features.append(f'gen_div_{a}_{b}')
        pairwise_count += 2
print(f'    Создано: {pairwise_count}')

# Логарифмы правоскошенных
print('  Группа 4: Логарифмы')
log_features = ['story_point', 'description_length', 'description_code_length',
                'issue_links_count', 'blocking_links_count', 'duplicate_links_count',
                'comments_count', 'components_count']
for f in log_features:
    df[f'gen_log_{f}'] = np.log1p(df[f].clip(lower=0))
    generated_features.append(f'gen_log_{f}')
print(f'    Создано: {len(log_features)}')

# Бинарные индикаторы (пороги по train)
print('  Группа 5: Бинарные индикаторы')
desc_length_p75 = df.loc[train_mask, 'description_length'].quantile(0.75)
busy_day_p75 = df.loc[train_mask, 'tasks_created_same_day'].quantile(0.75)

df['gen_is_large_task'] = (df['story_point'] > 8).astype(int)
df['gen_has_blockers'] = (df['blocking_links_count'] > 0).astype(int)
df['gen_has_duplicates'] = (df['duplicate_links_count'] > 0).astype(int)
df['gen_has_components'] = (df['components_count'] > 0).astype(int)
df['gen_has_comments'] = (df['comments_count'] > 0).astype(int)
df['gen_has_long_description'] = (df['description_length'] > desc_length_p75).astype(int)
df['gen_has_code_in_description'] = (df['description_code_length'] > 0).astype(int)
df['gen_is_weekend_creation'] = df['day_of_week'].isin([5, 6]).astype(int)
df['gen_is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
df['gen_is_busy_day'] = (df['tasks_created_same_day'] > busy_day_p75).astype(int)

binary_cols = [c for c in df.columns if c.startswith('gen_is_') or c.startswith('gen_has_')]
generated_features.extend(binary_cols)
print(f'    Создано: {len(binary_cols)}')

# Производные от исполнителя
print('  Группа 6: Производные от исполнителя')
df['gen_assignee_load_ratio'] = df['assignee_open_tasks_count'] / (df['assignee_completed_count'] + 1)
df['gen_assignee_experience_per_task'] = df['assignee_days_since_first_task'] / (df['assignee_completed_count'] + 1)
df['gen_assignee_consistency'] = df['assignee_std_duration'] / (df['assignee_mean_duration'] + 1)
derived_cols = ['gen_assignee_load_ratio', 'gen_assignee_experience_per_task', 'gen_assignee_consistency']
generated_features.extend(derived_cols)
print(f'    Создано: {len(derived_cols)}')

# Заменяем inf/-inf на 0 во всех сгенерированных признаках
df[generated_features] = df[generated_features].replace([np.inf, -np.inf], 0).fillna(0)

print(f'\nИтого сгенерировано кандидатов: {len(generated_features)}')

# Использую SULOV Featurewiz
print('\nЭтап SULOV: отсечение коррелированных пар')

sulov_succeeded = False
sulov_features = generated_features.copy()

try:
    from featurewiz import FeatureWiz
    print(' FeatureWiz API')

    fw = FeatureWiz(
        corr_limit=CORR_LIMIT,
        verbose=0,
        feature_engg='',
        category_encoders='',
    )
    X_train_gen = df.loc[train_mask, generated_features].copy()
    y_train_reg = df.loc[train_mask, 'target_log_days']

    fw.fit(X_train_gen, y_train_reg)
    sulov_features = list(fw.features) if hasattr(fw, 'features') else generated_features
    sulov_succeeded = True
    print(f'  SULOV отобрал {len(sulov_features)} из {len(generated_features)}')
except Exception as e:
    print(f'  Featurewiz упал ({type(e).__name__}: {str(e)[:100]})')
    print('  Использую fallback: ручная корреляционная фильтрация')
 # Пайплайн падал несколько раз, сделал ручками fallback в виде простой корреляционной фильтрации
if not sulov_succeeded:
    # Из коррелированной пары оставляем тот, у кого выше |corr| с target_log_days
    print('  Расчёт корреляций между кандидатами и с target')
    X_train_gen = df.loc[train_mask, generated_features].copy()
    y_train_reg = df.loc[train_mask, 'target_log_days']

    # Корреляции с target
    target_corr = X_train_gen.apply(lambda col: abs(col.corr(y_train_reg))).fillna(0)

    # Парные корреляции
    corr_matrix = X_train_gen.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > CORR_LIMIT:
                # Бросаем тот у кого меньше корреляция с target
                if target_corr.get(row, 0) < target_corr.get(col, 0):
                    to_drop.add(row)
                else:
                    to_drop.add(col)

    sulov_features = [f for f in generated_features if f not in to_drop]
    print(f'  Fallback оставил {len(sulov_features)} из {len(generated_features)}')

# Catboost feature importance
print('\nЭтап CatBoost: вычисление важностей')

from catboost import CatBoostRegressor, CatBoostClassifier

X_train_sulov = df.loc[train_mask, sulov_features]

def select_top_features(importances_df, threshold=IMPORTANCE_CUMSUM_THRESHOLD): 
    df_sorted = importances_df.sort_values('importance', ascending=False).reset_index(drop=True) # Возвращает признаки, покрывающие cumsum threshold от суммы важностей
    df_sorted['cumsum_share'] = df_sorted['importance'].cumsum() / df_sorted['importance'].sum()
    selected = df_sorted[df_sorted['cumsum_share'] <= threshold]['feature'].tolist()
    # Гарантируем минимум один признак
    if len(selected) == 0:
        selected = [df_sorted.iloc[0]['feature']]
    # Если threshold обрезал слишком грубо, добавим ещё один признак, превышающий порог
    if len(selected) < len(df_sorted):
        next_idx = len(selected)
        if df_sorted.iloc[next_idx - 1]['cumsum_share'] < threshold:
            selected.append(df_sorted.iloc[next_idx]['feature'])
    return selected, df_sorted

# Для регрессии
print('  CatBoost Regressor (target_log_days)')
y_train_reg = df.loc[train_mask, 'target_log_days']
cb_reg = CatBoostRegressor(
    iterations=500, depth=6, learning_rate=0.05,
    random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
)
cb_reg.fit(X_train_sulov, y_train_reg)
imp_reg = pd.DataFrame({
    'feature': sulov_features,
    'importance': cb_reg.feature_importances_,
})
top_reg, sorted_reg = select_top_features(imp_reg)
print(f'    Отобрано для регрессии: {len(top_reg)} из {len(sulov_features)}')

# Для классификации
print('  CatBoost Classifier (target_is_delayed)')
y_train_clf = df.loc[train_mask, 'target_is_delayed']
cb_clf = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05,
    random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
)
cb_clf.fit(X_train_sulov, y_train_clf)
imp_clf = pd.DataFrame({
    'feature': sulov_features,
    'importance': cb_clf.feature_importances_,
})
top_clf, sorted_clf = select_top_features(imp_clf)
print(f'    Отобрано для классификации: {len(top_clf)} из {len(sulov_features)}')

# Собираем финальные наборы
# Каждый набор это служебные колонки + базовые "сырые" атрибуты задачи + отобранные авто-признаки
service_cols = ['issue_id', '_split', 'target_log_days', 'target_is_delayed',
                'assignee_id', 'project_id']

# Базовые признаки задачи всегда сохраняем (issue_type, priority, story_point и т. п.)
# Они нужны в препроцессинге (один раз я их забыл)
base_task_features = [
    'issue_type', 'priority', 'story_point',
    'description_length', 'description_code_length',
    'issue_links_count', 'blocking_links_count', 'duplicate_links_count',
    'components_count', 'comments_count', 'is_in_sprint',
    'day_of_week', 'month', 'quarter', 'tasks_created_same_day',
    'has_history',
]

# Регрессионный набор
auto_reg_cols = service_cols + base_task_features + top_reg
df_auto_reg = df[auto_reg_cols].copy()
auto_reg_path = os.path.join(FEATURES_AUTO_DIR, 'tawos_features_auto_regression.csv')
df_auto_reg.to_csv(auto_reg_path, index=False)
print(f'\nСохранён авто-набор для регрессии: {auto_reg_path}')
print(f'  Размерность: {df_auto_reg.shape}')
print(f'  Базовых признаков задачи: {len(base_task_features)}')
print(f'  Авто-признаков (gen_*): {len(top_reg)}')

# Классификационный набор
auto_clf_cols = service_cols + base_task_features + top_clf
df_auto_clf = df[auto_clf_cols].copy()
auto_clf_path = os.path.join(FEATURES_AUTO_DIR, 'tawos_features_auto_classification.csv')
df_auto_clf.to_csv(auto_clf_path, index=False)
print(f'\nСохранён авто-набор для классификации: {auto_clf_path}')
print(f'  Размерность: {df_auto_clf.shape}')
print(f'  Базовых признаков задачи: {len(base_task_features)}')
print(f'  Авто-признаков (gen_*): {len(top_clf)}')

# График feature importance топ-20 (для регрессии)
print('\nПостроение графика важностей (регрессия)')
top20_reg = sorted_reg.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_reg['feature'], top20_reg['importance'], color=MAIN_COLOR, edgecolor='black')
ax.set_title('CatBoost feature importance (top-20, регрессия)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_catboost_importance_regression.png'))
plt.close()

# График feature importance топ-20 (для классификации)
print('Построение графика важностей (классификация)')
top20_clf = sorted_clf.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_clf['feature'], top20_clf['importance'], color=ACCENT_COLOR, edgecolor='black')
ax.set_title('CatBoost feature importance (top-20, классификация)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_catboost_importance_classification.png'))
plt.close()

# Сохранение полных таблиц важностей признаков
sorted_reg.to_csv(os.path.join(RESULTS_DIR, 'auto_importance_regression.csv'), index=False)
sorted_clf.to_csv(os.path.join(RESULTS_DIR, 'auto_importance_classification.csv'), index=False)

# Сводка
auto_summary = pd.DataFrame({
    'metric': [
        'candidates_generated',
        'sulov_used',
        'after_sulov',
        'top_for_regression',
        'top_for_classification',
        'overlap_reg_and_clf',
    ],
    'value': [
        len(generated_features),
        'featurewiz' if sulov_succeeded else 'fallback_corr_filter',
        len(sulov_features),
        len(top_reg),
        len(top_clf),
        len(set(top_reg) & set(top_clf)),
    ],
})
auto_summary.to_csv(os.path.join(RESULTS_DIR, 'auto_features_summary.csv'), index=False)

print('\nСводка пайплайна 3Б:')
for _, row in auto_summary.iterrows():
    print(f'  {row["metric"]}: {row["value"]}')

print('\nПайплайн 3Б завершён')