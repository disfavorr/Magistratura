import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Пути
ROOT = r'D:\blablabla\Magistr'
DATA_PATH = os.path.join(ROOT, 'Dataset', 'raw', 'tawos_clean.csv')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'eda_plots')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Стиль графиков
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'

# Цвета
MAIN_COLOR = '#3a76b5'
ACCENT_COLOR = '#a83232'

# Загрузка
print('Загрузка датасета')
df = pd.read_csv(DATA_PATH)
print(f'Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов')
print(f'Столбцы: {list(df.columns)}')

# Типы и пропуски
print('\nТипы данных:')
print(df.dtypes)

missing_share = (df.isna().mean() * 100).sort_values(ascending=False)
print('\nДоля пропусков, проценты:')
print(missing_share)

# Сохранение базовой статистики
df.describe(include='all').to_csv(os.path.join(RESULTS_DIR, 'eda_describe_full.csv'))
missing_share.to_frame('missing_pct').to_csv(os.path.join(RESULTS_DIR, 'eda_missing.csv'))

# Целевая переменная
print('\nresolution_days статистика:')
print(df['resolution_days'].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

below_half_day = int((df['resolution_days'] < 0.5).sum())
sp_p99 = df['story_point'].quantile(0.99)
above_p99 = int((df['story_point'] > sp_p99).sum())
print(f'\nЗадач короче 0.5 дня: {below_half_day} ({below_half_day / len(df) * 100:.1f}%)')
print(f'Story Point выше 99-го перцентиля ({sp_p99:.0f}): {above_p99}')

# График 01: распределение target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['resolution_days'].clip(upper=df['resolution_days'].quantile(0.99)),
             bins=80, color=MAIN_COLOR, edgecolor='black', alpha=0.75)
axes[0].set_title('Длительность задач, исходная шкала (обрезано по 99-му перцентилю)')
axes[0].set_xlabel('Длительность, дней')
axes[0].set_ylabel('Число задач')

axes[1].hist(np.log1p(df['resolution_days']), bins=80,
             color=MAIN_COLOR, edgecolor='black', alpha=0.75)
axes[1].set_title('Длительность задач, log1p-шкала')
axes[1].set_xlabel('log(1 + длительность)')
axes[1].set_ylabel('Число задач')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_target_distribution.png'))
plt.close()
print('Сохранён график 01_target_distribution.png')

# График 02: target по типам задач
fig, ax = plt.subplots(figsize=(8, 5))
type_order = df.groupby('issue_type')['resolution_days'].median().sort_values().index
sns.boxplot(data=df, x='issue_type', y='resolution_days',
            order=type_order, ax=ax, showfliers=False, color=MAIN_COLOR)
ax.set_title('Длительность задач по типу')
ax.set_xlabel('Тип задачи')
ax.set_ylabel('Длительность, дней')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_target_by_type.png'))
plt.close()
print('Сохранён график 02_target_by_type.png')

# График 03: топ-15 проектов
top_projects = df['project_name'].value_counts().head(15)
project_medians = df[df['project_name'].isin(top_projects.index)] \
    .groupby('project_name')['resolution_days'].median().reindex(top_projects.index)

fig, ax1 = plt.subplots(figsize=(11, 5))
x = np.arange(len(top_projects))
ax1.bar(x, top_projects.values, color=MAIN_COLOR, alpha=0.7, label='Число задач')
ax1.set_xticks(x)
ax1.set_xticklabels(top_projects.index, rotation=45, ha='right')
ax1.set_ylabel('Число задач', color=MAIN_COLOR)
ax1.set_xlabel('Проект')
ax1.tick_params(axis='y', labelcolor=MAIN_COLOR)

ax2 = ax1.twinx()
ax2.plot(x, project_medians.values, color=ACCENT_COLOR, marker='o', linewidth=2, label='Медиана')
ax2.set_ylabel('Медиана длительности, дней', color=ACCENT_COLOR)
ax2.tick_params(axis='y', labelcolor=ACCENT_COLOR)
ax2.grid(False)

plt.title('Топ-15 проектов по числу задач')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_top_projects.png'))
plt.close()
print('Сохранён график 03_top_projects.png')

# График 04: Story Points
fig, ax = plt.subplots(figsize=(8, 4))
sp_positive = df['story_point'].dropna()
sp_positive = sp_positive[sp_positive > 0]
bins = np.logspace(0, np.log10(sp_positive.max() + 1), 60)
ax.hist(sp_positive, bins=bins, color=MAIN_COLOR, edgecolor='black', alpha=0.75)
ax.set_xscale('log')
ax.set_title('Распределение Story Points, log-шкала')
ax.set_xlabel('Story Points')
ax.set_ylabel('Число задач')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_story_point_distribution.png'))
plt.close()
print('Сохранён график 04_story_point_distribution.png')

# График 05: SP x duration
fig, ax = plt.subplots(figsize=(8, 5))
sample = df[(df['story_point'] > 0) & (df['resolution_days'] > 0)] \
    .sample(min(5000, len(df)), random_state=42)
ax.scatter(sample['story_point'], sample['resolution_days'],
           alpha=0.2, s=10, color=MAIN_COLOR)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Связь Story Point и фактической длительности')
ax.set_xlabel('Story Points')
ax.set_ylabel('Длительность, дней')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_sp_vs_duration.png'))
plt.close()
print('Сохранён график 05_sp_vs_duration.png')

# График 06: пропуски
fig, ax = plt.subplots(figsize=(10, 5))
missing_filtered = missing_share[missing_share > 0].sort_values()
if len(missing_filtered) > 0:
    ax.barh(missing_filtered.index, missing_filtered.values, color=MAIN_COLOR)
    ax.set_title('Доля пропусков по колонкам')
    ax.set_xlabel('Доля пропусков, проценты')
else:
    ax.text(0.5, 0.5, 'Пропусков нет', ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_missing_share.png'))
plt.close()
print('Сохранён график 06_missing_share.png')

# График 07: распределение по годам
df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
df['year'] = df['creation_date'].dt.year
year_counts = df['year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(year_counts.index.astype(int), year_counts.values, color=MAIN_COLOR, edgecolor='black')
ax.set_title('Распределение задач по годам создания')
ax.set_xlabel('Год')
ax.set_ylabel('Число задач')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_year_distribution.png'))
plt.close()
print('Сохранён график 07_year_distribution.png')

# Сводная таблица ключевых метрик
summary = pd.DataFrame({
    'metric': [
        'rows',
        'columns',
        'unique_projects',
        'unique_assignees',
        'unique_priority_values',
        'priority_nan_share_pct',
        'resolution_days_min',
        'resolution_days_p01',
        'resolution_days_median',
        'resolution_days_mean',
        'resolution_days_p99',
        'resolution_days_max',
        'tasks_below_0_5_day',
        'tasks_below_0_5_day_pct',
        'story_point_min',
        'story_point_median',
        'story_point_p99',
        'story_point_max',
        'tasks_with_sprint_pct',
        'tasks_with_assignee_pct',
        'first_year',
        'last_year',
    ],
    'value': [
        df.shape[0],
        df.shape[1],
        df['project_name'].nunique(),
        df['assignee_id'].nunique(),
        df['priority'].nunique(dropna=True),
        round(df['priority'].isna().mean() * 100, 2),
        round(df['resolution_days'].min(), 4),
        round(df['resolution_days'].quantile(0.01), 4),
        round(df['resolution_days'].median(), 2),
        round(df['resolution_days'].mean(), 2),
        round(df['resolution_days'].quantile(0.99), 2),
        round(df['resolution_days'].max(), 2),
        below_half_day,
        round(below_half_day / len(df) * 100, 2),
        df['story_point'].min(),
        df['story_point'].median(),
        df['story_point'].quantile(0.99),
        df['story_point'].max(),
        round(df['sprint_id'].notna().mean() * 100, 2),
        round(df['assignee_id'].notna().mean() * 100, 2),
        int(df['year'].min()) if df['year'].notna().any() else None,
        int(df['year'].max()) if df['year'].notna().any() else None,
    ],
})
summary.to_csv(os.path.join(RESULTS_DIR, 'eda_summary.csv'), index=False)

# Распределение категорий
df['priority'].fillna('NaN').value_counts().to_frame('count').to_csv(
    os.path.join(RESULTS_DIR, 'eda_priority_counts.csv'))
df['issue_type'].value_counts().to_frame('count').to_csv(
    os.path.join(RESULTS_DIR, 'eda_issue_type_counts.csv'))

# Финальный вывод
print('\nКлючевые сводные статистики:')
for _, row in summary.iterrows():
    print(f'  {row["metric"]}: {row["value"]}')

print('\nСохранено:')
print(f'  Графики: {PLOTS_DIR}')
print(f'  Сводка: {os.path.join(RESULTS_DIR, "eda_summary.csv")}')
print(f'  Описательная статистика: {os.path.join(RESULTS_DIR, "eda_describe_full.csv")}')
print(f'  Пропуски: {os.path.join(RESULTS_DIR, "eda_missing.csv")}')
print(f'  Распределение priority: {os.path.join(RESULTS_DIR, "eda_priority_counts.csv")}')
print(f'  Распределение issue_type: {os.path.join(RESULTS_DIR, "eda_issue_type_counts.csv")}')

print('\nПайплайн 1 завершён')