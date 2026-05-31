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
DATA_PATH = os.path.join(ROOT, 'Dataset', 'raw', 'tawos_clean.csv')
PREPARED_DIR = os.path.join(ROOT, 'Dataset', 'prepared')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'eda_plots')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')

os.makedirs(PREPARED_DIR, exist_ok=True)
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

MAIN_COLOR = '#3a76b5'
ACCENT_COLOR = '#a83232'

# Параметры
MIN_RESOLUTION_DAYS = 1 # огромная часть данных была в диапазоне до 1, посчитал за человеческий фактор (задачи меньше дня не рассматриваю)
MAX_STORY_POINT = 20
DELAY_FACTOR = 1.5
TRAIN_RATIO = 0.8

# Mapping для priority в 6 категорий
PRIORITY_MAP = {
    'Blocker': 'Blocker', 'Blocker - P1': 'Blocker',
    'Critical': 'Critical', 'Critical - P2': 'Critical', 'Highest': 'Critical',
    'Major': 'Major', 'Major - P3': 'Major', 'High': 'Major',
    'Medium': 'Medium', 'To be reviewed': 'Medium',
    'Minor': 'Minor', 'Minor - P4': 'Minor', 'Low': 'Minor',
    'Trivial': 'Minor', 'Trivial - P5': 'Minor', 'Lowest': 'Minor',
}

# Загрузка
print('Загрузка датасета') # Эти принты по большей части для того, чтобы понимать где крашутся скрипты
df = pd.read_csv(DATA_PATH)
print(f'Исходная размерность: {df.shape[0]} строк x {df.shape[1]} столбцов')

# Сохранение исходного размера для отчёта о фильтрации
n_initial = len(df)

# Фильтр по длительности
n_before = len(df)
df = df[df['resolution_days'] >= MIN_RESOLUTION_DAYS].copy()
n_after = len(df)
print(f'Фильтр resolution_days >= {MIN_RESOLUTION_DAYS}: удалено {n_before - n_after}, осталось {n_after}')

# Фильтр по Story Points
n_before = len(df)
df = df[df['story_point'] <= MAX_STORY_POINT].copy()
n_after = len(df)
print(f'Фильтр story_point <= {MAX_STORY_POINT}: удалено {n_before - n_after}, осталось {n_after}')

# Удаление leakage-полей
leakage_cols = ['title_changed', 'sp_changed']
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
print(f'Удалены поля с data leakage: {leakage_cols}')

# Унификация priority
n_unmapped = df['priority'].notna().sum() - df['priority'].isin(PRIORITY_MAP.keys()).sum()
df['priority'] = df['priority'].map(PRIORITY_MAP).fillna('Unknown')
print(f'Унификация priority: 16 значений -> {df["priority"].nunique()} категорий')
print(f'Распределение priority после унификации:')
print(df['priority'].value_counts())
if n_unmapped > 0:
    print(f'Внимание: {n_unmapped} значений priority не попали в mapping и стали Unknown')

# Бинарный признак is_in_sprint
df['is_in_sprint'] = df['sprint_id'].notna().astype(int)
print(f'is_in_sprint: {df["is_in_sprint"].mean() * 100:.2f}% задач в спринтах')

# Регрессионный target
df['target_log_days'] = np.log1p(df['resolution_days'])
print(f'target_log_days: min={df["target_log_days"].min():.4f}, '
      f'median={df["target_log_days"].median():.4f}, '
      f'max={df["target_log_days"].max():.4f}')

# Time-based split
df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')
df = df.sort_values('creation_date').reset_index(drop=True)

split_idx = int(len(df) * TRAIN_RATIO)
df['_split'] = 'test'
df.iloc[:split_idx, df.columns.get_loc('_split')] = 'train'

split_date = df.iloc[split_idx]['creation_date']
print(f'\nTime-based split:')
print(f'  Train: {df["_split"].eq("train").sum()} задач, '
      f'{df.loc[df["_split"] == "train", "creation_date"].min().date()} - '
      f'{df.loc[df["_split"] == "train", "creation_date"].max().date()}')
print(f'  Test:  {df["_split"].eq("test").sum()} задач, '
      f'{df.loc[df["_split"] == "test", "creation_date"].min().date()} - '
      f'{df.loc[df["_split"] == "test", "creation_date"].max().date()}')
print(f'  Граница split: {split_date.date()}')

# Классификационный target
# Порог = DELAY_FACTOR * median(resolution_days по issue_type x project_id) на train
train_mask = df['_split'] == 'train'
thresholds = (df.loc[train_mask]
              .groupby(['issue_type', 'project_id'])['resolution_days']
              .median() * DELAY_FACTOR)
thresholds = thresholds.reset_index()
thresholds.columns = ['issue_type', 'project_id', 'delay_threshold']

# Глобальный fallback на случай если в test есть пара (type, project) которой не было в train
global_threshold = df.loc[train_mask, 'resolution_days'].median() * DELAY_FACTOR
print(f'\nГлобальный fallback порог (train median * {DELAY_FACTOR}): {global_threshold:.2f} дней')

# Применение порогов
df = df.merge(thresholds, on=['issue_type', 'project_id'], how='left')
n_fallback = df['delay_threshold'].isna().sum()
df['delay_threshold'] = df['delay_threshold'].fillna(global_threshold)
df['target_is_delayed'] = (df['resolution_days'] > df['delay_threshold']).astype(int)

print(f'Пар (issue_type, project_id) в train: {len(thresholds)}')
print(f'Задач с глобальным fallback порогом (нет пары в train): {n_fallback}')

# Распределение классов в train и test
train_delay_rate = df.loc[df['_split'] == 'train', 'target_is_delayed'].mean()
test_delay_rate = df.loc[df['_split'] == 'test', 'target_is_delayed'].mean()
print(f'\nДоля задержек в train: {train_delay_rate * 100:.2f}%')
print(f'Доля задержек в test:  {test_delay_rate * 100:.2f}%')

# Сохранение
output_path = os.path.join(PREPARED_DIR, 'tawos_prepared.csv')
df.to_csv(output_path, index=False)
print(f'\nСохранён очищенный датасет: {output_path}')
print(f'Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов')

# Также сохраняем словарь порогов отдельно для прозрачности
thresholds.to_csv(os.path.join(PREPARED_DIR, 'delay_thresholds.csv'), index=False)
print(f'Сохранён словарь порогов: {os.path.join(PREPARED_DIR, "delay_thresholds.csv")}')

# Графики

# График 08: эффект фильтра 0.5 дня
df_raw = pd.read_csv(DATA_PATH)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_raw['resolution_days'].clip(upper=df_raw['resolution_days'].quantile(0.99)),
             bins=80, color=ACCENT_COLOR, edgecolor='black', alpha=0.6)
axes[0].set_title(f'До фильтрации ({len(df_raw)} задач)')
axes[0].set_xlabel('Длительность, дней')
axes[0].set_ylabel('Число задач')
axes[0].axvline(MIN_RESOLUTION_DAYS, color='black', linestyle='--', linewidth=1.5,
                label=f'Порог {MIN_RESOLUTION_DAYS} дн')
axes[0].legend()

axes[1].hist(df['resolution_days'].clip(upper=df['resolution_days'].quantile(0.99)),
             bins=80, color=MAIN_COLOR, edgecolor='black', alpha=0.7)
axes[1].set_title(f'После фильтрации ({len(df)} задач)')
axes[1].set_xlabel('Длительность, дней')
axes[1].set_ylabel('Число задач')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_filter_effect.png'))
plt.close()
print('Сохранён график 08_filter_effect.png')

# График 09: распределение target_is_delayed
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
train_counts = df.loc[df['_split'] == 'train', 'target_is_delayed'].value_counts().sort_index()
test_counts = df.loc[df['_split'] == 'test', 'target_is_delayed'].value_counts().sort_index()

axes[0].bar(['Не задержано', 'Задержано'], train_counts.values,
            color=[MAIN_COLOR, ACCENT_COLOR], edgecolor='black')
axes[0].set_title(f'Train ({train_counts.sum()} задач)')
axes[0].set_ylabel('Число задач')
for i, v in enumerate(train_counts.values):
    axes[0].text(i, v, f'{v}\n({v / train_counts.sum() * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=11)

axes[1].bar(['Не задержано', 'Задержано'], test_counts.values,
            color=[MAIN_COLOR, ACCENT_COLOR], edgecolor='black')
axes[1].set_title(f'Test ({test_counts.sum()} задач)')
axes[1].set_ylabel('Число задач')
for i, v in enumerate(test_counts.values):
    axes[1].text(i, v, f'{v}\n({v / test_counts.sum() * 100:.1f}%)',
                 ha='center', va='bottom', fontsize=11)

plt.suptitle('Распределение target_is_delayed (порог из train)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '09_target_class_balance.png'))
plt.close()
print('Сохранён график 09_target_class_balance.png')

# Сводка
prepare_summary = pd.DataFrame({
    'metric': [
        'rows_initial',
        'rows_after_min_duration_filter',
        'rows_after_sp_filter',
        'rows_final',
        'columns_final',
        'priority_categories_after_unification',
        'is_in_sprint_share_pct',
        'split_train_size',
        'split_test_size',
        'split_boundary_date',
        'target_is_delayed_train_pct',
        'target_is_delayed_test_pct',
        'global_delay_threshold_days',
        'delay_threshold_pairs_count',
        'tasks_with_fallback_threshold',
    ],
    'value': [
        n_initial,
        len(df_raw[df_raw['resolution_days'] >= MIN_RESOLUTION_DAYS]),
        len(df),  # после обоих фильтров
        len(df),
        df.shape[1],
        df['priority'].nunique(),
        round(df['is_in_sprint'].mean() * 100, 2),
        int(train_mask.sum()),
        int((~train_mask).sum()),
        str(split_date.date()),
        round(train_delay_rate * 100, 2),
        round(test_delay_rate * 100, 2),
        round(global_threshold, 2),
        len(thresholds),
        int(n_fallback),
    ],
})
prepare_summary.to_csv(os.path.join(RESULTS_DIR, 'prepare_summary.csv'), index=False)

print('\nСводка пайплайна 2:')
for _, row in prepare_summary.iterrows():
    print(f'  {row["metric"]}: {row["value"]}')

print('\nПайплайн 2 завершён')