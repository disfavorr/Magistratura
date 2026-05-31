import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Пути
ROOT = r'D:\blablabla\Magistr'
PREPARED_PATH = os.path.join(ROOT, 'Dataset', 'prepared', 'tawos_prepared.csv')
FEATURES_DIR = os.path.join(ROOT, 'Dataset', 'features_manual')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка
print('Загрузка подготовленного датасета')
df = pd.read_csv(PREPARED_PATH, parse_dates=['creation_date', 'resolution_date'])
print(f'Размерность: {df.shape[0]} строк x {df.shape[1]} столбцов')
print(f'Train: {(df["_split"] == "train").sum()}, Test: {(df["_split"] == "test").sum()}')

# Сохранение исходных колонок
original_cols = set(df.columns)

# Контекстные признаки
print('\nПостроение контекстных признаков')
df['day_of_week'] = df['creation_date'].dt.dayofweek
df['month'] = df['creation_date'].dt.month
df['quarter'] = df['creation_date'].dt.quarter

# tasks_created_same_day: число задач в тот же день в том же проекте
df['_creation_day'] = df['creation_date'].dt.date
same_day_counts = df.groupby(['project_id', '_creation_day']).size().reset_index(name='tasks_created_same_day')
df = df.merge(same_day_counts, on=['project_id', '_creation_day'], how='left')
# Текущая задача сама входит в счёт: вычитаем единицу
df['tasks_created_same_day'] = df['tasks_created_same_day'] - 1
df = df.drop(columns=['_creation_day'])
print(f'  day_of_week, month, quarter, tasks_created_same_day готовы')
print(f'  tasks_created_same_day: median={df["tasks_created_same_day"].median()}, '
      f'max={df["tasks_created_same_day"].max()}')

# Признаки исполнителя
print('\nПостроение признаков исполнителя на train')
train_df = df[df['_split'] == 'train'].copy()

# Агрегаты по assignee_id на train
assignee_agg = (train_df.dropna(subset=['assignee_id'])
                .groupby('assignee_id')
                .agg(
                    assignee_mean_duration=('resolution_days', 'mean'),
                    assignee_median_duration=('resolution_days', 'median'),
                    assignee_std_duration=('resolution_days', 'std'),
                    assignee_delay_rate=('target_is_delayed', 'mean'),
                    assignee_completed_count=('issue_id', 'count'),
                )
                .reset_index())
print(f'  Уникальных исполнителей в train: {len(assignee_agg)}')

# days_since_first_task: дней с первой задачи исполнителя в train
first_task = (train_df.dropna(subset=['assignee_id'])
              .groupby('assignee_id')['creation_date']
              .min()
              .reset_index()
              .rename(columns={'creation_date': '_first_task_date'}))

# Применение через merge
df = df.merge(assignee_agg, on='assignee_id', how='left')
df = df.merge(first_task, on='assignee_id', how='left')

# has_history: 1 если исполнитель встречался в train, 0 иначе
df['has_history'] = df['assignee_completed_count'].notna().astype(int)

# Заполнение NaN для исполнителей без истории (новые в test или NaN assignee)
agg_cols = ['assignee_mean_duration', 'assignee_median_duration', 'assignee_std_duration',
            'assignee_delay_rate', 'assignee_completed_count']
df[agg_cols] = df[agg_cols].fillna(0)

# days_since_first_task: дни с первой задачи; 0 если нет истории
df['assignee_days_since_first_task'] = (
    (df['creation_date'] - df['_first_task_date']).dt.total_seconds() / 86400
).fillna(0)
df['assignee_days_since_first_task'] = df['assignee_days_since_first_task'].clip(lower=0)
df = df.drop(columns=['_first_task_date'])

print(f'  Задач с has_history=1: {df["has_history"].sum()} ({df["has_history"].mean() * 100:.2f}%)')
print(f'  assignee_completed_count: median={df["assignee_completed_count"].median()}')
print(f'  assignee_mean_duration: train_median={df.loc[df["_split"] == "train", "assignee_mean_duration"].median():.2f}')

# Assignee_open_tasks_count - динамический признак на момент cutoff
# Для каждой задачи смотрим число задач исполнителя, у которых creation <= current_creation < resolution
print('\nРасчёт assignee_open_tasks_count (динамически на cutoff)')
# Это требует двойного цикла по исполнителям. Сортируем.
def calc_open_tasks(group):
    # group отсортирована по creation_date
    open_counts = []
    creations = group['creation_date'].values
    resolutions = group['resolution_date'].values
    for i, cur in enumerate(creations):
        # сколько задач началось до cur и ещё не закрылось на момент cur
        started_before = (creations < cur)
        not_finished = (resolutions > cur)
        open_counts.append(int(np.sum(started_before & not_finished)))
    group['assignee_open_tasks_count'] = open_counts
    return group

df_sorted = df.sort_values(['assignee_id', 'creation_date']).reset_index()
# Применяем только к задачам с непустым assignee_id
df_with_assignee = df_sorted.dropna(subset=['assignee_id']).copy()
df_no_assignee = df_sorted[df_sorted['assignee_id'].isna()].copy()
df_no_assignee['assignee_open_tasks_count'] = 0

df_with_assignee = df_with_assignee.groupby('assignee_id', group_keys=False).apply(calc_open_tasks)

df_combined = pd.concat([df_with_assignee, df_no_assignee]).sort_values('index').reset_index(drop=True)
df['assignee_open_tasks_count'] = df_combined['assignee_open_tasks_count'].values
print(f'  assignee_open_tasks_count: median={df["assignee_open_tasks_count"].median()}, '
      f'max={df["assignee_open_tasks_count"].max()}')

# Агрегаты команды
print('\nПостроение агрегатов команды по project_id на train')
team_agg = (train_df
            .groupby('project_id')
            .agg(
                team_mean_duration=('resolution_days', 'mean'),
                team_median_duration=('resolution_days', 'median'),
                team_completed_tasks_count=('issue_id', 'count'),
                team_unique_assignees=('assignee_id', 'nunique'),
                team_delay_rate=('target_is_delayed', 'mean'),
            )
            .reset_index())
print(f'  Уникальных проектов в train: {len(team_agg)}')

df = df.merge(team_agg, on='project_id', how='left')

# Если в test попал проект, которого не было в train (проверка на всякий случай)
team_cols = ['team_mean_duration', 'team_median_duration',
             'team_completed_tasks_count', 'team_unique_assignees', 'team_delay_rate']
n_missing_team = df[team_cols[0]].isna().sum()
if n_missing_team > 0:
    print(f'  Внимание: {n_missing_team} задач без записи в team agg (новый проект в test)')
    # fallback: глобальные средние из train
    fallback = train_df.agg({
        'resolution_days': ['mean', 'median'],
        'target_is_delayed': 'mean',
    })
    df['team_mean_duration'] = df['team_mean_duration'].fillna(fallback['resolution_days']['mean'])
    df['team_median_duration'] = df['team_median_duration'].fillna(fallback['resolution_days']['median'])
    df['team_delay_rate'] = df['team_delay_rate'].fillna(fallback['target_is_delayed']['mean'])
    df['team_completed_tasks_count'] = df['team_completed_tasks_count'].fillna(0)
    df['team_unique_assignees'] = df['team_unique_assignees'].fillna(0)


# Ручной набор
manual_features = [
    # Контекстные
    'day_of_week', 'month', 'quarter', 'tasks_created_same_day',
    # Атрибуты задачи 
    'issue_type', 'priority', 'story_point',
    'description_length', 'description_code_length',
    'issue_links_count', 'blocking_links_count', 'duplicate_links_count',
    'components_count', 'comments_count', 'is_in_sprint',
    # Признаки исполнителя
    'assignee_mean_duration', 'assignee_median_duration', 'assignee_std_duration',
    'assignee_delay_rate', 'assignee_completed_count',
    'assignee_open_tasks_count', 'assignee_days_since_first_task', 'has_history',
    # Агрегаты команды
    'team_mean_duration', 'team_median_duration', 'team_completed_tasks_count',
    'team_unique_assignees', 'team_delay_rate',
]
service_cols = ['issue_id', '_split', 'target_log_days', 'target_is_delayed',
                'assignee_id', 'project_id']

print(f'\nЧисло ручных признаков: {len(manual_features)}')

final_df = df[service_cols + manual_features].copy()
print(f'Размерность финального набора: {final_df.shape[0]} строк x {final_df.shape[1]} столбцов')
print(f'Из них служебных: {len(service_cols)}, признаков: {len(manual_features)}')

# Проверка на NaN в признаках
nan_in_features = final_df[manual_features].isna().sum()
nan_features = nan_in_features[nan_in_features > 0]
if len(nan_features) > 0:
    print(f'\nВнимание: NaN в признаках:')
    print(nan_features)
else:
    print('\nNaN в признаках: нет')

# Сохранение
output_path = os.path.join(FEATURES_DIR, 'tawos_features_manual.csv')
final_df.to_csv(output_path, index=False)
print(f'\nСохранён ручной набор: {output_path}')

# Описательная статистика признаков
manual_stats = final_df[manual_features].describe(include='all').T[['mean', 'std', 'min', '50%', 'max']]
manual_stats.columns = ['mean', 'std', 'min', 'median', 'max']
manual_stats.to_csv(os.path.join(RESULTS_DIR, 'manual_features_stats.csv'))
print(f'Сохранена статистика признаков: {os.path.join(RESULTS_DIR, "manual_features_stats.csv")}')

print('\nКлючевые статистики ручного набора:')
print(f'  Признаков всего: {len(manual_features)}')
print(f'  Из них числовых: {final_df[manual_features].select_dtypes(include=np.number).shape[1]}')
print(f'  Из них категориальных: {final_df[manual_features].select_dtypes(include=["object", "category"]).shape[1]}')
print(f'  has_history=1: {final_df["has_history"].mean() * 100:.2f}% задач')
print(f'  is_in_sprint=1: {final_df["is_in_sprint"].mean() * 100:.2f}% задач')

print('\nПайплайн 3А завершён')