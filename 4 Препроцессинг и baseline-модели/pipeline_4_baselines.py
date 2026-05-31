import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, recall_score, precision_score,
)

warnings.filterwarnings('ignore')

# Пути
ROOT = r'D:\blablabla\Magistr'
FEATURES_MANUAL_PATH = os.path.join(ROOT, 'Dataset', 'features_manual', 'tawos_features_manual.csv')
FEATURES_AUTO_REG_PATH = os.path.join(ROOT, 'Dataset', 'features_auto', 'tawos_features_auto_regression.csv')
FEATURES_AUTO_CLF_PATH = os.path.join(ROOT, 'Dataset', 'features_auto', 'tawos_features_auto_classification.csv')

PREPARED_PATH = os.path.join(ROOT, 'Dataset', 'prepared', 'tawos_prepared.csv')
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
PREP_DIR = os.path.join(SPLITS_DIR, 'preprocessing_objects')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_ENCODING_K = 10
WINSORIZE_LOW = 0.01
WINSORIZE_HIGH = 0.99

    # Винсоризация по перцентилям
def winsorize_train_test(X_train, X_test, num_cols):
    bounds = {}
    for col in num_cols:
        low = X_train[col].quantile(WINSORIZE_LOW)
        high = X_train[col].quantile(WINSORIZE_HIGH)
        bounds[col] = (low, high)
        X_train[col] = X_train[col].clip(low, high)
        X_test[col] = X_test[col].clip(low, high)
    return X_train, X_test, bounds

    # Импутация числовых медианой train, категориальных - модой.
def impute_train_test(X_train, X_test, num_cols, cat_cols):
    impute_values = {}
    for col in num_cols:
        med = X_train[col].median()
        impute_values[col] = med
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    for col in cat_cols:
        mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
        impute_values[col] = mode_val
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)
    return X_train, X_test, impute_values

    # Target encoding для assignee_id со сглаживанием.   encoded = (n * mean_per_assignee + k * global_mean) / (n + k)
def target_encode_assignee(X_train, X_test, y_train, k=TARGET_ENCODING_K):
    global_mean = y_train.mean()
    df_te = pd.DataFrame({'assignee_id': X_train['assignee_id'].values, 'y': y_train.values})
    agg = df_te.groupby('assignee_id')['y'].agg(['mean', 'count']).reset_index()
    agg['te_value'] = (agg['count'] * agg['mean'] + k * global_mean) / (agg['count'] + k)
    te_map = dict(zip(agg['assignee_id'], agg['te_value']))

    X_train['te_assignee'] = X_train['assignee_id'].map(te_map).fillna(global_mean)
    X_test['te_assignee'] = X_test['assignee_id'].map(te_map).fillna(global_mean)
    return X_train, X_test, {'te_map': te_map, 'global_mean': global_mean}

    # One-Hot Encoding с drop='first'. Только категории, виденные в train
def one_hot_encode(X_train, X_test, cat_cols):
    
    encoders = {}
    for col in cat_cols:
        train_categories = sorted(X_train[col].dropna().unique().tolist())
        if len(train_categories) <= 1:
            continue
        # drop_first: удаляем первую категорию по алфавиту как базовую
        keep_categories = train_categories[1:]
        encoders[col] = {'base': train_categories[0], 'keep': keep_categories}
        for cat in keep_categories:
            X_train[f'{col}_{cat}'] = (X_train[col] == cat).astype(int)
            X_test[f'{col}_{cat}'] = (X_test[col] == cat).astype(int)
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])
    return X_train, X_test, encoders

    # Полный препроцессинг для одного набора признаков. Возвращает три версии (для всех моделей в будущем): raw, processed, scaled + объекты препроцессинга
def preprocess_dataset(df, target_col_for_te, name):
    
    print(f'\n Препроцессинг набора: {name}')

    # Разделение train/test
    train_df = df[df['_split'] == 'train'].copy()
    test_df = df[df['_split'] == 'test'].copy()
    print(f'Train: {len(train_df)}, Test: {len(test_df)}')

    # Сохранение целевых отдельно
    y_reg_train = train_df['target_log_days'].copy()
    y_reg_test = test_df['target_log_days'].copy()
    y_clf_train = train_df['target_is_delayed'].copy()
    y_clf_test = test_df['target_is_delayed'].copy()

    # Сохранение оригинальных resolution_days для метрик регрессии в исходной шкале
    # Восстанавливаем из target_log_days (точно): np.expm1
    y_reg_train_days = np.expm1(y_reg_train)
    y_reg_test_days = np.expm1(y_reg_test)

    service_cols = ['issue_id', '_split', 'target_log_days', 'target_is_delayed']
    drop_cols = service_cols + ['project_id']

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).copy()
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()

    # RAW версия: до любых преобразований (для baseline и SP-baseline)
    raw_train = X_train.copy()
    raw_test = X_test.copy()

    # Определение типов
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols and c != 'assignee_id']
    print(f'  Числовых колонок: {len(num_cols)}, категориальных: {len(cat_cols)}')

    # Шаг 1: винсоризация числовых
    X_train, X_test, winsor_bounds = winsorize_train_test(X_train, X_test, num_cols)

    # Шаг 2: импутация
    X_train, X_test, impute_values = impute_train_test(X_train, X_test, num_cols, cat_cols)

    # Шаг 3: target encoding для assignee_id (по target_log_days)
    X_train, X_test, te_objects = target_encode_assignee(X_train, X_test, y_reg_train, TARGET_ENCODING_K)

    # Удаляем assignee_id (после TE он не нужен в данных модели)
    X_train = X_train.drop(columns=['assignee_id'])
    X_test = X_test.drop(columns=['assignee_id'])

    # Шаг 4: One-Hot Encoding для категориальных
    X_train, X_test, ohe_encoders = one_hot_encode(X_train, X_test, cat_cols)

    # PROCESSED версия (для деревьев и XGBoost)
    processed_train = X_train.copy()
    processed_test = X_test.copy()

    # Шаг 5: StandardScaler (для линейных, k-NN, TabNet)
    scaler = StandardScaler()
    scaled_train_arr = scaler.fit_transform(X_train.values)
    scaled_test_arr = scaler.transform(X_test.values)
    scaled_train = pd.DataFrame(scaled_train_arr, columns=X_train.columns, index=X_train.index)
    scaled_test = pd.DataFrame(scaled_test_arr, columns=X_test.columns, index=X_test.index)

    print(f'  Финальное число признаков (после OHE): {X_train.shape[1]}')

    # Сохранение
    set_dir = os.path.join(SPLITS_DIR, name)
    os.makedirs(os.path.join(set_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(set_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(set_dir, 'scaled'), exist_ok=True)

    # RAW
    raw_train.to_csv(os.path.join(set_dir, 'raw', 'X_train.csv'), index=False)
    raw_test.to_csv(os.path.join(set_dir, 'raw', 'X_test.csv'), index=False)

    # PROCESSED
    processed_train.to_csv(os.path.join(set_dir, 'processed', 'X_train.csv'), index=False)
    processed_test.to_csv(os.path.join(set_dir, 'processed', 'X_test.csv'), index=False)

    # SCALED
    scaled_train.to_csv(os.path.join(set_dir, 'scaled', 'X_train.csv'), index=False)
    scaled_test.to_csv(os.path.join(set_dir, 'scaled', 'X_test.csv'), index=False)

    # Целевые (одинаковые для всех версий, дублируем в каждую папку для удобства)
    for sub in ['raw', 'processed', 'scaled']:
        y_reg_train.to_csv(os.path.join(set_dir, sub, 'y_reg_train.csv'), index=False, header=['target_log_days'])
        y_reg_test.to_csv(os.path.join(set_dir, sub, 'y_reg_test.csv'), index=False, header=['target_log_days'])
        y_clf_train.to_csv(os.path.join(set_dir, sub, 'y_clf_train.csv'), index=False, header=['target_is_delayed'])
        y_clf_test.to_csv(os.path.join(set_dir, sub, 'y_clf_test.csv'), index=False, header=['target_is_delayed'])
        # Дни в исходной шкале для метрик регрессии (MAPE, PRED25 и т. п.)
        y_reg_train_days.to_csv(os.path.join(set_dir, sub, 'y_reg_train_days.csv'), index=False, header=['resolution_days'])
        y_reg_test_days.to_csv(os.path.join(set_dir, sub, 'y_reg_test_days.csv'), index=False, header=['resolution_days'])

    # Объекты препроцессинга
    preprocessor = {
        'winsor_bounds': winsor_bounds,
        'impute_values': impute_values,
        'te_objects': te_objects,
        'ohe_encoders': ohe_encoders,
        'scaler': scaler,
        'final_feature_names': X_train.columns.tolist(),
        'cat_cols_original': cat_cols,
        'num_cols_original': num_cols,
    }
    with open(os.path.join(PREP_DIR, f'{name}_preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f'  Сохранены данные и pickle для {name}')

    return raw_train, raw_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_reg_train_days, y_reg_test_days


# Метрики
# Метрики регрессии в исходной шкале
def regression_metrics(y_true_days, y_pred_days):
    eps = 1e-8
    abs_pct_err = np.abs((y_true_days - y_pred_days) / (y_true_days + eps))
    mape = np.mean(abs_pct_err) * 100
    mmre = np.mean(abs_pct_err)
    pred25 = np.mean(abs_pct_err < 0.25) * 100
    mae = mean_absolute_error(y_true_days, y_pred_days)
    rmse = np.sqrt(mean_squared_error(y_true_days, y_pred_days))
    # R2 в исходной шкале (на дни) для baseline-сравнений
    r2 = r2_score(y_true_days, y_pred_days)
    return {
        'MAPE_pct': round(mape, 2),
        'MMRE': round(mmre, 4),
        'PRED25_pct': round(pred25, 2),
        'MAE_days': round(mae, 2),
        'RMSE_days': round(rmse, 2),
        'R2': round(r2, 4),
    }

# Метрики классификации
def classification_metrics(y_true, y_pred, y_proba=None):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5
    return {
        'F1': round(f1, 4),
        'ROC_AUC': round(auc, 4),
        'Recall': round(recall, 4),
        'Precision': round(precision, 4),
    }


# 1. Загрузка трёх наборов
print('Загрузка трёх наборов признаков')
df_manual = pd.read_csv(FEATURES_MANUAL_PATH)
df_auto_reg = pd.read_csv(FEATURES_AUTO_REG_PATH)
df_auto_clf = pd.read_csv(FEATURES_AUTO_CLF_PATH)

print(f'  manual: {df_manual.shape}')
print(f'  auto_regression: {df_auto_reg.shape}')
print(f'  auto_classification: {df_auto_clf.shape}')

# 2. Препроцессинг для каждого набора
manual_data = preprocess_dataset(df_manual, 'target_log_days', 'manual')
auto_reg_data = preprocess_dataset(df_auto_reg, 'target_log_days', 'auto_regression')
auto_clf_data = preprocess_dataset(df_auto_clf, 'target_log_days', 'auto_classification')

# 3. бейслайны
print('\n Обучение baseline-моделей ')

# Используем raw-версию manual для baseline 
raw_train_m, raw_test_m, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te, y_reg_tr_days, y_reg_te_days = manual_data

baseline_results = []

# 3.1 Median baseline (регрессия)
print('\nMedian baseline (регрессия)')
median_train = np.median(y_reg_tr_days)
y_pred_median = np.full(len(y_reg_te_days), median_train)
m = regression_metrics(y_reg_te_days.values, y_pred_median)
m.update({'model': 'Median', 'task': 'regression', 'subset': 'all'})
baseline_results.append(m)
print(f'  median train value: {median_train:.2f} дней')
print(f'  {m}')

# 3.2 Majority class (классификация)
print('\nMajority class (классификация)')
majority = int(y_clf_tr.mode()[0])
y_pred_majority = np.full(len(y_clf_te), majority)
m = classification_metrics(y_clf_te.values, y_pred_majority,
                            y_proba=np.full(len(y_clf_te), majority))
m.update({'model': 'Majority', 'task': 'classification', 'subset': 'all'})
baseline_results.append(m)
print(f'  majority class: {majority}')
print(f'  {m}')

# 3.3 Story Point baseline (регрессия)
print('\nStory Point baseline (регрессия)')
# Считаем коэффициент k на train: median(resolution_days / SP)
sp_train = raw_train_m['story_point'].values
ratios = y_reg_tr_days.values / np.maximum(sp_train, 1e-3)
k_sp = np.median(ratios)
print(f'  Коэффициент k = median(days/SP) на train: {k_sp:.2f} дней на 1 SP')

sp_test = raw_test_m['story_point'].values
y_pred_sp = sp_test * k_sp
m = regression_metrics(y_reg_te_days.values, y_pred_sp)
m.update({'model': 'StoryPoint', 'task': 'regression', 'subset': 'all'})
baseline_results.append(m)
print(f'  {m}')

# 3.4 Срез метрик по has_history
print('\n Срез по has_history ')
has_hist_test = raw_test_m['has_history'].values
mask_with = has_hist_test == 1
mask_without = has_hist_test == 0

print(f'\n  Test с историей: {mask_with.sum()}, без истории: {mask_without.sum()}')

if mask_with.sum() > 0:
    m = regression_metrics(y_reg_te_days.values[mask_with], y_pred_sp[mask_with])
    m.update({'model': 'StoryPoint', 'task': 'regression', 'subset': 'has_history=1'})
    baseline_results.append(m)
    print(f'  StoryPoint (has_history=1): {m}')

    m = classification_metrics(y_clf_te.values[mask_with], y_pred_majority[mask_with],
                                y_proba=np.full(mask_with.sum(), majority))
    m.update({'model': 'Majority', 'task': 'classification', 'subset': 'has_history=1'})
    baseline_results.append(m)

if mask_without.sum() > 0:
    m = regression_metrics(y_reg_te_days.values[mask_without], y_pred_sp[mask_without])
    m.update({'model': 'StoryPoint', 'task': 'regression', 'subset': 'has_history=0'})
    baseline_results.append(m)
    print(f'  StoryPoint (has_history=0): {m}')

    m = classification_metrics(y_clf_te.values[mask_without], y_pred_majority[mask_without],
                                y_proba=np.full(mask_without.sum(), majority))
    m.update({'model': 'Majority', 'task': 'classification', 'subset': 'has_history=0'})
    baseline_results.append(m)

# Сохранение
results_df = pd.DataFrame(baseline_results)
results_path = os.path.join(RESULTS_DIR, 'baseline_results.csv')
results_df.to_csv(results_path, index=False)
print(f'\nСохранены baseline-результаты: {results_path}')

# Сводка по препроцессингу
print('\n Структура папок Dataset/splits/ ')
for set_name in ['manual', 'auto_regression', 'auto_classification']:
    print(f'\n  {set_name}/')
    set_dir = os.path.join(SPLITS_DIR, set_name)
    for sub in ['raw', 'processed', 'scaled']:
        files = os.listdir(os.path.join(set_dir, sub))
        print(f'    {sub}/: {len(files)} файлов')

# Сводка для спокойствия
prep_summary = pd.DataFrame({
    'feature_set': ['manual', 'auto_regression', 'auto_classification'],
    'n_features_after_ohe': [
        pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'X_train.csv')).shape[1],
        pd.read_csv(os.path.join(SPLITS_DIR, 'auto_regression', 'processed', 'X_train.csv')).shape[1],
        pd.read_csv(os.path.join(SPLITS_DIR, 'auto_classification', 'processed', 'X_train.csv')).shape[1],
    ],
})
prep_summary.to_csv(os.path.join(RESULTS_DIR, 'preprocessing_summary.csv'), index=False)
print('\nИтоговое число признаков после OHE:')
print(prep_summary.to_string(index=False))

print('\nПайплайн 4 завершён')