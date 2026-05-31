"""
Pipeline 8: Финальный анализ.
Сводная таблица, Wilcoxon на CV-парах, bootstrap CI на test, SHAP, анализ ошибок.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import shap

warnings.filterwarnings('ignore')

ROOT = r'D:\blablabla\Magistr'
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
MODELS_DIR = os.path.join(ROOT, 'Dataset', 'models')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'plots', 'analysis')
SHAP_PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'shap_plots')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_PRIMARY = '#3a76b5'
COLOR_ACCENT = '#a83232'
COLOR_BAR = '#7fbc41'

RANDOM_STATE = 42
N_BOOTSTRAP = 2000


# 1. Загрузка всех результатов

print('Загрузка результатов всех пайплайнов')

baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, 'baseline_results.csv'))
grid_df = pd.read_csv(os.path.join(RESULTS_DIR, 'grid_results_core.csv'))
optuna_df = pd.read_csv(os.path.join(RESULTS_DIR, 'optuna_results_core.csv'))
tabnet_df = pd.read_csv(os.path.join(RESULTS_DIR, 'tabnet_results.csv'))

with open(os.path.join(RESULTS_DIR, 'grid_cv_scores.pkl'), 'rb') as f:
    grid_cv_scores = pickle.load(f)
with open(os.path.join(RESULTS_DIR, 'optuna_cv_scores.pkl'), 'rb') as f:
    optuna_cv_scores = pickle.load(f)
with open(os.path.join(RESULTS_DIR, 'tabnet_cv_scores.pkl'), 'rb') as f:
    tabnet_cv_scores = pickle.load(f)

print(f'  Baseline: {len(baseline_df)} строк')
print(f'  GridSearch core: {len(grid_df)} строк')
print(f'  Optuna core: {len(optuna_df)} строк')
print(f'  TabNet: {len(tabnet_df)} строк')


# 2. Сводная сравнительная таблица

print('\nПостроение сводной таблицы')

# Унифицируем структуру
def normalize_baseline(df):
    df = df.copy()
    df['method'] = 'Baseline'
    df['feature_set'] = '-'
    df = df.rename(columns={'subset': 'subset_orig'})
    return df

def add_subset_all(df):
    if 'subset' not in df.columns:
        df['subset'] = 'all'
    return df

baseline_norm = normalize_baseline(baseline_df)
grid_norm = add_subset_all(grid_df.copy())
optuna_norm = add_subset_all(optuna_df.copy())
tabnet_norm = add_subset_all(tabnet_df.copy())

all_results = pd.concat([baseline_norm, grid_norm, optuna_norm, tabnet_norm], ignore_index=True, sort=False)

# Оставляем строки с subset='all' для главной сравнительной таблицы
if 'subset_orig' in all_results.columns:
    main_table = all_results[
        (all_results['subset_orig'].isna()) | (all_results['subset_orig'] == 'all')
    ].copy()
else:
    main_table = all_results.copy()

# Сортировка для регрессии и классификации
reg_cols = ['method', 'model', 'feature_set', 'best_cv_score',
            'MAPE_pct', 'MMRE', 'PRED25_pct', 'MAE_days', 'RMSE_days', 'R2',
            'has_history_1_MAPE_pct', 'has_history_0_new_MAPE_pct', 'has_history_0_nan_MAPE_pct']

clf_cols = ['method', 'model', 'feature_set', 'best_cv_score',
            'F1', 'ROC_AUC', 'Recall', 'Precision',
            'has_history_1_F1', 'has_history_0_new_F1', 'has_history_0_nan_F1']

reg_table = main_table[main_table['task'] == 'regression'] if 'task' in main_table.columns else pd.DataFrame()
if reg_table.empty:
    reg_table = main_table[main_table.get('task').astype(str).str.startswith('reg')] if 'task' in main_table.columns else pd.DataFrame()

# Используем универсальный фильтр по task
def filter_task(df, task_prefix):
    if 'task' not in df.columns:
        return pd.DataFrame()
    return df[df['task'].astype(str).str.startswith(task_prefix)].copy()

reg_table = filter_task(main_table, 'reg')
clf_table = filter_task(main_table, 'cl')

# Маскируем Majority Logistic запись правильно
# Берём только нужные колонки
reg_have = [c for c in reg_cols if c in reg_table.columns]
clf_have = [c for c in clf_cols if c in clf_table.columns]

reg_final = reg_table[reg_have].copy()
clf_final = clf_table[clf_have].copy()

# Сортировка по основной метрике
if 'MAPE_pct' in reg_final.columns:
    reg_final = reg_final.sort_values('MAPE_pct', na_position='last')
if 'F1' in clf_final.columns:
    clf_final = clf_final.sort_values('F1', ascending=False, na_position='last')

reg_final.to_csv(os.path.join(RESULTS_DIR, 'comparison_table_regression.csv'), index=False)
clf_final.to_csv(os.path.join(RESULTS_DIR, 'comparison_table_classification.csv'), index=False)
print(f'  Регрессия: {len(reg_final)} строк -> comparison_table_regression.csv')
print(f'  Классификация: {len(clf_final)} строк -> comparison_table_classification.csv')

# Объединённая
combined = pd.concat([reg_final.assign(track='regression'), clf_final.assign(track='classification')], ignore_index=True, sort=False)
combined.to_csv(os.path.join(RESULTS_DIR, 'comparison_table.csv'), index=False)


# 3. Wilcoxon на CV-парах (только для классических моделей)

print('\nWilcoxon signed-rank на CV-парах')

wilcoxon_results = []

# Grid vs Optuna для каждого классического эксперимента
for key in grid_cv_scores.keys():
    if key in optuna_cv_scores:
        a = np.array(grid_cv_scores[key])
        b = np.array(optuna_cv_scores[key])
        if len(a) == len(b) and len(a) >= 6:
            try:
                stat, p = wilcoxon(a, b, zero_method='wilcox')
                wilcoxon_results.append({
                    'comparison': f'Grid vs Optuna',
                    'config': key,
                    'mean_grid': round(np.mean(a), 4),
                    'mean_optuna': round(np.mean(b), 4),
                    'diff_mean': round(np.mean(b) - np.mean(a), 4),
                    'wilcoxon_stat': round(stat, 4),
                    'p_value': round(p, 4),
                    'significant_005': p < 0.05,
                    'n_pairs': len(a),
                })
            except Exception as e:
                wilcoxon_results.append({
                    'comparison': 'Grid vs Optuna',
                    'config': key,
                    'error': str(e),
                })

# XGBoost vs RF (между лидерами)
xgb_rf_pairs = [
    ('xgb_reg_manual', 'rf_reg_manual'),
    ('xgb_reg_auto_regression', 'rf_reg_auto_regression'),
    ('xgb_clf_manual', 'rf_clf_manual'),
    ('xgb_clf_auto_classification', 'rf_clf_auto_classification'),
]

for xgb_key, rf_key in xgb_rf_pairs:
    if xgb_key in grid_cv_scores and rf_key in grid_cv_scores:
        a = np.array(grid_cv_scores[xgb_key])
        b = np.array(grid_cv_scores[rf_key])
        if len(a) == len(b) and len(a) >= 6:
            try:
                stat, p = wilcoxon(a, b, zero_method='wilcox')
                wilcoxon_results.append({
                    'comparison': 'XGBoost vs RF (Grid)',
                    'config': xgb_key.replace('xgb_', '').replace('_manual', '_man').replace('_auto_regression', '_auto').replace('_auto_classification', '_auto'),
                    'mean_xgb': round(np.mean(a), 4),
                    'mean_rf': round(np.mean(b), 4),
                    'diff_mean': round(np.mean(a) - np.mean(b), 4),
                    'wilcoxon_stat': round(stat, 4),
                    'p_value': round(p, 4),
                    'significant_005': p < 0.05,
                    'n_pairs': len(a),
                })
            except Exception as e:
                pass

# XGBoost manual vs XGBoost auto
manual_auto_pairs = [
    ('xgb_reg_manual', 'xgb_reg_auto_regression'),
    ('xgb_clf_manual', 'xgb_clf_auto_classification'),
    ('rf_reg_manual', 'rf_reg_auto_regression'),
    ('rf_clf_manual', 'rf_clf_auto_classification'),
]

for man_key, auto_key in manual_auto_pairs:
    if man_key in grid_cv_scores and auto_key in grid_cv_scores:
        a = np.array(grid_cv_scores[man_key])
        b = np.array(grid_cv_scores[auto_key])
        if len(a) == len(b) and len(a) >= 6:
            try:
                stat, p = wilcoxon(a, b, zero_method='wilcox')
                wilcoxon_results.append({
                    'comparison': 'Manual vs Auto features',
                    'config': man_key,
                    'mean_manual': round(np.mean(a), 4),
                    'mean_auto': round(np.mean(b), 4),
                    'diff_mean': round(np.mean(a) - np.mean(b), 4),
                    'wilcoxon_stat': round(stat, 4),
                    'p_value': round(p, 4),
                    'significant_005': p < 0.05,
                    'n_pairs': len(a),
                })
            except Exception as e:
                pass

wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_results.csv'), index=False)
print(f'  Сохранено {len(wilcoxon_df)} результатов Wilcoxon -> wilcoxon_results.csv')


# 4. Bootstrap CI на test (2000 ресемплов)

print(f'\nBootstrap CI на test ({N_BOOTSTRAP} ресемплов)')

# Загружаем test для лидеров
def load_test(feature_set, version):
    base = os.path.join(SPLITS_DIR, feature_set, version)
    X_test = pd.read_csv(os.path.join(base, 'X_test.csv'))
    y_reg_test = pd.read_csv(os.path.join(base, 'y_reg_test.csv')).iloc[:, 0]
    y_clf_test = pd.read_csv(os.path.join(base, 'y_clf_test.csv')).iloc[:, 0]
    y_reg_test_days = pd.read_csv(os.path.join(base, 'y_reg_test_days.csv')).iloc[:, 0]
    return X_test, y_reg_test, y_clf_test, y_reg_test_days


# Получаем предсказания моделей
def get_predictions_regression(model_path):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model']

def mape_metric(y_true_d, y_pred_d):
    eps = 1e-8
    return np.mean(np.abs((y_true_d - y_pred_d) / (y_true_d + eps))) * 100

def f1_metric(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, zero_division=0)

# XGBoost manual reg (лидер регрессии)
print('  Регрессия: XGBoost manual Grid vs SP-baseline')
X_test_proc, _, _, y_test_days = load_test('manual', 'processed')
xgb_reg = get_predictions_regression(os.path.join(MODELS_DIR, 'core', 'grid_xgb_reg_manual.pkl'))
pred_log = xgb_reg.predict(X_test_proc)
pred_days_xgb = np.expm1(pred_log)

# SP baseline (k=8.38 из Пайплайна 4)
X_test_raw, _, _, _ = load_test('manual', 'raw')
sp_test = X_test_raw['story_point'].values
pred_days_sp = sp_test * 8.38

# Bootstrap MAPE для XGB и SP, разница
n_test = len(y_test_days)
boot_mape_xgb = []
boot_mape_sp = []
boot_diff = []
rng = np.random.default_rng(RANDOM_STATE)
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_test, size=n_test)
    y_true = y_test_days.values[idx]
    p_xgb = pred_days_xgb[idx]
    p_sp = pred_days_sp[idx]
    m_xgb = mape_metric(y_true, p_xgb)
    m_sp = mape_metric(y_true, p_sp)
    boot_mape_xgb.append(m_xgb)
    boot_mape_sp.append(m_sp)
    boot_diff.append(m_xgb - m_sp)

bootstrap_results = []
bootstrap_results.append({
    'comparison': 'XGBoost manual Grid vs SP-baseline',
    'metric': 'MAPE',
    'mean_a': round(np.mean(boot_mape_xgb), 2),
    'ci_low_a': round(np.percentile(boot_mape_xgb, 2.5), 2),
    'ci_high_a': round(np.percentile(boot_mape_xgb, 97.5), 2),
    'mean_b': round(np.mean(boot_mape_sp), 2),
    'ci_low_b': round(np.percentile(boot_mape_sp, 2.5), 2),
    'ci_high_b': round(np.percentile(boot_mape_sp, 97.5), 2),
    'mean_diff': round(np.mean(boot_diff), 2),
    'ci_low_diff': round(np.percentile(boot_diff, 2.5), 2),
    'ci_high_diff': round(np.percentile(boot_diff, 97.5), 2),
    'a_better_share': round(np.mean(np.array(boot_diff) < 0) * 100, 2),
})

# XGBoost vs Median baseline
median_train = 24.37  # из Пайплайна 4
pred_days_median = np.full(n_test, median_train)
boot_mape_median = []
boot_diff_med = []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_test, size=n_test)
    y_true = y_test_days.values[idx]
    p_xgb = pred_days_xgb[idx]
    p_med = pred_days_median[idx]
    m_xgb = mape_metric(y_true, p_xgb)
    m_med = mape_metric(y_true, p_med)
    boot_mape_median.append(m_med)
    boot_diff_med.append(m_xgb - m_med)

bootstrap_results.append({
    'comparison': 'XGBoost manual Grid vs Median-baseline',
    'metric': 'MAPE',
    'mean_a': round(np.mean(boot_mape_xgb), 2),
    'ci_low_a': round(np.percentile(boot_mape_xgb, 2.5), 2),
    'ci_high_a': round(np.percentile(boot_mape_xgb, 97.5), 2),
    'mean_b': round(np.mean(boot_mape_median), 2),
    'ci_low_b': round(np.percentile(boot_mape_median, 2.5), 2),
    'ci_high_b': round(np.percentile(boot_mape_median, 97.5), 2),
    'mean_diff': round(np.mean(boot_diff_med), 2),
    'ci_low_diff': round(np.percentile(boot_diff_med, 2.5), 2),
    'ci_high_diff': round(np.percentile(boot_diff_med, 97.5), 2),
    'a_better_share': round(np.mean(np.array(boot_diff_med) < 0) * 100, 2),
})

# XGBoost manual reg vs TabNet manual Optuna reg
print('  Регрессия: XGBoost manual Grid vs TabNet manual Optuna')
try:
    # Попробуем найти TabNet модель
    from pytorch_tabnet.tab_model import TabNetRegressor
    tabnet_path = os.path.join(MODELS_DIR, 'tabnet', 'optuna_tabnet_reg_manual.zip')
    X_test_scaled, _, _, _ = load_test('manual', 'scaled')
    tabnet_model = TabNetRegressor()
    tabnet_model.load_model(tabnet_path)
    pred_tabnet = tabnet_model.predict(X_test_scaled.values).flatten()
    pred_days_tabnet = np.expm1(pred_tabnet)

    boot_mape_tabnet = []
    boot_diff_tab = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_test, size=n_test)
        y_true = y_test_days.values[idx]
        p_xgb = pred_days_xgb[idx]
        p_tab = pred_days_tabnet[idx]
        m_xgb = mape_metric(y_true, p_xgb)
        m_tab = mape_metric(y_true, p_tab)
        boot_mape_tabnet.append(m_tab)
        boot_diff_tab.append(m_xgb - m_tab)

    bootstrap_results.append({
        'comparison': 'XGBoost manual Grid vs TabNet manual Optuna',
        'metric': 'MAPE',
        'mean_a': round(np.mean(boot_mape_xgb), 2),
        'ci_low_a': round(np.percentile(boot_mape_xgb, 2.5), 2),
        'ci_high_a': round(np.percentile(boot_mape_xgb, 97.5), 2),
        'mean_b': round(np.mean(boot_mape_tabnet), 2),
        'ci_low_b': round(np.percentile(boot_mape_tabnet, 2.5), 2),
        'ci_high_b': round(np.percentile(boot_mape_tabnet, 97.5), 2),
        'mean_diff': round(np.mean(boot_diff_tab), 2),
        'ci_low_diff': round(np.percentile(boot_diff_tab, 2.5), 2),
        'ci_high_diff': round(np.percentile(boot_diff_tab, 97.5), 2),
        'a_better_share': round(np.mean(np.array(boot_diff_tab) < 0) * 100, 2),
    })
except Exception as e:
    print(f'    TabNet bootstrap не выполнен: {e}')

# Классификация: XGBoost manual vs Majority и vs TabNet
print('  Классификация: XGBoost manual Grid vs Majority class')
xgb_clf = get_predictions_regression(os.path.join(MODELS_DIR, 'core', 'grid_xgb_clf_manual.pkl'))
proba_xgb = xgb_clf.predict_proba(X_test_proc)[:, 1]
pred_xgb_clf = (proba_xgb >= 0.5).astype(int)

_, _, y_test_clf, _ = load_test('manual', 'processed')

# Majority = 0
pred_majority = np.zeros(n_test, dtype=int)

boot_f1_xgb = []
boot_f1_majority = []
boot_diff_clf = []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_test, size=n_test)
    y_true = y_test_clf.values[idx]
    p_xgb = pred_xgb_clf[idx]
    p_maj = pred_majority[idx]
    f1_x = f1_metric(y_true, p_xgb)
    f1_m = f1_metric(y_true, p_maj)
    boot_f1_xgb.append(f1_x)
    boot_f1_majority.append(f1_m)
    boot_diff_clf.append(f1_x - f1_m)

bootstrap_results.append({
    'comparison': 'XGBoost manual Grid vs Majority class',
    'metric': 'F1',
    'mean_a': round(np.mean(boot_f1_xgb), 4),
    'ci_low_a': round(np.percentile(boot_f1_xgb, 2.5), 4),
    'ci_high_a': round(np.percentile(boot_f1_xgb, 97.5), 4),
    'mean_b': round(np.mean(boot_f1_majority), 4),
    'ci_low_b': round(np.percentile(boot_f1_majority, 2.5), 4),
    'ci_high_b': round(np.percentile(boot_f1_majority, 97.5), 4),
    'mean_diff': round(np.mean(boot_diff_clf), 4),
    'ci_low_diff': round(np.percentile(boot_diff_clf, 2.5), 4),
    'ci_high_diff': round(np.percentile(boot_diff_clf, 97.5), 4),
    'a_better_share': round(np.mean(np.array(boot_diff_clf) > 0) * 100, 2),
})

# XGBoost clf vs TabNet clf
print('  Классификация: XGBoost manual Grid vs TabNet manual Optuna')
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    tabnet_clf_path = os.path.join(MODELS_DIR, 'tabnet', 'optuna_tabnet_clf_manual.zip')
    X_test_scaled, _, _, _ = load_test('manual', 'scaled')
    tabnet_clf_model = TabNetClassifier()
    tabnet_clf_model.load_model(tabnet_clf_path)
    proba_tabnet = tabnet_clf_model.predict_proba(X_test_scaled.values)[:, 1]
    pred_tabnet_clf = (proba_tabnet >= 0.5).astype(int)

    boot_f1_tabnet = []
    boot_diff_tab_clf = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_test, size=n_test)
        y_true = y_test_clf.values[idx]
        p_xgb = pred_xgb_clf[idx]
        p_tab = pred_tabnet_clf[idx]
        f1_x = f1_metric(y_true, p_xgb)
        f1_t = f1_metric(y_true, p_tab)
        boot_f1_tabnet.append(f1_t)
        boot_diff_tab_clf.append(f1_x - f1_t)

    bootstrap_results.append({
        'comparison': 'XGBoost manual Grid vs TabNet manual Optuna',
        'metric': 'F1',
        'mean_a': round(np.mean(boot_f1_xgb), 4),
        'ci_low_a': round(np.percentile(boot_f1_xgb, 2.5), 4),
        'ci_high_a': round(np.percentile(boot_f1_xgb, 97.5), 4),
        'mean_b': round(np.mean(boot_f1_tabnet), 4),
        'ci_low_b': round(np.percentile(boot_f1_tabnet, 2.5), 4),
        'ci_high_b': round(np.percentile(boot_f1_tabnet, 97.5), 4),
        'mean_diff': round(np.mean(boot_diff_tab_clf), 4),
        'ci_low_diff': round(np.percentile(boot_diff_tab_clf, 2.5), 4),
        'ci_high_diff': round(np.percentile(boot_diff_tab_clf, 97.5), 4),
        'a_better_share': round(np.mean(np.array(boot_diff_tab_clf) > 0) * 100, 2),
    })
except Exception as e:
    print(f'    TabNet clf bootstrap не выполнен: {e}')

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_ci_results.csv'), index=False)
print(f'  Сохранено {len(bootstrap_df)} bootstrap-сравнений -> bootstrap_ci_results.csv')


# 5. SHAP-анализ для XGBoost manual

print('\nSHAP-анализ для XGBoost manual')

try:
    explainer_reg = shap.TreeExplainer(xgb_reg)
    sample_n = min(1000, len(X_test_proc))
    sample_idx = rng.integers(0, len(X_test_proc), size=sample_n)
    X_sample = X_test_proc.iloc[sample_idx]
    shap_values_reg = explainer_reg.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_reg, X_sample, plot_size=(10, 8), show=False, max_display=15)
    plt.title('SHAP summary plot, XGBoost регрессия (ручной набор)', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '27_shap_summary_reg.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('  Сохранён 27_shap_summary_reg.png')

    # Dependence plots топ-3
    mean_abs_shap = np.abs(shap_values_reg).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[::-1][:3]
    top3_features = X_sample.columns[top3_idx].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, feat in zip(axes, top3_features):
        shap.dependence_plot(feat, shap_values_reg, X_sample, ax=ax, show=False)
    plt.suptitle('SHAP dependence plot, топ-3 признака, регрессия')
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '28_shap_dependence_reg.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('  Сохранён 28_shap_dependence_reg.png')
except Exception as e:
    print(f'  Ошибка SHAP reg: {e}')

try:
    explainer_clf = shap.TreeExplainer(xgb_clf)
    X_sample_clf = X_test_proc.iloc[sample_idx]
    shap_values_clf = explainer_clf.shap_values(X_sample_clf)
    if isinstance(shap_values_clf, list) and len(shap_values_clf) == 2:
        shap_values_clf = shap_values_clf[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_clf, X_sample_clf, plot_size=(10, 8), show=False, max_display=15)
    plt.title('SHAP summary plot, XGBoost классификация (ручной набор)', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '29_shap_summary_clf.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('  Сохранён 29_shap_summary_clf.png')
except Exception as e:
    print(f'  Ошибка SHAP clf: {e}')


# 6. Анализ ошибок XGBoost manual

print('\nАнализ ошибок XGBoost manual')

# Загружаем prepared для метаданных
prepared = pd.read_csv(os.path.join(ROOT, 'Dataset', 'prepared', 'tawos_prepared.csv'))
features_manual = pd.read_csv(os.path.join(ROOT, 'Dataset', 'features_manual', 'tawos_features_manual.csv'))
test_ids = features_manual.loc[features_manual['_split'] == 'test', 'issue_id'].values
test_meta = prepared.set_index('issue_id').loc[test_ids].reset_index()

# MAPE регрессии по типам и приоритетам
err_reg = pd.DataFrame({
    'issue_type': test_meta['issue_type'].values,
    'priority': test_meta['priority'].values,
    'resolution_days': test_meta['resolution_days'].values,
    'pred_days': pred_days_xgb,
    'abs_pct_err': np.abs(test_meta['resolution_days'].values - pred_days_xgb) / (test_meta['resolution_days'].values + 1e-8),
})

err_by_type = err_reg.groupby('issue_type')['abs_pct_err'].agg(['mean', 'count']).reset_index()
err_by_type['MAPE_pct'] = (err_by_type['mean'] * 100).round(2)
err_by_type = err_by_type[['issue_type', 'count', 'MAPE_pct']].rename(columns={'count': 'n'})
err_by_type.to_csv(os.path.join(RESULTS_DIR, 'errors_by_type_regression.csv'), index=False)
print(f'  Регрессия по типам: {len(err_by_type)} строк')

err_by_priority = err_reg.groupby('priority')['abs_pct_err'].agg(['mean', 'count']).reset_index()
err_by_priority['MAPE_pct'] = (err_by_priority['mean'] * 100).round(2)
err_by_priority = err_by_priority[['priority', 'count', 'MAPE_pct']].rename(columns={'count': 'n'})
err_by_priority.to_csv(os.path.join(RESULTS_DIR, 'errors_by_priority_regression.csv'), index=False)

# MAPE по диапазонам resolution_days
err_reg['duration_bin'] = pd.cut(err_reg['resolution_days'],
    bins=[0, 7, 30, 90, 365],
    labels=['1-7 дней', '7-30 дней', '30-90 дней', '90-365 дней'])
err_by_bin = err_reg.groupby('duration_bin')['abs_pct_err'].agg(['mean', 'count']).reset_index()
err_by_bin['MAPE_pct'] = (err_by_bin['mean'] * 100).round(2)
err_by_bin = err_by_bin[['duration_bin', 'count', 'MAPE_pct']].rename(columns={'count': 'n'})
err_by_bin.to_csv(os.path.join(RESULTS_DIR, 'errors_by_duration_regression.csv'), index=False)

# График: ошибка по типам и приоритетам
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(err_by_type['issue_type'], err_by_type['MAPE_pct'], color=COLOR_BAR, edgecolor='black')
axes[0].set_title('MAPE по типам задач')
axes[0].set_xlabel('issue_type')
axes[0].set_ylabel('MAPE, %')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

axes[1].bar(err_by_priority['priority'], err_by_priority['MAPE_pct'], color=COLOR_BAR, edgecolor='black')
axes[1].set_title('MAPE по приоритетам')
axes[1].set_xlabel('priority')
axes[1].set_ylabel('MAPE, %')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

axes[2].bar([str(b) for b in err_by_bin['duration_bin']], err_by_bin['MAPE_pct'], color=COLOR_BAR, edgecolor='black')
axes[2].set_title('MAPE по диапазонам длительности')
axes[2].set_xlabel('Диапазон')
axes[2].set_ylabel('MAPE, %')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('Анализ ошибок XGBoost manual Grid (регрессия)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '30_error_analysis_regression.png'))
plt.close()
print('  Сохранён 30_error_analysis_regression.png')

# Анализ ошибок классификации (F1 по типам)
err_clf = pd.DataFrame({
    'issue_type': test_meta['issue_type'].values,
    'priority': test_meta['priority'].values,
    'resolution_days': test_meta['resolution_days'].values,
    'true': y_test_clf.values,
    'pred': pred_xgb_clf,
})

from sklearn.metrics import f1_score
def f1_per_group(g):
    if g['true'].sum() == 0 or g['pred'].sum() == 0:
        return np.nan
    return f1_score(g['true'], g['pred'], zero_division=0)

f1_by_type = err_clf.groupby('issue_type').apply(f1_per_group).reset_index()
f1_by_type.columns = ['issue_type', 'F1']
f1_by_type['n'] = err_clf.groupby('issue_type').size().values
f1_by_type.to_csv(os.path.join(RESULTS_DIR, 'errors_by_type_classification.csv'), index=False)

f1_by_priority = err_clf.groupby('priority').apply(f1_per_group).reset_index()
f1_by_priority.columns = ['priority', 'F1']
f1_by_priority['n'] = err_clf.groupby('priority').size().values
f1_by_priority.to_csv(os.path.join(RESULTS_DIR, 'errors_by_priority_classification.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(f1_by_type['issue_type'], f1_by_type['F1'], color=COLOR_BAR, edgecolor='black')
axes[0].set_title('F1 по типам задач')
axes[0].set_xlabel('issue_type')
axes[0].set_ylabel('F1-score')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

axes[1].bar(f1_by_priority['priority'], f1_by_priority['F1'], color=COLOR_BAR, edgecolor='black')
axes[1].set_title('F1 по приоритетам')
axes[1].set_xlabel('priority')
axes[1].set_ylabel('F1-score')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('Анализ ошибок XGBoost manual Grid (классификация)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '31_error_analysis_classification.png'))
plt.close()
print('  Сохранён 31_error_analysis_classification.png')

print('\nПайплайн 8 завершён')

# Итоговая сводка
print('\nСохранены файлы:')
print(f'  comparison_table_regression.csv')
print(f'  comparison_table_classification.csv')
print(f'  comparison_table.csv')
print(f'  wilcoxon_results.csv')
print(f'  bootstrap_ci_results.csv')
print(f'  errors_by_type_regression.csv, errors_by_priority_regression.csv, errors_by_duration_regression.csv')
print(f'  errors_by_type_classification.csv, errors_by_priority_classification.csv')
print(f'  shap_plots/27-29_*.png')
print(f'  plots/analysis/30-31_*.png')