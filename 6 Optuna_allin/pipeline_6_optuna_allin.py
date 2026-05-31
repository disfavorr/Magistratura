import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import NSGAIISampler

from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, recall_score, precision_score,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r'D:\blablabla\Magistr'
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
MODELS_DIR = os.path.join(ROOT, 'Dataset', 'models', 'core')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'plots', 'optuna')
STUDIES_DIR = os.path.join(RESULTS_DIR, 'optuna_studies')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STUDIES_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_PRIMARY = '#3a76b5'
COLOR_ACCENT = '#a83232'
COLOR_PARETO = '#d7191c'

RANDOM_STATE = 42
N_FOLDS = 10
N_TRIALS = 80
NSGA2_POPULATION = 10


# Единые функции early stopping для XGBoost

def fit_xgb_regressor_es(X_train, y_train, params, val_size=0.15):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=RANDOM_STATE,
    )
    model = XGBRegressor(
        n_estimators=500,
        eval_metric='mae',
        early_stopping_rounds=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    return model


def fit_xgb_classifier_es(X_train, y_train, params, val_size=0.15):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=RANDOM_STATE,
        stratify=y_train,
    )
    model = XGBClassifier(
        n_estimators=500,
        eval_metric='auc',
        early_stopping_rounds=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    return model


# Метрики

def regression_metrics(y_true_days, y_pred_days):
    eps = 1e-8
    abs_pct_err = np.abs((y_true_days - y_pred_days) / (y_true_days + eps))
    return {
        'MAPE_pct': round(np.mean(abs_pct_err) * 100, 2),
        'MMRE': round(np.mean(abs_pct_err), 4),
        'PRED25_pct': round(np.mean(abs_pct_err < 0.25) * 100, 2),
        'MAE_days': round(mean_absolute_error(y_true_days, y_pred_days), 2),
        'RMSE_days': round(np.sqrt(mean_squared_error(y_true_days, y_pred_days)), 2),
        'R2': round(r2_score(y_true_days, y_pred_days), 4),
    }


def classification_metrics(y_true, y_pred, y_proba):
    return {
        'F1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC_AUC': round(roc_auc_score(y_true, y_proba), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
    }


# Загрузка данных

def load_feature_set(name, version):
    base = os.path.join(SPLITS_DIR, name, version)
    X_train = pd.read_csv(os.path.join(base, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(base, 'X_test.csv'))
    y_reg_train = pd.read_csv(os.path.join(base, 'y_reg_train.csv')).iloc[:, 0]
    y_reg_test = pd.read_csv(os.path.join(base, 'y_reg_test.csv')).iloc[:, 0]
    y_clf_train = pd.read_csv(os.path.join(base, 'y_clf_train.csv')).iloc[:, 0]
    y_clf_test = pd.read_csv(os.path.join(base, 'y_clf_test.csv')).iloc[:, 0]
    y_reg_train_days = pd.read_csv(os.path.join(base, 'y_reg_train_days.csv')).iloc[:, 0]
    y_reg_test_days = pd.read_csv(os.path.join(base, 'y_reg_test_days.csv')).iloc[:, 0]
    return (X_train, X_test, y_reg_train, y_reg_test,
            y_clf_train, y_clf_test, y_reg_train_days, y_reg_test_days)


def load_test_subgroup_masks(feature_set):
    raw_test = pd.read_csv(os.path.join(SPLITS_DIR, feature_set, 'raw', 'X_test.csv'))
    has_hist = raw_test['has_history'].values
    prepared = pd.read_csv(os.path.join(ROOT, 'Dataset', 'prepared', 'tawos_prepared.csv'))
    features = pd.read_csv(os.path.join(ROOT, 'Dataset', 'features_manual', 'tawos_features_manual.csv'))
    test_ids = features.loc[features['_split'] == 'test', 'issue_id'].values
    test_assignees = prepared.set_index('issue_id').loc[test_ids, 'assignee_id'].values

    mask_h1 = has_hist == 1
    mask_h0_new = (has_hist == 0) & (~np.isnan(test_assignees))
    mask_h0_nan = (has_hist == 0) & (np.isnan(test_assignees))
    return mask_h1, mask_h0_new, mask_h0_nan


def get_cv_splitter(task):
    if task == 'clf':
        return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# Suggest функции по границам Grid v2

def suggest_ridge(trial):
    return {'alpha': trial.suggest_float('alpha', 1e-1, 1e5, log=True)}


def suggest_logistic(trial):
    return {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
    }


def suggest_knn(trial):
    return {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 100),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'p': trial.suggest_categorical('p', [1, 2]),
    }


def suggest_rf(trial, task):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'max_depth': trial.suggest_categorical('max_depth', [20, None]),
    }
    if task == 'reg':
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 5)
    else:
        params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 5, 20)
        params['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
    return params


def suggest_xgb(trial, task):
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 10, 50, log=True),
    }
    if task == 'reg':
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.7, 1.0)
    else:
        params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1.0, 1.55)
    return params


# Универсальная оценка CV

def evaluate_cv(model_name, task, X_train, y_train, params):
    splitter = get_cv_splitter(task)
    if task == 'clf':
        split_iter = list(splitter.split(X_train, y_train))
    else:
        split_iter = list(splitter.split(X_train))

    fold_scores = []
    fold_times = []

    for fold_train_idx, fold_val_idx in split_iter:
        X_fold_tr = X_train.iloc[fold_train_idx]
        X_fold_val = X_train.iloc[fold_val_idx]
        y_fold_tr = y_train.iloc[fold_train_idx]
        y_fold_val = y_train.iloc[fold_val_idx]

        t_start = time.time()

        if model_name == 'xgb':
            if task == 'reg':
                model = fit_xgb_regressor_es(X_fold_tr, y_fold_tr, params)
            else:
                model = fit_xgb_classifier_es(X_fold_tr, y_fold_tr, params)
        else:
            if model_name == 'ridge':
                model = Ridge(**params)
            elif model_name == 'logistic':
                model = LogisticRegression(solver='saga', max_iter=5000, random_state=RANDOM_STATE, **params)
            elif model_name == 'knn':
                if task == 'reg':
                    model = KNeighborsRegressor(n_jobs=-1, **params)
                else:
                    model = KNeighborsClassifier(n_jobs=-1, **params)
            elif model_name == 'rf':
                if task == 'reg':
                    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
                else:
                    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
            model.fit(X_fold_tr, y_fold_tr)

        fit_time = time.time() - t_start
        fold_times.append(fit_time)

        if task == 'reg':
            pred = model.predict(X_fold_val)
            y_true_d = np.expm1(y_fold_val.values)
            y_pred_d = np.expm1(pred)
            eps = 1e-8
            score = -np.mean(np.abs((y_true_d - y_pred_d) / (y_true_d + eps))) * 100
        else:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_fold_val)[:, 1]
            else:
                scores = model.decision_function(X_fold_val)
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            pred = (proba >= 0.5).astype(int)
            score = f1_score(y_fold_val, pred, zero_division=0)

        fold_scores.append(score)

    return float(np.mean(fold_scores)), float(np.mean(fold_times)), fold_scores


# Один эксперимент

def run_optuna_experiment(model_name, task, feature_set):
    print(f'\n  Эксперимент: {model_name} | {task} | {feature_set}')
    t0 = time.time()

    if model_name in ['ridge', 'logistic', 'knn']:
        version = 'scaled'
    else:
        version = 'processed'

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, _, y_reg_test_days = load_feature_set(feature_set, version)

    if task == 'reg':
        y_train, y_test = y_reg_train, y_reg_test
    else:
        y_train, y_test = y_clf_train, y_clf_test

    def objective(trial):
        if model_name == 'ridge':
            params = suggest_ridge(trial)
        elif model_name == 'logistic':
            params = suggest_logistic(trial)
        elif model_name == 'knn':
            params = suggest_knn(trial)
        elif model_name == 'rf':
            params = suggest_rf(trial, task)
        elif model_name == 'xgb':
            params = suggest_xgb(trial, task)

        try:
            mean_score, mean_time, fold_scores = evaluate_cv(model_name, task, X_train, y_train, params)
        except Exception:
            return -1e9, 1e9

        trial.set_user_attr('fold_scores', fold_scores)
        trial.set_user_attr('mean_score', mean_score)

        return mean_score, mean_time

    storage_path = f'sqlite:///{os.path.join(STUDIES_DIR, f"{model_name}_{task}_{feature_set}.db")}'
    study_name = f'{model_name}_{task}_{feature_set}'

    sampler = NSGAIISampler(seed=RANDOM_STATE, population_size=NSGA2_POPULATION)
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=sampler,
        storage=storage_path,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    pareto_trials = study.best_trials
    print(f'    Pareto-фронт: {len(pareto_trials)} точек')

    # С Pareto-фронта берём точку с лучшим качеством (правило фиксировано заранее)
    best_pareto_trial = max(pareto_trials, key=lambda t: t.values[0])
    best_params = best_pareto_trial.params
    best_score = best_pareto_trial.values[0]
    best_time = best_pareto_trial.values[1]

    cv_fold_scores = best_pareto_trial.user_attrs.get('fold_scores')
    # Fallback: если user_attrs не восстановились из SQLite, пересчитываем
    if cv_fold_scores is None:
        print('    user_attrs.fold_scores отсутствует, пересчёт CV')
        _, _, cv_fold_scores = evaluate_cv(model_name, task, X_train, y_train, best_params)

    print(f'    Лучшая точка: качество={best_score:.4f} | время фолда={best_time:.2f}с')

    # Финальная модель на полном train
    if model_name == 'xgb':
        if task == 'reg':
            best_model = fit_xgb_regressor_es(X_train, y_train, best_params)
        else:
            best_model = fit_xgb_classifier_es(X_train, y_train, best_params)
    elif model_name == 'ridge':
        best_model = Ridge(**best_params)
        best_model.fit(X_train, y_train)
    elif model_name == 'logistic':
        best_model = LogisticRegression(solver='saga', max_iter=5000, random_state=RANDOM_STATE, **best_params)
        best_model.fit(X_train, y_train)
    elif model_name == 'knn':
        if task == 'reg':
            best_model = KNeighborsRegressor(n_jobs=-1, **best_params)
        else:
            best_model = KNeighborsClassifier(n_jobs=-1, **best_params)
        best_model.fit(X_train, y_train)
    elif model_name == 'rf':
        if task == 'reg':
            best_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        else:
            best_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
        best_model.fit(X_train, y_train)

    # Метрики на test
    if task == 'reg':
        pred_log = best_model.predict(X_test)
        pred_days = np.expm1(pred_log)
        y_true_days = y_reg_test_days.values
        metrics = regression_metrics(y_true_days, pred_days)
    else:
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(X_test)[:, 1]
        else:
            scores = best_model.decision_function(X_test)
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        pred = (proba >= 0.5).astype(int)
        metrics = classification_metrics(y_test.values, pred, proba)

    # Срезы
    mask_h1, mask_h0_new, mask_h0_nan = load_test_subgroup_masks(feature_set)
    sub_metrics = {}
    for sub_name, mask in [('has_history_1', mask_h1),
                            ('has_history_0_new', mask_h0_new),
                            ('has_history_0_nan', mask_h0_nan)]:
        if mask.sum() == 0:
            continue
        if task == 'reg':
            sub_m = regression_metrics(y_true_days[mask], pred_days[mask])
        else:
            if mask.sum() < 5:
                continue
            sub_m = classification_metrics(y_test.values[mask], pred[mask], proba[mask])
        for k, v in sub_m.items():
            sub_metrics[f'{sub_name}_{k}'] = v

    # Сохранение модели
    model_path = os.path.join(MODELS_DIR, f'optuna_{model_name}_{task}_{feature_set}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'best_params': best_params,
            'cv_fold_scores': cv_fold_scores,
            'best_cv_score': best_score,
            'best_fit_time': best_time,
            'pareto_size': len(pareto_trials),
        }, f)

    elapsed = time.time() - t0

    result_row = {
        'method': 'Optuna_NSGA2',
        'model': model_name,
        'task': task,
        'feature_set': feature_set,
        'best_params': str(best_params),
        'best_cv_score': round(best_score, 4),
        'best_fit_time_sec': round(best_time, 2),
        'pareto_size': len(pareto_trials),
        'elapsed_sec': round(elapsed, 1),
    }
    result_row.update(metrics)
    result_row.update(sub_metrics)

    main_metric_value = list(metrics.values())[0]
    print(f'    Время: {elapsed:.1f} сек | best_cv: {best_score:.4f} | test main: {main_metric_value}')

    return result_row, cv_fold_scores, study


# План экспериментов

experiments = []
for task in ['reg', 'clf']:
    base_lin = 'ridge' if task == 'reg' else 'logistic'
    auto_set = 'auto_regression' if task == 'reg' else 'auto_classification'
    for model_name in [base_lin, 'knn', 'rf', 'xgb']:
        for feature_set in ['manual', auto_set]:
            experiments.append((model_name, task, feature_set))

print(f'Всего запланировано экспериментов: {len(experiments)}')
print(f'Trials на эксперимент: {N_TRIALS}')
print(f'NSGA-II population size: {NSGA2_POPULATION}')
print('Пайплайн 6 запущен')

results = []
all_cv_scores = {}
studies = {}

for i, (model_name, task, feature_set) in enumerate(experiments, 1):
    print(f'\n[{i}/{len(experiments)}]')
    try:
        row, fold_scores, study = run_optuna_experiment(model_name, task, feature_set)
        results.append(row)
        all_cv_scores[f'{model_name}_{task}_{feature_set}'] = fold_scores
        studies[f'{model_name}_{task}_{feature_set}'] = study
    except Exception as e:
        print(f'    ОШИБКА: {type(e).__name__}: {e}')
        results.append({
            'method': 'Optuna_NSGA2',
            'model': model_name,
            'task': task,
            'feature_set': feature_set,
            'error': str(e),
        })

results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, 'optuna_results_core.csv')
results_df.to_csv(results_path, index=False)
print(f'\nСохранены результаты: {results_path}')

cv_path = os.path.join(RESULTS_DIR, 'optuna_cv_scores.pkl')
with open(cv_path, 'wb') as f:
    pickle.dump(all_cv_scores, f)
print(f'Сохранены CV-скоры: {cv_path}')


# Графики

print('\nПостроение графиков')

# 19. Pareto-фронт XGBoost регрессии (manual)

try:
    study = studies.get('xgb_reg_manual')
    if study is not None:
        all_trials = [t for t in study.trials if t.values is not None and len(t.values) == 2]
        scores = [t.values[0] for t in all_trials]
        times = [t.values[1] for t in all_trials]
        pareto = study.best_trials
        pareto_scores = [t.values[0] for t in pareto]
        pareto_times = [t.values[1] for t in pareto]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(times, scores, alpha=0.4, s=30, color=COLOR_PRIMARY, label='Все trials')
        ax.scatter(pareto_times, pareto_scores, alpha=0.95, s=80,
                   color=COLOR_PARETO, edgecolor='black', label=f'Pareto-фронт ({len(pareto)} точек)')
        ax.set_xlabel('Среднее время обучения на фолде, секунды')
        ax.set_ylabel('CV качество (отрицательный MAPE, %)')
        ax.set_title('Pareto-фронт NSGA-II, XGBoost регрессия, ручной набор')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '19_optuna_pareto_xgb_reg.png'))
        plt.close()
        print('  Сохранён 19_optuna_pareto_xgb_reg.png')
except Exception as e:
    print(f'  Ошибка Pareto reg: {e}')


# 20. Pareto-фронт XGBoost классификации (manual)

try:
    study = studies.get('xgb_clf_manual')
    if study is not None:
        all_trials = [t for t in study.trials if t.values is not None and len(t.values) == 2]
        scores = [t.values[0] for t in all_trials]
        times = [t.values[1] for t in all_trials]
        pareto = study.best_trials
        pareto_scores = [t.values[0] for t in pareto]
        pareto_times = [t.values[1] for t in pareto]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(times, scores, alpha=0.4, s=30, color=COLOR_PRIMARY, label='Все trials')
        ax.scatter(pareto_times, pareto_scores, alpha=0.95, s=80,
                   color=COLOR_PARETO, edgecolor='black', label=f'Pareto-фронт ({len(pareto)} точек)')
        ax.set_xlabel('Среднее время обучения на фолде, секунды')
        ax.set_ylabel('CV F1-score')
        ax.set_title('Pareto-фронт NSGA-II, XGBoost классификация, ручной набор')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '20_optuna_pareto_xgb_clf.png'))
        plt.close()
        print('  Сохранён 20_optuna_pareto_xgb_clf.png')
except Exception as e:
    print(f'  Ошибка Pareto clf: {e}')


# 21. Кривая сходимости XGBoost

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, study_key, ylabel, title in [
        (axes[0], 'xgb_reg_manual', 'Best CV score (отриц. MAPE, %)', 'XGBoost регрессия (manual)'),
        (axes[1], 'xgb_clf_manual', 'Best CV F1', 'XGBoost классификация (manual)'),
    ]:
        study = studies.get(study_key)
        if study is None:
            continue
        all_trials = [t for t in study.trials if t.values is not None]
        if len(all_trials) == 0:
            continue
        scores_per_trial = [t.values[0] for t in all_trials]
        running_best = np.maximum.accumulate(scores_per_trial)
        ax.plot(range(1, len(running_best) + 1), running_best,
                color=COLOR_ACCENT, linewidth=2, label='Best (running)')
        ax.scatter(range(1, len(scores_per_trial) + 1), scores_per_trial,
                   alpha=0.3, s=20, color=COLOR_PRIMARY, label='Trial scores')
        ax.set_xlabel('Trial')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.suptitle('Сходимость Optuna: лучшее CV-качество по итерациям')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '21_optuna_convergence_xgb.png'))
    plt.close()
    print('  Сохранён 21_optuna_convergence_xgb.png')
except Exception as e:
    print(f'  Ошибка сходимости: {e}')


# 22. Важность гиперпараметров XGBoost (Optuna fANOVA)

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, study_key, title in [
        (axes[0], 'xgb_reg_manual', 'XGBoost регрессия (manual)'),
        (axes[1], 'xgb_clf_manual', 'XGBoost классификация (manual)'),
    ]:
        study = studies.get(study_key)
        if study is None:
            continue
        try:
            importances = optuna.importance.get_param_importances(
                study, target=lambda t: t.values[0],
            )
            params = list(importances.keys())
            values = list(importances.values())
            y_pos = np.arange(len(params))
            ax.barh(y_pos, values, color=COLOR_PRIMARY, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('Важность')
            ax.set_title(title)
            ax.invert_yaxis()
        except Exception as e_inner:
            ax.text(0.5, 0.5, f'Не удалось вычислить:\n{e_inner}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    plt.suptitle('Важность гиперпараметров (Optuna fANOVA)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '22_optuna_param_importance_xgb.png'))
    plt.close()
    print('  Сохранён 22_optuna_param_importance_xgb.png')
except Exception as e:
    print(f'  Ошибка param importance: {e}')

print('\nПайплайн 6 завершён')