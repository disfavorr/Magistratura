import os
import time
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import optuna
from optuna.samplers import NSGAIISampler

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, recall_score, precision_score,
)

from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = r'D:\blablabla\Magistr'
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
MODELS_DIR = os.path.join(ROOT, 'Dataset', 'models', 'tabnet')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'plots', 'tabnet')
STUDIES_DIR = os.path.join(RESULTS_DIR, 'optuna_studies_tabnet')

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

COLOR_TRAIN = '#3a76b5'
COLOR_VAL = '#a83232'
COLOR_PRIMARY = '#3a76b5'
COLOR_PARETO = '#d7191c'

RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS_OPTUNA = 50
NSGA2_POPULATION = 10
DEVICE = 'cpu'
MAX_EPOCHS = 80
PATIENCE = 15


# Фабрика TabNet моделей

def make_tabnet_regressor(params):
    return TabNetRegressor(
        n_d=params['n_d'],
        n_a=params['n_d'],
        n_steps=params['n_steps'],
        lambda_sparse=params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': params.get('learning_rate', 2e-2)},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size': 20, 'gamma': 0.9},
        seed=RANDOM_STATE,
        verbose=0,
        device_name=DEVICE,
    )


def make_tabnet_classifier(params):
    return TabNetClassifier(
        n_d=params['n_d'],
        n_a=params['n_d'],
        n_steps=params['n_steps'],
        lambda_sparse=params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': params.get('learning_rate', 2e-2)},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={'step_size': 20, 'gamma': 0.9},
        seed=RANDOM_STATE,
        verbose=0,
        device_name=DEVICE,
    )


# Обучение TabNet с early stopping (внутренний val 15%)

def fit_tabnet_regressor(X_train, y_train, params, val_size=0.15):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=RANDOM_STATE,
    )
    model = make_tabnet_regressor(params)
    model.fit(
        X_tr.values if hasattr(X_tr, 'values') else X_tr,
        y_tr.values.reshape(-1, 1) if hasattr(y_tr, 'values') else y_tr.reshape(-1, 1),
        eval_set=[
            (X_tr.values if hasattr(X_tr, 'values') else X_tr,
             y_tr.values.reshape(-1, 1) if hasattr(y_tr, 'values') else y_tr.reshape(-1, 1)),
            (X_val.values if hasattr(X_val, 'values') else X_val,
             y_val.values.reshape(-1, 1) if hasattr(y_val, 'values') else y_val.reshape(-1, 1)),
        ],
        eval_name=['train', 'val'],
        eval_metric=['mae'],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    return model


def fit_tabnet_classifier(X_train, y_train, params, val_size=0.15):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=RANDOM_STATE,
        stratify=y_train,
    )
    model = make_tabnet_classifier(params)
    model.fit(
        X_tr.values if hasattr(X_tr, 'values') else X_tr,
        y_tr.values if hasattr(y_tr, 'values') else y_tr,
        eval_set=[
            (X_tr.values if hasattr(X_tr, 'values') else X_tr,
             y_tr.values if hasattr(y_tr, 'values') else y_tr),
            (X_val.values if hasattr(X_val, 'values') else X_val,
             y_val.values if hasattr(y_val, 'values') else y_val),
        ],
        eval_name=['train', 'val'],
        eval_metric=['auc'],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=1,
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


# Загрузка scaled-данных (TabNet требует нормализации)

def load_feature_set_scaled(name):
    base = os.path.join(SPLITS_DIR, name, 'scaled')
    X_train = pd.read_csv(os.path.join(base, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(base, 'X_test.csv'))
    y_reg_train = pd.read_csv(os.path.join(base, 'y_reg_train.csv')).iloc[:, 0]
    y_reg_test = pd.read_csv(os.path.join(base, 'y_reg_test.csv')).iloc[:, 0]
    y_clf_train = pd.read_csv(os.path.join(base, 'y_clf_train.csv')).iloc[:, 0]
    y_clf_test = pd.read_csv(os.path.join(base, 'y_clf_test.csv')).iloc[:, 0]
    y_reg_test_days = pd.read_csv(os.path.join(base, 'y_reg_test_days.csv')).iloc[:, 0]
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_reg_test_days


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


# Universal CV evaluation

def evaluate_cv(task, X_train, y_train, params):
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

        if task == 'reg':
            model = fit_tabnet_regressor(X_fold_tr, y_fold_tr, params)
            pred = model.predict(X_fold_val.values).flatten()
            y_true_d = np.expm1(y_fold_val.values)
            y_pred_d = np.expm1(pred)
            eps = 1e-8
            score = -np.mean(np.abs((y_true_d - y_pred_d) / (y_true_d + eps))) * 100
        else:
            model = fit_tabnet_classifier(X_fold_tr, y_fold_tr, params)
            proba = model.predict_proba(X_fold_val.values)[:, 1]
            pred = (proba >= 0.5).astype(int)
            score = f1_score(y_fold_val, pred, zero_division=0)

        fold_times.append(time.time() - t_start)
        fold_scores.append(score)

    return float(np.mean(fold_scores)), float(np.mean(fold_times)), fold_scores


# GridSearch для TabNet

def grid_search_tabnet(task, X_train, y_train, param_grid):
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in product(*values_lists)]

    best_score = -np.inf
    best_params = None
    best_fold_scores = None

    for i, params in enumerate(combos):
        print(f'    Grid {i+1}/{len(combos)}: {params}')
        try:
            mean_score, mean_time, fold_scores = evaluate_cv(task, X_train, y_train, params)
            print(f'      score={mean_score:.4f}, time={mean_time:.1f}с')
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_fold_scores = fold_scores
        except Exception as e:
            print(f'      ОШИБКА: {e}')

    return best_params, best_score, best_fold_scores


# Optuna для TabNet

def suggest_tabnet(trial):
    return {
        'n_d': trial.suggest_categorical('n_d', [16, 24, 32]),
        'n_steps': trial.suggest_int('n_steps', 3, 5),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-3, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-2, log=True),
    }


def optuna_search_tabnet(task, X_train, y_train, feature_set):
    storage_path = f'sqlite:///{os.path.join(STUDIES_DIR, f"tabnet_{task}_{feature_set}.db")}'

    def objective(trial):
        params = suggest_tabnet(trial)
        try:
            mean_score, mean_time, fold_scores = evaluate_cv(task, X_train, y_train, params)
        except Exception:
            return -1e9, 1e9
        trial.set_user_attr('fold_scores', fold_scores)
        return mean_score, mean_time

    sampler = NSGAIISampler(seed=RANDOM_STATE, population_size=NSGA2_POPULATION)
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=sampler,
        storage=storage_path,
        study_name=f'tabnet_{task}_{feature_set}',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=False)

    pareto_trials = study.best_trials
    print(f'    Pareto-фронт: {len(pareto_trials)} точек')

    best_pareto_trial = max(pareto_trials, key=lambda t: t.values[0])
    best_params = best_pareto_trial.params
    best_score = best_pareto_trial.values[0]
    best_time = best_pareto_trial.values[1]
    cv_fold_scores = best_pareto_trial.user_attrs.get('fold_scores')

    if cv_fold_scores is None:
        print('    user_attrs.fold_scores отсутствует, пересчёт CV')
        _, _, cv_fold_scores = evaluate_cv(task, X_train, y_train, best_params)

    return best_params, best_score, best_time, cv_fold_scores, study


# Один эксперимент GridSearch

def run_grid_experiment(task, feature_set):
    print(f'\n  GridSearch TabNet | {task} | {feature_set}')
    t0 = time.time()

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_reg_test_days = load_feature_set_scaled(feature_set)

    if task == 'reg':
        y_train, y_test = y_reg_train, y_reg_test
    else:
        y_train, y_test = y_clf_train, y_clf_test

    grid = {
        'n_d': [16, 32],
        'n_steps': [3, 5],
        'lambda_sparse': [1e-4, 1e-3],
        'learning_rate': [2e-2],
    }

    best_params, best_score, fold_scores = grid_search_tabnet(task, X_train, y_train, grid)
    print(f'    Лучшие параметры: {best_params}')
    print(f'    Best CV: {best_score:.4f}')

    # Финальная модель на полном train
    if task == 'reg':
        best_model = fit_tabnet_regressor(X_train, y_train, best_params)
        pred = best_model.predict(X_test.values).flatten()
        pred_days = np.expm1(pred)
        y_true_days = y_reg_test_days.values
        metrics = regression_metrics(y_true_days, pred_days)
    else:
        best_model = fit_tabnet_classifier(X_train, y_train, best_params)
        proba = best_model.predict_proba(X_test.values)[:, 1]
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

    # Сохранение
    save_path = os.path.join(MODELS_DIR, f'grid_tabnet_{task}_{feature_set}')
    best_model.save_model(save_path)

    pkl_path = os.path.join(MODELS_DIR, f'grid_tabnet_{task}_{feature_set}_meta.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'best_params': best_params,
            'cv_fold_scores': fold_scores,
            'best_cv_score': best_score,
            'model_path': save_path + '.zip',
        }, f)

    elapsed = time.time() - t0
    result_row = {
        'method': 'GridSearch',
        'model': 'tabnet',
        'task': task,
        'feature_set': feature_set,
        'best_params': str(best_params),
        'best_cv_score': round(best_score, 4),
        'elapsed_sec': round(elapsed, 1),
    }
    result_row.update(metrics)
    result_row.update(sub_metrics)

    main_metric_value = list(metrics.values())[0]
    print(f'    Время: {elapsed:.1f} сек | test main: {main_metric_value}')
    return result_row, fold_scores, best_model


# Один эксперимент Optuna

def run_optuna_experiment(task, feature_set):
    print(f'\n  Optuna TabNet | {task} | {feature_set}')
    t0 = time.time()

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_reg_test_days = load_feature_set_scaled(feature_set)

    if task == 'reg':
        y_train, y_test = y_reg_train, y_reg_test
    else:
        y_train, y_test = y_clf_train, y_clf_test

    best_params, best_score, best_time, fold_scores, study = optuna_search_tabnet(task, X_train, y_train, feature_set)
    print(f'    Лучшие параметры: {best_params}')
    print(f'    Best CV: {best_score:.4f}, время фолда: {best_time:.2f}с')

    # Финальная модель
    if task == 'reg':
        best_model = fit_tabnet_regressor(X_train, y_train, best_params)
        pred = best_model.predict(X_test.values).flatten()
        pred_days = np.expm1(pred)
        y_true_days = y_reg_test_days.values
        metrics = regression_metrics(y_true_days, pred_days)
    else:
        best_model = fit_tabnet_classifier(X_train, y_train, best_params)
        proba = best_model.predict_proba(X_test.values)[:, 1]
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

    save_path = os.path.join(MODELS_DIR, f'optuna_tabnet_{task}_{feature_set}')
    best_model.save_model(save_path)

    pkl_path = os.path.join(MODELS_DIR, f'optuna_tabnet_{task}_{feature_set}_meta.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'best_params': best_params,
            'cv_fold_scores': fold_scores,
            'best_cv_score': best_score,
            'best_fit_time': best_time,
            'pareto_size': len(study.best_trials),
            'model_path': save_path + '.zip',
        }, f)

    elapsed = time.time() - t0
    result_row = {
        'method': 'Optuna_NSGA2',
        'model': 'tabnet',
        'task': task,
        'feature_set': feature_set,
        'best_params': str(best_params),
        'best_cv_score': round(best_score, 4),
        'best_fit_time_sec': round(best_time, 2),
        'pareto_size': len(study.best_trials),
        'elapsed_sec': round(elapsed, 1),
    }
    result_row.update(metrics)
    result_row.update(sub_metrics)

    main_metric_value = list(metrics.values())[0]
    print(f'    Время: {elapsed:.1f} сек | test main: {main_metric_value}')
    return result_row, fold_scores, best_model, study


# Запуск

experiments = []
for task in ['reg', 'clf']:
    auto_set = 'auto_regression' if task == 'reg' else 'auto_classification'
    for feature_set in ['manual', auto_set]:
        experiments.append((task, feature_set))

print(f'TabNet: {N_FOLDS}-fold CV, CPU')
print(f'Эксперименты: {len(experiments)} × 2 метода (Grid + Optuna) = {len(experiments) * 2}')
print('Пайплайн 7 запущен')

results = []
all_cv_scores = {}
saved_models = {}
saved_studies = {}

for i, (task, feature_set) in enumerate(experiments, 1):
    print(f'\n[{i}/{len(experiments)} наборов]')

    # Grid
    try:
        row, fold_scores, model = run_grid_experiment(task, feature_set)
        results.append(row)
        all_cv_scores[f'grid_tabnet_{task}_{feature_set}'] = fold_scores
        saved_models[f'grid_{task}_{feature_set}'] = model
    except Exception as e:
        print(f'    ОШИБКА Grid: {type(e).__name__}: {e}')
        results.append({'method': 'GridSearch', 'model': 'tabnet', 'task': task,
                        'feature_set': feature_set, 'error': str(e)})

    # Optuna
    try:
        row, fold_scores, model, study = run_optuna_experiment(task, feature_set)
        results.append(row)
        all_cv_scores[f'optuna_tabnet_{task}_{feature_set}'] = fold_scores
        saved_models[f'optuna_{task}_{feature_set}'] = model
        saved_studies[f'optuna_{task}_{feature_set}'] = study
    except Exception as e:
        print(f'    ОШИБКА Optuna: {type(e).__name__}: {e}')
        results.append({'method': 'Optuna_NSGA2', 'model': 'tabnet', 'task': task,
                        'feature_set': feature_set, 'error': str(e)})

results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, 'tabnet_results.csv')
results_df.to_csv(results_path, index=False)
print(f'\nСохранены результаты: {results_path}')

cv_path = os.path.join(RESULTS_DIR, 'tabnet_cv_scores.pkl')
with open(cv_path, 'wb') as f:
    pickle.dump(all_cv_scores, f)
print(f'Сохранены CV-скоры: {cv_path}')


# Графики

print('\nПостроение графиков')

# 23. Train vs val MAE TabNet регрессии (manual, лучшая модель)

try:
    model = saved_models.get('grid_reg_manual') or saved_models.get('optuna_reg_manual')
    if model is not None and hasattr(model, 'history'):
        history = model.history
        train_mae = history.get('train_mae', [])
        val_mae = history.get('val_mae', [])
        if train_mae and val_mae:
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(train_mae) + 1)
            ax.plot(epochs, train_mae, color=COLOR_TRAIN, label='Train MAE', alpha=0.85)
            ax.plot(epochs, val_mae, color=COLOR_VAL, label='Validation MAE', alpha=0.85)
            ax.axvline(len(train_mae), color='gray', linestyle='--', alpha=0.5,
                       label=f'Stop: {len(train_mae)} эпох')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('MAE (log_days)')
            ax.set_title('TabNet регрессия (ручной набор), train vs val MAE по эпохам')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, '23_tabnet_train_val_loss_reg.png'))
            plt.close()
            print('  Сохранён 23_tabnet_train_val_loss_reg.png')
except Exception as e:
    print(f'  Ошибка графика 23: {e}')


# 24. Train vs val AUC TabNet классификации

try:
    model = saved_models.get('grid_clf_manual') or saved_models.get('optuna_clf_manual')
    if model is not None and hasattr(model, 'history'):
        history = model.history
        train_auc = history.get('train_auc', [])
        val_auc = history.get('val_auc', [])
        if train_auc and val_auc:
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(train_auc) + 1)
            ax.plot(epochs, train_auc, color=COLOR_TRAIN, label='Train AUC', alpha=0.85)
            ax.plot(epochs, val_auc, color=COLOR_VAL, label='Validation AUC', alpha=0.85)
            ax.axvline(len(train_auc), color='gray', linestyle='--', alpha=0.5,
                       label=f'Stop: {len(train_auc)} эпох')
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('AUC')
            ax.set_title('TabNet классификация (ручной набор), train vs val AUC по эпохам')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, '24_tabnet_train_val_loss_clf.png'))
            plt.close()
            print('  Сохранён 24_tabnet_train_val_loss_clf.png')
except Exception as e:
    print(f'  Ошибка графика 24: {e}')


# 25. Pareto-фронт Optuna TabNet (обе задачи на одном)

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, study_key, ylabel, title in [
        (axes[0], 'optuna_reg_manual', 'CV качество (отриц. MAPE, %)', 'TabNet регрессия (manual)'),
        (axes[1], 'optuna_clf_manual', 'CV F1', 'TabNet классификация (manual)'),
    ]:
        study = saved_studies.get(study_key)
        if study is None:
            continue
        all_trials = [t for t in study.trials if t.values is not None and len(t.values) == 2]
        scores = [t.values[0] for t in all_trials]
        times = [t.values[1] for t in all_trials]
        pareto = study.best_trials
        pareto_scores = [t.values[0] for t in pareto]
        pareto_times = [t.values[1] for t in pareto]

        ax.scatter(times, scores, alpha=0.4, s=30, color=COLOR_PRIMARY, label='Все trials')
        ax.scatter(pareto_times, pareto_scores, alpha=0.95, s=80,
                   color=COLOR_PARETO, edgecolor='black',
                   label=f'Pareto-фронт ({len(pareto)})')
        ax.set_xlabel('Время на фолде, секунды')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    plt.suptitle('Pareto-фронт NSGA-II, TabNet')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '25_tabnet_optuna_pareto.png'))
    plt.close()
    print('  Сохранён 25_tabnet_optuna_pareto.png')
except Exception as e:
    print(f'  Ошибка графика 25: {e}')


# 26. Маски внимания TabNet (для интерпретируемости)

try:
    model = saved_models.get('grid_clf_manual') or saved_models.get('optuna_clf_manual')
    if model is not None:
        X_te = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'scaled', 'X_test.csv'))
        feature_names = X_te.columns.tolist()
        # explain returns aggregated mask
        sample_size = min(500, len(X_te))
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_te), sample_size, replace=False)
        explain_matrix, masks = model.explain(X_te.values[sample_idx])
        # average over samples
        avg_importance = explain_matrix.mean(axis=0)
        # top-15
        top_idx = np.argsort(avg_importance)[::-1][:15]
        top_features = [feature_names[i] for i in top_idx]
        top_values = avg_importance[top_idx]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.arange(len(top_features)), top_values[::-1], color=COLOR_PRIMARY, edgecolor='black')
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel('Усреднённая важность (TabNet attention)')
        ax.set_title('TabNet: топ-15 признаков по усреднённой маске внимания (классификация)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '26_tabnet_attention_masks.png'))
        plt.close()
        print('  Сохранён 26_tabnet_attention_masks.png')
except Exception as e:
    print(f'  Ошибка графика 26: {e}')

print('\nПайплайн 7 завершён')