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

from sklearn.model_selection import (
    KFold, StratifiedKFold, GridSearchCV, train_test_split, learning_curve,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, recall_score, precision_score,
    make_scorer, roc_curve, precision_recall_curve, average_precision_score,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings('ignore')

ROOT = r'D:\blablabla\Magistr'
SPLITS_DIR = os.path.join(ROOT, 'Dataset', 'splits')
MODELS_DIR = os.path.join(ROOT, 'Dataset', 'models', 'core')
RESULTS_DIR = os.path.join(ROOT, 'Dataset', 'results')
PLOTS_DIR = os.path.join(ROOT, 'Dataset', 'plots', 'training')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLOR_LINEAR = '#2c7fb8'
COLOR_KNN = '#7fbc41'
COLOR_RF = '#fdae61'
COLOR_XGB = '#d7191c'
COLOR_TRAIN = '#3a76b5'
COLOR_VAL = '#a83232'

RANDOM_STATE = 42
N_FOLDS = 10
N_JOBS = -1


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


# Метрики на исходной шкале

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


def neg_mape_log_scale(y_true_log, y_pred_log):
    y_true_days = np.expm1(y_true_log)
    y_pred_days = np.expm1(y_pred_log)
    eps = 1e-8
    return -np.mean(np.abs((y_true_days - y_pred_days) / (y_true_days + eps))) * 100


mape_scorer = make_scorer(neg_mape_log_scale, greater_is_better=True)


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


# Сетки гиперпараметров

def get_param_grid(model_name, task):
    if model_name == 'ridge':
        return {'alpha': np.logspace(-3, 5, 25)}

    if model_name == 'logistic':
        return {
            'C': np.logspace(-3, 3, 13),
            'class_weight': [None, 'balanced'],
            'penalty': ['l1', 'l2'],
        }

    if model_name == 'knn':
        return {
            'n_neighbors': [3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

    if model_name == 'rf':
        if task == 'reg':
            return {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 15, 20, 30, None],
                'min_samples_leaf': [1, 5, 20],
            }
        else:
            return {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 15, 20, 30, None],
                'min_samples_leaf': [1, 5, 20],
                'class_weight': [None, 'balanced'],
            }

    if model_name == 'xgb':
        if task == 'reg':
            return {
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.7, 1.0],
                'reg_lambda': [1, 10, 50, 100],
            }
        else:
            return {
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'reg_lambda': [1, 10, 50, 100],
                'scale_pos_weight': [1.0, 1.55],
            }
def grid_search_xgb(X_train, y_train, task, param_grid, n_folds=N_FOLDS):
    keys = list(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in product(*values_lists)]

    if task == 'clf':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    best_score = -np.inf
    best_params = None
    best_fold_scores = None

    for params in combos:
        fold_scores = []
        if task == 'clf':
            split_iter = kf.split(X_train, y_train)
        else:
            split_iter = kf.split(X_train)

        for fold_train_idx, fold_val_idx in split_iter:
            X_fold_tr = X_train.iloc[fold_train_idx]
            X_fold_val = X_train.iloc[fold_val_idx]
            y_fold_tr = y_train.iloc[fold_train_idx]
            y_fold_val = y_train.iloc[fold_val_idx]
            if task == 'reg':
                model = fit_xgb_regressor_es(X_fold_tr, y_fold_tr, params)
                pred = model.predict(X_fold_val)
                y_true_d = np.expm1(y_fold_val.values)
                y_pred_d = np.expm1(pred)
                eps = 1e-8
                score = -np.mean(np.abs((y_true_d - y_pred_d) / (y_true_d + eps))) * 100
            else:
                model = fit_xgb_classifier_es(X_fold_tr, y_fold_tr, params)
                proba = model.predict_proba(X_fold_val)[:, 1]
                pred = (proba >= 0.5).astype(int)
                score = f1_score(y_fold_val, pred, zero_division=0)
            fold_scores.append(score)
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_fold_scores = fold_scores

    return best_params, best_score, best_fold_scores


# Один эксперимент

def run_experiment(model_name, task, feature_set):
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

    param_grid = get_param_grid(model_name, task)

    if model_name == 'xgb':
        best_params, best_score, cv_fold_scores = grid_search_xgb(X_train, y_train, task, param_grid)
        if task == 'reg':
            best_model = fit_xgb_regressor_es(X_train, y_train, best_params)
        else:
            best_model = fit_xgb_classifier_es(X_train, y_train, best_params)
    else:
        if model_name == 'ridge':
            base = Ridge()
        elif model_name == 'logistic':
            base = LogisticRegression(solver='saga', max_iter=5000, random_state=RANDOM_STATE)
        elif model_name == 'knn':
            base = KNeighborsRegressor(n_jobs=1) if task == 'reg' else KNeighborsClassifier(n_jobs=1)
        elif model_name == 'rf':
            if task == 'reg':
                base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
            else:
                base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)

        scoring = mape_scorer if task == 'reg' else 'f1'
        cv_splitter = get_cv_splitter(task)

        gs = GridSearchCV(
            base, param_grid, scoring=scoring, cv=cv_splitter,
            n_jobs=N_JOBS, verbose=0, return_train_score=False,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_params = gs.best_params_
        best_score = gs.best_score_
        best_idx = gs.best_index_
        cv_fold_scores = [gs.cv_results_[f'split{i}_test_score'][best_idx] for i in range(N_FOLDS)]
        best_model = gs.best_estimator_

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

    model_path = os.path.join(MODELS_DIR, f'grid_{model_name}_{task}_{feature_set}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'best_params': best_params,
            'cv_fold_scores': cv_fold_scores,
            'best_cv_score': best_score,
        }, f)

    elapsed = time.time() - t0

    result_row = {
        'method': 'GridSearch',
        'model': model_name,
        'task': task,
        'feature_set': feature_set,
        'best_params': str(best_params),
        'best_cv_score': round(best_score, 4),
        'elapsed_sec': round(elapsed, 1),
    }
    result_row.update(metrics)
    result_row.update(sub_metrics)

    main_metric_value = list(metrics.values())[0]
    print(f'    Время: {elapsed:.1f} сек | best_cv: {best_score:.4f} | test main: {main_metric_value}')
    return result_row, cv_fold_scores


# План экспериментов

experiments = []
for task in ['reg', 'clf']:
    base_lin = 'ridge' if task == 'reg' else 'logistic'
    auto_set = 'auto_regression' if task == 'reg' else 'auto_classification'
    for model_name in [base_lin, 'knn', 'rf', 'xgb']:
        for feature_set in ['manual', auto_set]:
            experiments.append((model_name, task, feature_set))

print(f'Всего запланировано экспериментов: {len(experiments)}')
print('Пайплайн 5 запущен')

results = []
all_cv_scores = {}

for i, (model_name, task, feature_set) in enumerate(experiments, 1):
    print(f'\n[{i}/{len(experiments)}]')
    try:
        row, fold_scores = run_experiment(model_name, task, feature_set)
        results.append(row)
        all_cv_scores[f'{model_name}_{task}_{feature_set}'] = fold_scores
    except Exception as e:
        print(f'    ОШИБКА: {type(e).__name__}: {e}')
        results.append({
            'method': 'GridSearch',
            'model': model_name,
            'task': task,
            'feature_set': feature_set,
            'error': str(e),
        })

results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, 'grid_results_core.csv')
results_df.to_csv(results_path, index=False)
print(f'\nСохранены результаты: {results_path}')

cv_path = os.path.join(RESULTS_DIR, 'grid_cv_scores.pkl')
with open(cv_path, 'wb') as f:
    pickle.dump(all_cv_scores, f)
print(f'Сохранены CV-скоры: {cv_path}')


# Графики

print('\nПостроение графиков')

# Регрессия. Learning curve для RF (manual)

try:
    X_tr = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'X_train.csv'))
    y_tr_log = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'y_reg_train.csv')).iloc[:, 0]
    rf_model = pickle.load(open(os.path.join(MODELS_DIR, 'grid_rf_reg_manual.pkl'), 'rb'))['model']

    train_sizes, train_scores, val_scores = learning_curve(
        rf_model, X_tr, y_tr_log,
        train_sizes=np.linspace(0.25, 1.0, 4),
        cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, # уменьшено для скорости графика, основной CV использует N_FOLDS=10

    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, -train_scores.mean(axis=1), color=COLOR_TRAIN, marker='o', label='Train MAE')
    ax.plot(train_sizes, -val_scores.mean(axis=1), color=COLOR_VAL, marker='s', label='Validation MAE')
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('MAE (log_days)')
    ax.set_title('Learning curve, Random Forest, регрессия, ручной набор')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '12_learning_curve_rf.png'))
    plt.close()
    print('  Сохранён 12_learning_curve_rf.png')
except Exception as e:
    print(f'  Ошибка learning curve: {e}')


# Регрессия. Train vs val MAE по итерациям XGBoost

try:
    pkl = pickle.load(open(os.path.join(MODELS_DIR, 'grid_xgb_reg_manual.pkl'), 'rb'))
    xgb_model = pkl['model']
    if hasattr(xgb_model, 'evals_result_') and xgb_model.evals_result_:
        evals = xgb_model.evals_result_
        keys = list(evals.keys())
        metric_key = list(evals[keys[0]].keys())[0]
        train_loss = evals[keys[0]][metric_key]
        val_loss = evals[keys[1]][metric_key] if len(keys) > 1 else None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_loss, color=COLOR_TRAIN, label=f'Train {metric_key}', alpha=0.85)
        if val_loss is not None:
            ax.plot(val_loss, color=COLOR_VAL, label=f'Validation {metric_key}', alpha=0.85)
        ax.axvline(xgb_model.best_iteration, color='gray', linestyle='--', alpha=0.5,
            label=f'Early stop: {xgb_model.best_iteration} итер')
        ax.set_xlabel('Итерация')
        ax.set_ylabel(metric_key)
        ax.set_title('XGBoost, регрессия, ручной набор: train vs val MAE')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '13_xgboost_train_val_loss_reg.png'))
        plt.close()
        print('  Сохранён 13_xgboost_train_val_loss_reg.png')
except Exception as e:
    print(f'  Ошибка XGBoost loss reg: {e}')


# Регрессия. CV-score boxplot

try:
    cv_for_boxplot_reg = {
        'Ridge': all_cv_scores.get('ridge_reg_manual'),
        'k-NN': all_cv_scores.get('knn_reg_manual'),
        'RF': all_cv_scores.get('rf_reg_manual'),
        'XGBoost': all_cv_scores.get('xgb_reg_manual'),
    }
    cv_for_boxplot_reg = {k: v for k, v in cv_for_boxplot_reg.items() if v is not None}
    if cv_for_boxplot_reg:
        labels = list(cv_for_boxplot_reg.keys())
        data = [cv_for_boxplot_reg[k] for k in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = [COLOR_LINEAR, COLOR_KNN, COLOR_RF, COLOR_XGB][:len(labels)]
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel('CV score (отрицательный MAPE, %)')
        ax.set_title('Распределение CV-скоров по 10 фолдам, регрессия, ручной набор')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '14_cv_boxplot_regression.png'))
        plt.close()
        print('  Сохранён 14_cv_boxplot_regression.png')
except Exception as e:
    print(f'  Ошибка boxplot reg: {e}')


# Классификация. ROC-AUC и Precision-Recall

try:
    X_te_proc = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'X_test.csv'))
    y_te_clf = pd.read_csv(os.path.join(SPLITS_DIR, 'manual', 'processed', 'y_clf_test.csv')).iloc[:, 0].values

    rf_clf = pickle.load(open(os.path.join(MODELS_DIR, 'grid_rf_clf_manual.pkl'), 'rb'))['model']
    xgb_clf = pickle.load(open(os.path.join(MODELS_DIR, 'grid_xgb_clf_manual.pkl'), 'rb'))['model']

    rf_proba = rf_clf.predict_proba(X_te_proc)[:, 1]
    xgb_proba = xgb_clf.predict_proba(X_te_proc)[:, 1]

    fpr_rf, tpr_rf, _ = roc_curve(y_te_clf, rf_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_te_clf, xgb_proba)
    auc_rf = roc_auc_score(y_te_clf, rf_proba)
    auc_xgb = roc_auc_score(y_te_clf, xgb_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_rf, tpr_rf, color=COLOR_RF, lw=2, label=f'Random Forest (AUC={auc_rf:.3f})')
    ax.plot(fpr_xgb, tpr_xgb, color=COLOR_XGB, lw=2, label=f'XGBoost (AUC={auc_xgb:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривые, классификация, ручной набор')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '15_roc_curves_classification.png'))
    plt.close()
    print('  Сохранён 15_roc_curves_classification.png')

    pr_p_rf, pr_r_rf, _ = precision_recall_curve(y_te_clf, rf_proba)
    pr_p_xgb, pr_r_xgb, _ = precision_recall_curve(y_te_clf, xgb_proba)
    ap_rf = average_precision_score(y_te_clf, rf_proba)
    ap_xgb = average_precision_score(y_te_clf, xgb_proba)
    pos_share = y_te_clf.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pr_r_rf, pr_p_rf, color=COLOR_RF, lw=2, label=f'Random Forest (AP={ap_rf:.3f})')
    ax.plot(pr_r_xgb, pr_p_xgb, color=COLOR_XGB, lw=2, label=f'XGBoost (AP={ap_xgb:.3f})')
    ax.axhline(pos_share, color='gray', linestyle='--', lw=1, label=f'Random (P={pos_share:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall кривые, классификация, ручной набор')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '16_pr_curves_classification.png'))
    plt.close()
    print('  Сохранён 16_pr_curves_classification.png')
except Exception as e:
    print(f'  Ошибка ROC/PR кривых: {e}')


# Классификация. Train vs val AUC XGBoost

try:
    pkl = pickle.load(open(os.path.join(MODELS_DIR, 'grid_xgb_clf_manual.pkl'), 'rb'))
    xgb_clf_model = pkl['model']
    if hasattr(xgb_clf_model, 'evals_result_') and xgb_clf_model.evals_result_:
        evals = xgb_clf_model.evals_result_
        keys = list(evals.keys())
        metric_key = list(evals[keys[0]].keys())[0]
        train_metric = evals[keys[0]][metric_key]
        val_metric = evals[keys[1]][metric_key] if len(keys) > 1 else None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_metric, color=COLOR_TRAIN, label=f'Train {metric_key}', alpha=0.85)
        if val_metric is not None:
            ax.plot(val_metric, color=COLOR_VAL, label=f'Validation {metric_key}', alpha=0.85)
        ax.axvline(len(train_metric) - 1, color='gray', linestyle='--', alpha=0.5,
                   label=f'Early stop: {len(train_metric)} итер')
        ax.set_xlabel('Итерация')
        ax.set_ylabel(metric_key)
        ax.set_title('XGBoost, классификация, ручной набор: train vs val AUC')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '17_xgboost_train_val_loss_clf.png'))
        plt.close()
        print('  Сохранён 17_xgboost_train_val_loss_clf.png')
except Exception as e:
    print(f'  Ошибка XGBoost loss clf: {e}')


# Классификация. CV F1 boxplot

try:
    cv_for_boxplot_clf = {
        'Logistic': all_cv_scores.get('logistic_clf_manual'),
        'k-NN': all_cv_scores.get('knn_clf_manual'),
        'RF': all_cv_scores.get('rf_clf_manual'),
        'XGBoost': all_cv_scores.get('xgb_clf_manual'),
    }
    cv_for_boxplot_clf = {k: v for k, v in cv_for_boxplot_clf.items() if v is not None}
    if cv_for_boxplot_clf:
        labels = list(cv_for_boxplot_clf.keys())
        data = [cv_for_boxplot_clf[k] for k in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = [COLOR_LINEAR, COLOR_KNN, COLOR_RF, COLOR_XGB][:len(labels)]
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_ylabel('CV F1-score')
        ax.set_title('Распределение CV F1 по 10 фолдам, классификация, ручной набор')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '18_cv_boxplot_classification.png'))
        plt.close()
        print('  Сохранён 18_cv_boxplot_classification.png')
except Exception as e:
    print(f'  Ошибка boxplot clf: {e}')

print('\nПайплайн 5 завершён')