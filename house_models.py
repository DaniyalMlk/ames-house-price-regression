"""Housing regression experiments. solns.csv is never read by this module."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer, PowerTransformer
from sklearn.svm import SVR


QUALITY = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORDINALS = {
    **{c: QUALITY for c in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']},
    'BsmtExposure': {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'CentralAir': {'N': 0, 'Y': 1},
}
STRUCTURAL_ZERO = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']


class CategoricalBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=1800, depth=5, learning_rate=.035, l2_leaf_reg=3, random_strength=1, bagging_temperature=1):
        self.iterations=iterations; self.depth=depth; self.learning_rate=learning_rate
        self.l2_leaf_reg=l2_leaf_reg; self.random_strength=random_strength; self.bagging_temperature=bagging_temperature

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(**self.get_params(), loss_function='RMSE', random_seed=73, thread_count=1, verbose=False, allow_writing_files=False)
        self.model_.fit(X,y,cat_features=X.select_dtypes(include=['object','string']).columns.tolist())
        return self

    def predict(self, X):
        return self.model_.predict(X)


class HouseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, engineered=True, log_inputs=False, interactions=False, native=False):
        self.engineered = engineered
        self.log_inputs = log_inputs
        self.interactions = interactions
        self.native = native

    def fit(self, X, y=None):
        self.frontage_ = X.groupby('Neighborhood', observed=True)['LotFrontage'].median()
        self.frontage_global_ = X['LotFrontage'].median()
        return self

    def transform(self, X):
        d = X.drop(columns=['Id', 'SalePrice'], errors='ignore').copy()
        d['LotFrontageMissing'] = d['LotFrontage'].isna().astype(int)
        d['LotFrontage'] = d['LotFrontage'].fillna(d['Neighborhood'].map(self.frontage_)).fillna(self.frontage_global_)
        d[STRUCTURAL_ZERO] = d[STRUCTURAL_ZERO].fillna(0)
        for c, mapping in ORDINALS.items():
            d[c] = d[c].map(mapping).fillna(0).astype(float)
        d['MSSubClass'] = d['MSSubClass'].astype(str)
        if self.engineered:
            d['TotalSF'] = d['TotalBsmtSF'] + d['1stFlrSF'] + d['2ndFlrSF']
            d['FinishedSF'] = d['GrLivArea'] + d['BsmtFinSF1'] + d['BsmtFinSF2']
            d['TotalBath'] = d['FullBath'] + d['BsmtFullBath'] + .5 * (d['HalfBath'] + d['BsmtHalfBath'])
            d['TotalPorchSF'] = d[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
            d['HouseAge'] = (d['YrSold'] - d['YearBuilt']).clip(lower=0)
            d['RemodelAge'] = (d['YrSold'] - d['YearRemodAdd']).clip(lower=0)
            d['GarageAge'] = (d['YrSold'] - d['GarageYrBlt']).clip(lower=0).fillna(0)
            d['Remodeled'] = (d['YearRemodAdd'] != d['YearBuilt']).astype(int)
            d['NewBuild'] = (d['YearBuilt'] == d['YrSold']).astype(int)
            for key, source in [('HasGarage', 'GarageArea'), ('HasBasement', 'TotalBsmtSF'), ('HasPool', 'PoolArea'), ('HasFireplace', 'Fireplaces'), ('HasSecondFloor', '2ndFlrSF')]:
                d[key] = (d[source] > 0).astype(int)
            d['FinishRatio'] = ((d['BsmtFinSF1'] + d['BsmtFinSF2']) / (d['TotalBsmtSF'] + 1)).clip(0, 1)
            d['LivingLotRatio'] = d['GrLivArea'] / (d['LotArea'] + 1)
            d['AverageRoomSF'] = d['GrLivArea'] / d['TotRmsAbvGrd'].clip(lower=1)
            d['QualityCondition'] = d['OverallQual'] * d['OverallCond']
            d['MonthSin'] = np.sin(2*np.pi*d['MoSold']/12)
            d['MonthCos'] = np.cos(2*np.pi*d['MoSold']/12)
        if self.interactions:
            d['QualLiving'] = d['OverallQual'] * np.log1p(d['GrLivArea'])
            d['QualBasement'] = d['OverallQual'] * np.log1p(d['TotalBsmtSF'])
            d['QualSquared'] = d['OverallQual'] ** 2
            d['KitchenLiving'] = d['KitchenQual'] * np.log1p(d['GrLivArea'])
            d['AgeQuality'] = d['OverallQual'] * np.log1p((d['YrSold']-d['YearBuilt']).clip(lower=0))
        if self.log_inputs:
            positive = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'TotalSF', 'FinishedSF', 'TotalPorchSF', 'HouseAge', 'RemodelAge', 'GarageAge', 'AverageRoomSF']
            for c in positive:
                if c in d:
                    d[c] = np.log1p(d[c].clip(lower=0))
        cats = d.select_dtypes(include=['object', 'string']).columns
        for c in cats:
            d[c] = d[c].fillna('None').astype(str)
            if self.native:
                d[c] = d[c].astype('category')
        return d.replace([np.inf, -np.inf], np.nan)


def make_model(spec):
    s = dict(spec)
    family = s.pop('family')
    target = s.pop('target', 'log')
    engineered = s.pop('engineered', True)
    log_inputs = s.pop('log_inputs', family in ['ridge', 'elastic', 'svr', 'krr'])
    interactions = s.pop('interactions', False)
    native = family in ['hist','lightgbm']
    features = HouseFeatures(engineered=engineered, log_inputs=log_inputs, interactions=interactions, native=native)
    if family == 'catboost':
        pipe = Pipeline([('features', features), ('model', CategoricalBoostRegressor(**s))])
    elif family == 'lightgbm':
        from lightgbm import LGBMRegressor
        pipe = Pipeline([('features',features),('model',LGBMRegressor(verbosity=-1, n_jobs=1, random_state=73, deterministic=True, force_col_wise=True, **s))])
    elif native:
        estimator = HistGradientBoostingRegressor(early_stopping=False, categorical_features='from_dtype', random_state=73, **s)
        pipe = Pipeline([('features', features), ('model', estimator)])
    else:
        scale = family in ['ridge', 'elastic', 'svr', 'krr']
        num = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scaler', StandardScaler() if scale else 'passthrough')])
        prep = ColumnTransformer([('num', num, make_column_selector(dtype_include=np.number)), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), make_column_selector(dtype_include=object))], sparse_threshold=0)
        if family == 'ridge': estimator = Ridge(**s)
        elif family == 'elastic': estimator = ElasticNet(max_iter=20000, tol=1e-5, **s)
        elif family == 'svr': estimator = SVR(**s)
        elif family == 'krr': estimator = KernelRidge(**s)
        elif family == 'gbr': estimator = GradientBoostingRegressor(random_state=73, **s)
        elif family == 'extra': estimator = ExtraTreesRegressor(random_state=73, n_jobs=1, **s)
        elif family == 'forest': estimator = RandomForestRegressor(random_state=73, n_jobs=1, **s)
        else: raise ValueError(family)
        pipe = Pipeline([('features', features), ('preprocess', prep), ('model', estimator)])
    if target == 'log':
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    if target == 'sqrt':
        return TransformedTargetRegressor(regressor=pipe, func=np.sqrt, inverse_func=np.square, check_inverse=False)
    if target == 'standard':
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    if target == 'log_centered':
        transform=Pipeline([('log',FunctionTransformer(np.log1p,inverse_func=np.expm1)),('scale',StandardScaler())])
        return TransformedTargetRegressor(regressor=pipe,transformer=transform)
    if target == 'power':
        return TransformedTargetRegressor(regressor=pipe,transformer=PowerTransformer(method='box-cox'))
    return pipe


def comparable_guard(X_train, y_train, X_new, predictions, k=1):
    """For living areas beyond the observed maximum, use local comparables.

    This rule is fitted afresh in each fold and never reads test labels.
    It is a deliberately narrow guard, not a replacement for the main model.
    """
    out=np.asarray(predictions,dtype=float).copy()
    if k==0: return out
    outside=X_new['GrLivArea'].to_numpy() > X_train['GrLivArea'].max()
    columns=['GrLivArea','TotalBsmtSF','OverallQual','YearBuilt']
    train=X_train[columns].fillna(0).to_numpy(float)
    new=X_new[columns].fillna(0).to_numpy(float)
    train[:,:2]=np.log1p(train[:,:2]);new[:,:2]=np.log1p(new[:,:2])
    scale=np.maximum(np.std(train,axis=0),1e-6)
    target=np.asarray(y_train)
    for i in np.flatnonzero(outside):
        same=(X_train['Neighborhood'].to_numpy()==X_new['Neighborhood'].iloc[i])
        indices=np.flatnonzero(same)
        if len(indices)<k: continue
        distances=np.sum(((train[indices]-new[i])/scale)**2,axis=1)
        chosen=indices[np.argsort(distances,kind='stable')[:k]]
        out[i]=np.mean(target[chosen])
    return out


def training_mask(X, y, policy='all'):
    if policy == 'all': return np.ones(len(X), dtype=bool)
    if policy == 'remove_extreme': return ~((X['GrLivArea'].to_numpy() > 4000) & (np.asarray(y) < 300000))
    raise ValueError(policy)
