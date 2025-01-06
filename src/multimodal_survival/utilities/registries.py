from typing import Any, Dict, Type

import torch.nn as nn
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, ReduceLROnPlateau

SURVIVAL_MODELS: Dict[str, Any] = {
    "coxph": CoxPHSurvivalAnalysis,
    "random_survival_forest": RandomSurvivalForest,
    "coxnet": CoxnetSurvivalAnalysis,
    "gradient_boosting": GradientBoostingSurvivalAnalysis,
}

SCORER_WRAPPER = {
    "cindex_ipcw": as_concordance_index_ipcw_scorer,
    "cum_dynamic": as_cumulative_dynamic_auc_scorer,
    "ibs": as_integrated_brier_score_scorer,
}

CROSS_VALIDATION: Dict[str, Type[_BaseKFold]] = {
    "repeated_kfold": RepeatedKFold,
    "repeated_stratified_kfold": RepeatedStratifiedKFold,
}

FEATURE_SELECTORS: Dict[str, Type[SelectorMixin]] = {
    "variance": VarianceThreshold,
}

FEATURE_IMPUTERS: Dict[str, Type[TransformerMixin]] = {
    "knn": KNNImputer,
    "simple": SimpleImputer,
}

FEATURE_TRANSFORMERS: Dict[str, Type[TransformerMixin]] = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
}

OPTIMIZERS = {"adam": Adam}

OPTIMIZER_SCHEDULERS = {
    "plateau": ReduceLROnPlateau,
    "cyclic": CyclicLR,
    "exp": ExponentialLR,
}

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "softmax": nn.Softmax(),
}

CLUSTERING_METHOD_FACTORY = {
    "kmeans": KMeans,
    "spectral": SpectralClustering,
    "heirarchical": AgglomerativeClustering,
    "dbscan": DBSCAN,
}

CLUSTERING_METRIC_FACTORY = {
    "silhouette_avg": metrics.silhouette_score,
    "silhouette_sample": metrics.silhouette_samples,
    "ari": metrics.adjusted_rand_score,
    "ami": metrics.adjusted_mutual_info_score,
    "nmi": metrics.normalized_mutual_info_score,
    "homogeneity": metrics.homogeneity_score,
    "completeness": metrics.completeness_score,
    "vmeasure": metrics.v_measure_score,
}
