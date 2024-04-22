import pandas as pd
import numpy as np
import dill
import os
import logging
from datetime import datetime as dt

from catboost import CatBoostClassifier
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample


path = os.environ.get('PROJECT_PATH', '..')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'device_model',
        'utm_keyword',
        'device_os',
        'client_id'
    ]
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:  # define function so it can take DF and return DF
    def calc_outliers(data):
        q25 = data.quantile(0.25)  # take quantiles of data
        q75 = data.quantile(0.75)
        iqr = q75 - q25  # save median
        bounds = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)  # return boundaries tuple
        return bounds

    df = df.copy()  # make copy so everything can be safe
    boundaries = calc_outliers(df['visit_number'])
    df.loc[df['visit_number'] < boundaries[0], 'visit_number'] = round(boundaries[0])  # limit outliers within boundaries
    df.loc[df['visit_number'] > boundaries[1], 'visit_number'] = round(boundaries[1])

    return df


def types_handling(df: pd.DataFrame) -> pd.DataFrame:
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    df['visit_time'] = pd.to_datetime(df['visit_time']).dt.hour
    df['device_category'] = df['device_category'].astype('category')

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['month_visit'] = df['visit_date'].dt.month
    df['weekday_visit'] = df['visit_date'].dt.weekday
    df['day_visit'] = df['visit_date'].dt.day
    bins = [0, 5, 12, 18, 23]
    labels = ['night', 'morning', 'day', 'evening']
    df['daytime'] = pd.cut(df['visit_time'], bins=bins, labels=labels, right=False)

    df['utm_medium'] = df['utm_medium'].str.replace(r'\(|\)', '', regex=True).str.lower().str.replace(' ', '_')

    df['device_brand'] = df['device_brand'].str.replace(r'\(|\)', '', regex=True).str.lower().str.replace(' ', '_')

    df['width_res'] = df['device_screen_resolution'].str.split('x').str[0]
    df['height_res'] = df['device_screen_resolution'].str.split('x').str[1]
    df.loc[df['width_res'] == '(not set)', 'width_res'] = df['width_res'].mode().iloc[0]
    df['height_res'] = df['height_res'].fillna(df['height_res'].mode().iloc[0])
    df = df.drop(columns=['device_screen_resolution'])
    df['width_res'] = df['width_res'].astype('int32')
    df['height_res'] = df['height_res'].astype('int32')

    df['short_browser'] = df['device_browser'].str.replace(r'\(|\)', '', regex=True).str.lower().str.split(' ').str[0]
    df = df.drop('device_browser', axis=1)

    df['geo_country'] = df['geo_country'].str.replace(r'\(|\)', '', regex=True).str.lower().str.replace(' ', '_')
    df['geo_city'] = df['geo_city'].str.replace(r'\(|\)', '', regex=True).str.lower().str.replace(' ', '_')

    df = df.drop(columns='visit_date')

    return df


def pipeline() -> None:
    df_sessions = pd.read_csv(f'{path}/data/train/ga_sessions.csv')
    df_hits = pd.read_csv(f'{path}/data/train/ga_hits.csv')

    target_rows = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                   'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                   'sub_submit_success', 'sub_car_request_submit_click']

    df_hits['event_action_target'] = df_hits.event_action.apply(lambda x: 1 if x in target_rows else np.nan)
    df_hits_target = df_hits[df_hits['event_action_target'] == 1][['session_id', 'event_action_target']]
    df_hits_target = df_hits_target.drop_duplicates()

    df = pd.merge(df_sessions, df_hits_target, on='session_id', how='left')
    df['event_action_target'] = df['event_action_target'].fillna(0)
    df['event_action_target'] = df['event_action_target'].astype('int32')

    target_samples = df[df['event_action_target'] == 1]
    other_samples = df[df['event_action_target'] == 0]

    target_count = len(target_samples)

    downsampled = resample(other_samples, replace=False, n_samples=target_count, random_state=42)

    df = pd.concat([downsampled, target_samples])

    X = df.drop(['event_action_target', 'session_id'], axis=1)
    y = df.event_action_target

    numerical_features = make_column_selector(dtype_include=['int32', 'int64', 'float64', 'float32'])
    categorical_features = make_column_selector(dtype_include=['object', 'category'])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outlier_remover', FunctionTransformer(remove_outliers)),
        ('types_handling', FunctionTransformer(types_handling)),
        ('feature_creator', FunctionTransformer(create_features)),
        ('column_transformer', column_transformer)
    ])

    models = [
        CatBoostClassifier(depth=10, learning_rate=0.03, iterations=1000,
                           random_state=42, eval_metric='AUC', verbose=False, early_stopping_rounds=20),
        LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=42, verbose=False),
        RandomForestClassifier(max_depth=30, min_samples_leaf=10, min_samples_split=20,
                               n_estimators=500, random_state=42, verbose=False),
        LinearSVC(C=0.3, max_iter=1000, penalty='l2', random_state=42, verbose=False)
    ]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        logger.info(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    logger.info(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC-AUC: {best_score:.4f}')

    best_pipe.fit(X, y)
    model_filename = f'{path}/data/models/sber_auto_{dt.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Session target action prediction',
                'author': 'Roman Lapa',
                'version': 1,
                'date': dt.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'ROC-AUC': best_score
            }
        }, file)

    logger.info(f'Model is saved as {model_filename}')


if __name__ == '__main__':
    pipeline()
