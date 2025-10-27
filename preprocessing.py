import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, PowerTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, ADASYN


df = pd.read_csv('data/raw/twitter_human_bots_dataset.csv', index_col=0)

#1) Train_Test_Split
X = df.drop(columns=['account_type', 'id'])
y = df['account_type'].map({'bot':1, 'human':0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#2) Feature Engineering
def add_features(X):
    X = X.copy()

    # DROP unusable string/datetime columns early
    drop_cols = ['created_at', 'id','lang', 'location', 'profile_background_image_url', 'profile_image_url', 'screen_name']
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=col)

    # new features
    X['followers_friends_ratio'] = np.where(
        (X.get('followers_count', 0) != 0) & (X.get('friends_count', 0) != 0),
        X['followers_count'] / X['friends_count'], 0
    )

    X['description'] = X['description'].fillna("")
    X['description_length'] = X['description'].astype(str).str.len()
    # create has_description (1 if non-empty description, else 0)
    X['has_description'] = X.get('description', '').astype(str).str.strip().ne('').astype(bool)

    #Numerical Columns (log scale it)
    num_cols = X.select_dtypes(include=['float64','int64']).columns
    for col in num_cols:
        if col == 'account_age_days':
            #doesn't need to be log-transformed
            continue
        else:
            X[col] = np.log1p(X[col])
    
    #Boolean Columns: (0-1 it)
    X['default_profile'] = X['default_profile'].map({True:1, False:0})
    X['default_profile_image'] = X['default_profile_image'].map({True:1, False:0})
    X['geo_enabled'] = X['geo_enabled'].map({True:1, False:0})
    X['verified'] = X['verified'].map({True:1, False:0})
    X['has_description'] = X['has_description'].map({True:1, False:0})
    
    #Now drop description
    X = X.drop(columns='description')

    return X

# fe = FunctionTransformer(add_features, validate=False, feature_names_out='one-to-one')
fe = FunctionTransformer(add_features, validate=False)
fe.set_output(transform="pandas")


#3) Column Blocks (for transformation)
num_cols_standard = ['favourites_count', 'statuses_count']
num_cols_yeo_johnson  = ['average_tweets_per_day', 'account_age_days', 'description_length']
num_cols_robust   = ['followers_count', 'friends_count', 'followers_friends_ratio']
cat_cols = ['default_profile', 'default_profile_image', 'geo_enabled', 'verified', 'has_description']

#4) Imputing and Scaling
num_standard = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler())
])
num_yeo_johnson = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  PowerTransformer(method='yeo-johnson')),
    ('scale', StandardScaler())
])
num_robust = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  RobustScaler())
])
cat = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('oh',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

pre = ColumnTransformer(
    transformers=[
        ('num_std', num_standard, num_cols_standard),
        ('num_yj', num_yeo_johnson, num_cols_yeo_johnson),
        ('num_robust', num_robust, num_cols_robust),
        ('cat', cat, cat_cols)
    ],
    remainder='drop',
    n_jobs=None,
    verbose_feature_names_out=False
)

X_train_fe = fe.fit_transform(X_train)
cat_idx = [X_train_fe.columns.get_loc(c) for c in cat_cols]


pipe = Pipeline([
    ('fe',  fe),
    ('pre', pre),
])

pipe_smote = ImbPipeline([
    ('fe',  fe),
    ('smote', SMOTENC(categorical_features=cat_idx, sampling_strategy='auto', random_state=42)),
    ('pre', pre),
])


pipe_adasyn = ImbPipeline([
    ('fe',  fe),
    ('adasyn', ADASYN(sampling_strategy='auto', random_state=42)),
    ('pre', pre),
])


# Ingest CSV into interim folder

pipe.fit(X_train, y_train)

X_train_proc = pipe.transform(X_train)
X_test_proc  = pipe.transform(X_test)

feat_names = pipe.named_steps['pre'].get_feature_names_out()

X_train_proc_df = pd.DataFrame(X_train_proc, columns=feat_names, index=X_train.index)
X_test_proc_df  = pd.DataFrame(X_test_proc,  columns=feat_names, index=X_test.index)


train_out = X_train_proc_df.assign(account_type=y_train.values)
test_out  = X_test_proc_df.assign(account_type=y_test.values)


train_out.to_csv("data/interim/twitter_train_processed.csv", index=False)
test_out.to_csv("data/interim/twitter_test_processed.csv", index=False)

pipe_smote.fit(X_train, y_train)

X_train_proc = pipe.transform(X_train)
X_test_proc  = pipe.transform(X_test)

feat_names = pipe.named_steps['pre'].get_feature_names_out()

X_train_proc_df = pd.DataFrame(X_train_proc, columns=feat_names, index=X_train.index)
X_test_proc_df  = pd.DataFrame(X_test_proc,  columns=feat_names, index=X_test.index)


train_out = X_train_proc_df.assign(account_type=y_train.values)
test_out  = X_test_proc_df.assign(account_type=y_test.values)


train_out.to_csv("data/interim/twitter_train_processed_SMOTE.csv", index=False)
test_out.to_csv("data/interim/twitter_test_processed_SMOTE.csv", index=False)


pipe_adasyn.fit(X_train, y_train)

X_train_proc = pipe.transform(X_train)
X_test_proc  = pipe.transform(X_test)

feat_names = pipe.named_steps['pre'].get_feature_names_out()

X_train_proc_df = pd.DataFrame(X_train_proc, columns=feat_names, index=X_train.index)
X_test_proc_df  = pd.DataFrame(X_test_proc,  columns=feat_names, index=X_test.index)


train_out = X_train_proc_df.assign(account_type=y_train.values)
test_out  = X_test_proc_df.assign(account_type=y_test.values)


train_out.to_csv("data/interim/twitter_train_processed_adasyn.csv", index=False)
test_out.to_csv("data/interim/twitter_test_processed_adasyn.csv", index=False)



# Or can just export the pipe 
__all__ = ['pipe']
