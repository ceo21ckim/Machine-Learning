from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor 
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from typing import Union

def get_regressor(model_nm, random_state=42, **kwargs):
    model_dict = {
        'rf': RandomForestRegressor(random_state=random_state, **kwargs),
        'ridge': RidgeCV(**kwargs), 
        'ada': AdaBoostRegressor(random_state=random_state, **kwargs)
    }
    return model_dict[model_nm]


class MachineLearning:
    def __init__(self, estimators: Union[str, list], random_state:int = 42, pca: bool = False, **kwargs):

        self.estimators = estimators
        self.random_state = random_state 
        self.pca = pca 

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.pca:
            self.pca_transformer = ColumnTransformer(
                transformers = [
                    ('PCA', PCA(n_components=self.pca), self.num_feature_names)
                ]
            )
            self.numerical_transformer = Pipeline(
                steps = [
                    ('pca', self.pca_transformer), ('num_encoder', StandardScaler())
                    ]
            )

        else:
            self.numerical_transformer = Pipeline(
                steps=[
                    ('num_encoder', StandardScaler())
                    ]
            )

        self.categorical_transformer = Pipeline(
            steps = [
                ('encoder', OneHotEncoder())
                ]
        )

        self.preprocessor = ColumnTransformer(
            transformers= [
            ('num_encoder', self.numerical_transformer, self.num_feature_names), 
            ('cat_encoder', self.categorical_transformer, self.cat_feature_names)
        ] if self.cat_feature_names else [('num_encoder', self.numerical_transformer, self.num_feature_names)]
        )

        self.estimators = [
            (m, get_regressor(m, random_state=self.random_state)) for m in self.estimators
        ] if isinstance(self.estimators, list) else get_regressor(self.estimators, random_state=self.random_state)
        
        if isinstance(self.estimators, list):
            self.final_estimator = get_regressor(self.final_estimator)
            self.regressor = self.stacking_ensemble(self.preprocessor, self.estimators, self.final_estimator)
        else:
            self.regressor = Pipeline(
                steps = [
                    ('preprocessor', self.preprocessor),
                    ('estimator', self.estimators)
                ]
            )

    def stacking_ensemble(self, preprocessor, estimators, final_estimator):
        estimators = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        estimators = Pipeline(
            steps = [
                ('preprocessor', preprocessor),
                ('estimators', estimators)
            ]
        )
        return estimators
