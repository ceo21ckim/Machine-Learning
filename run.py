import os, argparse

from sklearn.datasets import load_diabetes

from machine_learning import MachineLearning
from dataloader import DataLoader

from metrics import metrics

import ast 
from settings import DATA_DIR, BASE_DIR, OUT_DIR


parser = argparse.ArgumentParser(description='machine learning example')
parser.add_argument('--fname', required=True)
parser.add_argument('--estimators', required=True, type=str)
parser.add_argument('--final_estimators', default='ridge', required=False, type=str)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--num_feature_names', type=str)
parser.add_argument('--cat_feature_names', default=None, type=str)
parser.add_argument('--target_name', default='target', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    # df = pd.read_csv(os.path.join(DATA_DIR, fname, encoding='utf-8-sig'))

    dataframe = load_diabetes(as_frame=True)
    df = dataframe['frame']
    
    args.cat_feature_names = ast.literal_eval(args.cat_feature_names) if args.cat_feature_names else None
    args.num_feature_names = ast.literal_eval(args.num_feature_names) if args.num_feature_names else df.columns.difference([args.target_name] + args.cat_feature_names if args.cat_feature_names else [args.target_name])
    feature_names = args.num_feature_names.tolist() + args.cat_feature_names if args.cat_feature_names else args.num_feature_names

    data_loader = DataLoader(dataframe=df, feature_names=feature_names, target_name=args.target_name) # feature_names=feature_names

    X_train, X_test, y_train, y_test = data_loader.train_test_split()

    model = MachineLearning(estimators=ast.literal_eval(args.estimators), final_estimator=args.final_estimators, num_feature_names=args.num_feature_names, cat_feature_names=args.cat_feature_names, random_state=args.random_state).regressor

    model.fit(X_train, y_train.values.ravel()) # y shape: (n_samples, )

    y_pred = model.predict(X_test)

    X_test['y_pred'] = y_pred 
    X_test['y_true'] = y_test

    scores = metrics(X_test['y_true'], X_test['y_pred'])

    scores.to_csv(os.path.join(OUT_DIR, 'scores.csv'), encoding='utf-8-sig', index=False)
