import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from model import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score

def dataPreprocessing():
    """ TODO, implement your own dataPreprocess function here. """

    raw_train_X=pd.read_csv('train_X.csv')
    raw_train_Y=pd.read_csv('train_y.csv')
    raw_test_X=pd.read_csv('test_X.csv')
    raw_test_Y=pd.read_csv('test_y.csv')

    # raw_train_X.columns[0]: select the first column
    # axis=1: specify that i am dropping a colunmn(not a row)
    # inplace=True: Applies the changes directly to the dataframe
    raw_train_X.drop(raw_train_X.columns[0], axis=1, inplace=True)
    raw_train_Y.drop(raw_train_Y.columns[0], axis=1, inplace=True)
    raw_test_X.drop(raw_test_X.columns[0], axis=1, inplace=True)
    raw_test_Y.drop(raw_test_Y.columns[0], axis=1, inplace=True)

    train_preprocessor=Preprocessor(raw_train_X)
    test_preprocessor=Preprocessor(raw_test_X)


    train_X=train_preprocessor.preprocess()
    test_X=test_preprocessor.preprocess()

    train_X_np=train_X.to_numpy()
    train_Y_np=raw_train_Y.to_numpy()
    test_X_np=test_X.to_numpy()
    test_Y_np=raw_test_Y.to_numpy()

    print("train_X_np:",train_X_np)

    return train_X_np, train_Y_np, test_X_np, test_Y_np # train, test data should be numpy array


def main():
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index
    model = LogisticRegressionClassifier()
    model.fit(train_X, train_y)

    pred = model.predict(test_X)
    prob = model.predict_proba(test_X)
    prob = [f'{x:.5f}' for x in prob]
    # print(f'Prob: {prob}')
    print(f'Acc: {accuracy_score(pred, test_y):.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

