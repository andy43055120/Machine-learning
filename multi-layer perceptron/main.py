import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import MLPClassifier, activation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import os
import torch


def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
     
    root_path = r"C:\Users\user\Desktop\introduction to machine learing\hw2" # change the root path 
    
    train_X_path = os.path.join(root_path, "train_x.csv")
    train_y_path = os.path.join(root_path, "train_y.csv")
    test_X_path = os.path.join(root_path, "test_x.csv")
    test_y_path = os.path.join(root_path, "test_y.csv")



    raw_train_X=pd.read_csv(train_X_path)
    raw_train_Y=pd.read_csv(train_y_path)
    raw_test_X=pd.read_csv(test_X_path)
    raw_test_Y=pd.read_csv(test_y_path)

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






        
    return train_X_np, train_Y_np, test_X_np, test_Y_np # train, test data should be numpy array


def main():
    
    train_X, train_y, test_X, test_y = dataPreprocessing() # train, test data should not contain index

    layers=[77,64,32,1]
    activation_functions='sigmoid'
    optimizer='SGD'
    learning_rate=0.05
    n_epoch=42
    
    model = MLPClassifier(layers, activation_functions,optimizer,learning_rate,n_epoch) # remember to change the hyperparameter
    model.fit(train_X, train_y)
    pred = model.predict(test_X,test_y)



    acc = accuracy_score(pred, test_y)
    f1 = f1_score(pred, test_y, zero_division=0)
    mcc = matthews_corrcoef(pred, test_y)

    print(f'Acc: {acc:.5f}')
    print(f'F1 score: {f1:.5f}')
    print(f'MCC: {mcc:.5f}')
    scoring = 0.3 * acc + 0.35 * f1 + 0.35 * mcc
    print(f'Scoring: {scoring:.5f}')


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

