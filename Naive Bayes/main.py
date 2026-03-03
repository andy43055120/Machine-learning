import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import os

def k_fold(model, X,y,k_folds=5):
    indices=np.arange(len(X))
    np.random.shuffle(indices)
    X,y=X[indices],y[indices]
    fold_size=len(X)//k_folds

    avg_acc=0
    sum_acc=0

    for i in range(k_folds):
        start,end=i*fold_size,(i+1)*fold_size
        X_test,y_test=X[start:end],y[start:end]
        X_train=np.concatenate([X[:start],X[end:]],axis=0)
        y_train=np.concatenate([y[:start],y[end:]],axis=0)

        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)

        correct_cnt=0
        for j in range(len(y_test)):
            if y_pred[j]==y_test[j]:
                correct_cnt+=1

        acc=correct_cnt/len(y_test)

        sum_acc+=acc
    avg_acc=sum_acc/k_folds

    return avg_acc

def dataPreprocessing():
    """ TODO, use your own dataPreprocess function here. """
    root_path = r"C:\Users\user\Desktop\introduction to machine learing\hw3"
    
    train_X_path = os.path.join(root_path, "train_x.csv")
    train_y_path = os.path.join(root_path, "train_y.csv")
    test_X_path = os.path.join(root_path, "test_x.csv")

    raw_train_X=pd.read_csv(train_X_path)
    raw_train_Y=pd.read_csv(train_y_path)
    raw_test_X=pd.read_csv(test_X_path)

    raw_train_X.drop(raw_train_X.columns[0], axis=1, inplace=True)
    raw_train_Y.drop(raw_train_Y.columns[0], axis=1, inplace=True)
    raw_test_X.drop(raw_test_X.columns[0], axis=1, inplace=True)

    train_preprocessor=Preprocessor(raw_train_X)
    test_preprocessor=Preprocessor(raw_test_X)

    train_X=train_preprocessor.preprocess()
    test_X=test_preprocessor.preprocess()


    train_X_np=train_X.to_numpy()
    train_Y_np=raw_train_Y.to_numpy()
    test_X_np=test_X.to_numpy()


    return train_X_np, train_Y_np, test_X_np



def main():
    train_X,list_train_y,test_X=dataPreprocessing()
    train_y=[]
    for i in range(len(list_train_y)):
        train_y.append(list_train_y[i][0])
    train_y=np.array(train_y)


    model = NaiveBayesClassifier()
    model.fit(train_X,train_y)



    predictions=model.predict(test_X)

    df = pd.DataFrame({
        'label': predictions
    })

    # Save to CSV with the desired format
    df.to_csv('predictions.csv', index=True,index_label='', header=['label'])
    

    #acc=k_fold(model,train_X,train_y,k_folds=5)
    #print("acc:",acc)


    # TODO 
    # build your NB model
    # predict the output of the testing data
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

