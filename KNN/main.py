import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import KNearestNeighborClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import os
import matplotlib.pyplot as plt

def k_fold(model, X,y,k_folds=5,feature_list=[]):
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

        model.fit(X_train,y_train,useful_feature=feature_list)
        y_pred=model.predict(X_test)

        correct_cnt=0
        for j in range(len(y_test)):
            if y_pred[j]==y_test[j][0]:
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

def data_selection(X,y):
    correlation_list=[[],[]]
    for i in range(len(X[0])):
        correlation_list[0].append(i)
        feature=[]
        for j in range(len(X)):
            feature.append(X[j][i])
    
        correlation_list[1].append(calculate_correlation(feature,y))

    combine=list(zip(correlation_list[0],correlation_list[1]))
    sorted_combine=sorted(combine,key=lambda x: x[1],reverse=True)
    correlation_list[0],correlation_list[1]=zip(*sorted_combine)
    correlation_list[0]=list(correlation_list[0])
    correlation_list[1]=list(correlation_list[1])

    use_feature_idx=[]
    correlation_threshold=0.1
    for i in range(len(correlation_list[0])):
        if correlation_list[1][i][0]>correlation_threshold:
            use_feature_idx.append(correlation_list[0][i])
        else:
            break
    #print("correlation:",correlation_list)
    return use_feature_idx

def calculate_correlation(feature,target):
    mean_x=sum(feature)/len(feature)
    mean_y=sum(target)/len(target)

    covariance=sum((x-mean_x)*(y-mean_y) for x,y in zip(feature,target))

    variance_x=sum((x-mean_x)**2 for x in feature)
    variance_y=sum((y-mean_y)**2 for y in target)

    correlation=covariance/(np.sqrt(variance_x)*np.sqrt(variance_y))
    return correlation



def main():
    train_X,train_y,test_X=dataPreprocessing()
    useful_feature=[]
    for i in range(77):
        useful_feature.append(i)
    useful_feature=data_selection(train_X,train_y)
    useful_feature=sorted(useful_feature)
    #print('useful feature:',useful_feature)
    model = KNearestNeighborClassifier(7,'euclidean')

    model.fit(train_X,train_y,useful_feature)

    pred=model.predict(test_X)
    '''
    error_rate=[]
    for i in range(1,11):
        print("iter:",i)
        model = KNearestNeighborClassifier(i,'euclidean')

        model.fit(train_X,train_y,useful_feature)

        #pred=model.predict(test_X)

        acc=k_fold(model,train_X,train_y,k_folds=9,feature_list=useful_feature)
        print("acc:",acc)

        error_rate.append(1-acc)
    print('error_rate',error_rate)
    
    k_values=list(range(1,11))
    plt.figure(figsize=(8, 6))  # Optional: Adjust the size of the plot
    plt.plot(k_values, error_rate, marker='o', color='b', linestyle='-', label='Error Rate')
    plt.title('Error Rate vs. k in kNN', fontsize=14)
    plt.xlabel('k (Number of Neighbors)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.xticks(k_values)  # Ensure we have ticks for k=1 to 10
    plt.grid(True)
    plt.legend()
    plt.show()
    '''
    #print("prediction:",pred)
    df = pd.DataFrame({
        'label': pred
    })

    # Save to CSV with the desired format
    df.to_csv('predictions.csv', index=True,index_label='', header=['label'])
    

    # TODO 
    # build your KNN model
    # predict the output of the testing data
    # remember to save the predict label as .csv file


if __name__ == "__main__":
    np.random.seed(0)
    main()
    

