import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from model import DecisionTreeClassifier
import os
import math

def k_fold(model_class, X,y,k=4,max_depth_values=[10,11,12,13,14]):
    n_samples=X.shape[0]
    indices=np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size=n_samples//k
    best_score=0
    best_model=None
    best_depth=None

    for max_depth in max_depth_values:
        fold_scores=[]

        for fold in range(k):
            # Create train/validation splits
            val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            # Instantiate and train the model
            model = model_class(max_depth=max_depth)
            model.fit(X_train, y_train)
            
            # Validate and calculate accuracy
            y_pred = model.predict(X_val)
            accuracy = np.mean(y_pred == y_val)
            fold_scores.append(accuracy)


        avg_score = np.mean(fold_scores)
        print(f"Max Depth: {max_depth}, Average Accuracy: {avg_score}")

        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_depth = max_depth

    print(f"Best Depth: {best_depth}, Best Score: {best_score}")
    return best_model, best_depth

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
    train_X,train_y,test_X=dataPreprocessing()

    
    Decision_tree = DecisionTreeClassifier(max_depth=12)
    Decision_tree.fit(train_X,train_y)
    Decision_tree.print_tree()
    #print("finish build tree")
    # TODO 
    # build your decision tree
    # predict the output of the testing data
    # remember to save the predict label as .csv file
    prediction=Decision_tree.predict(test_X)
    prediction=np.array(prediction)
    #print("prediction:",prediction)
    df = pd.DataFrame({
        'label': prediction
    })

    # Save to CSV with the desired format
    df.to_csv('predictions.csv', index=True,index_label='', header=['label'])


    #k_fold(DecisionTreeClassifier,train_X,train_y,k=6,max_depth_values=[10,11,12,13,14,15])




if __name__ == "__main__":
    np.random.seed(0)
    main()
    

