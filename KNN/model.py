import numpy as np
from abc import ABC, abstractmethod


# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass



# K-Nearest Neighbors Classifier
class KNearestNeighborClassifier(Classifier):
    def __init__(self, k=3, distance_metric='euclidean', p=3): 
        self.k=k
        if distance_metric=='manhattan':
            self.p=1
        elif distance_metric=='euclidean':
            self.p=2
        elif distance_metric=='chebyshev':
            self.p=0 #just be the indicator
        self.X_train=None
        self.y_train=None
        self.use_feature=None
        pass

    def fit(self, X, y,useful_feature): 
        self.X_train=X
        self.y_train=y
        self.use_feature=useful_feature
        pass

    def predict(self, X):
        y_pred = [self._predict(x) for x in X] 
        return np.array(y_pred)

    def _predict(self, x):
        k_nearest_neighbors_idx=self.get_neighbors(x)
        class_0_cnt=0
        class_1_cnt=0
        for i in range(len(k_nearest_neighbors_idx)):
            if self.y_train[k_nearest_neighbors_idx[i]][0]==0:
                class_0_cnt+=1
            else:
                class_1_cnt+=1
        if class_0_cnt>class_1_cnt:
            return 0
        else:
            return 1


    def get_neighbors(self,x):
        neighbors_list=[[],[]]
        for i in range(len(self.X_train)):# 0~425
            if self.p==0:
                neighbors_list[0].append(i)
                max_distance=-1
                for j in range(len(self.X_train[i])):
                    if not j in self.use_feature:
                        continue
                    if abs(x[j]-self.X_train[i][j])>max_distance:
                        max_distance=abs(x[j]-self.X_train[i][j])
                neighbors_list[1].append(max_distance)
                
                #print('max distance',max_distance)
            else:
                neighbors_list[0].append(i)
                sum_distance=0
                for j in range(len(self.X_train[i])):
                    if not j in self.use_feature:
                        continue
                    sum_distance+=pow(abs(x[j]-self.X_train[i][j]),self.p)
                sum_distance=pow(sum_distance,1/self.p)
                neighbors_list[1].append(sum_distance)

        combined=list(zip(neighbors_list[0],neighbors_list[1]))
        sorted_combined=sorted(combined,key=lambda x: x[1])
        neighbors_list[0],neighbors_list[1]=zip(*sorted_combined)
        neighbors_list[0] = list(neighbors_list[0])
        neighbors_list[1] = list(neighbors_list[1])

        k_nearest_neighbor_idx=[]
        for i in range(self.k) :
            k_nearest_neighbor_idx.append(neighbors_list[0][i])
        
        return k_nearest_neighbor_idx