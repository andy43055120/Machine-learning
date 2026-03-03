import numpy as np
from abc import ABC, abstractmethod
import math


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

    @abstractmethod
    def predict_proba(self, X):
        # Abstract method predict the probability of the dataset X
        pass


class DecisionTreeClassifier:
    def __init__(self, max_depth=1):
        self.tree=None
        self.max_depth = max_depth
        self.remaining_feature=[]
        

    def fit(self, X, y):
        for i in range(len(X[0])):
            self.remaining_feature.append(i)
        self.tree = self._grow_tree(X, y,depth=0)



    def _grow_tree(self, X, y, depth=0):
        if self.is_pure(y) or depth>=self.max_depth or not self.remaining_feature:
            return {'value':self.majority_class(y)} 
        
        best_feature_idx,feature_threshold=self.find_best_split(X,y)

        self.remaining_feature.remove(best_feature_idx)

        X_left,y_left,X_right,y_right=self.split_dataset(X,y,best_feature_idx,feature_threshold)

        
        if len(X_left)==0 or len(X_right)==0:
            return {'value':self.majority_class(y)}
        
        one_count=0
        zero_count=0
        for i in range(len(y)):
            if y[i][0]==1:
                one_count+=1
            else:
                zero_count+=1
        class_counts={0:zero_count,1:one_count}

        left_child=self._grow_tree(X_left,y_left,depth+1)
        right_child=self._grow_tree(X_right,y_right,depth+1)



        return {
            'feature_idx':best_feature_idx,
            'threshold':feature_threshold,
            'left':left_child,
            'right':right_child,
            'class_counts':class_counts
        }

        
        

    # Split dataset based on a feature and threshold
    def split_dataset(self,X, y, best_feature_idx,feature_threshold):
        X_left=[]
        y_left=[]

        X_right=[]
        y_right=[]
        for i in range(len(X)):
            if X[i][best_feature_idx]<feature_threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        X_left=np.array(X_left)
        y_left=np.array(y_left)
        X_right=np.array(X_right)
        y_right=np.array(y_right)

        return X_left,y_left,X_right,y_right



    # Find the best split for the dataset
    def find_best_split(self,X, y):
        num_feature=len(self.remaining_feature)
        num_data=len(X)
        target_feature=0
        target_feature_threshold=0
        
        entropy=0
        true_count=0
        false_count=0

        for i in range(len(y)):
            if y[i][0]==1:
                true_count+=1
            else:
                false_count+=1

        true_probability=true_count/num_data
        false_probability=false_count/num_data

        entropy=-true_probability*math.log(true_probability)-false_probability*math.log(false_probability)

        largest_gain_ratio=-1
        
        for i in range(num_feature):
            left_feature_data_idx=[]
            right_feature_data_idx=[]
            ith_feature_idx=self.remaining_feature[i]
            if ith_feature_idx<=16:
                ith_feature_threshold=self.numeric_ig(X,y,ith_feature_idx)
            else:
                ith_feature_threshold=0.5

            for j in range(num_data): 
                if X[j][ith_feature_idx]<ith_feature_threshold:
                    left_feature_data_idx.append(j)
                else:
                    right_feature_data_idx.append(j)


            left_data_true_count=0
            left_data_false_count=0
            left_entropy=0
            for j in range(len(left_feature_data_idx)):
                if y[left_feature_data_idx[j]][0]==1:
                    left_data_true_count+=1
                else:
                    left_data_false_count+=1
            
            if left_data_true_count==len(left_feature_data_idx) or left_data_true_count==0:
                left_entropy=0
            else:
                left_entropy=-(left_data_true_count/len(left_feature_data_idx))*math.log(left_data_true_count/len(left_feature_data_idx))
                left_entropy-=(left_data_false_count/len(left_feature_data_idx))*math.log(left_data_false_count/len(left_feature_data_idx))

            right_data_true_count=0
            right_data_false_count=0
            right_entropy=0
            for j in range(len(right_feature_data_idx)):
                if y[right_feature_data_idx[j]][0]==1:
                    right_data_true_count+=1
                else:
                    right_data_false_count+=1
            
            if right_data_true_count==len(right_feature_data_idx) or right_data_true_count==0:
                right_entropy=0
            else:
                right_entropy=-(right_data_true_count/len(right_feature_data_idx))*math.log(right_data_true_count/len(right_feature_data_idx))
                right_entropy-=(right_data_false_count/len(right_feature_data_idx))*math.log(right_data_false_count/len(right_feature_data_idx))
        
            rem=(len(right_feature_data_idx)/num_data)*right_entropy+(len(left_feature_data_idx)/num_data)*left_entropy

            ig=entropy-rem

            used_class=[]
            split_info=0
            for j in range(len(X)):
                class_a=X[j][ith_feature_idx]
                if class_a in used_class:
                    continue
                else:
                    used_class.append(class_a)
                    num_class_a=0
                    for k in range(len(X)):
                        if X[k][ith_feature_idx]==class_a:
                            num_class_a+=1
                split_info-=((num_class_a/len(X))*math.log(num_class_a/len(X)))

            if split_info==0:
                gain_ratio=0
            else:
                gain_ratio=ig/split_info

            if gain_ratio>largest_gain_ratio:
                largest_gain_ratio=gain_ratio
                target_feature=self.remaining_feature[i]
                target_feature_threshold=ith_feature_threshold
            
        return target_feature,target_feature_threshold

    def numeric_ig(self,X,y,ith_feature_idx):
        data_count=len(X)
        temp_arr=np.zeros((2,data_count))
        
        for j in range(len(X)):
            temp_arr[0][j]=j
            temp_arr[1][j]=X[j][ith_feature_idx]

        sorted_temp=temp_arr[:,temp_arr[1,:].argsort()]

        #print("sorted temp:",sorted_temp)
        candidate_num=[]
        current_label=y[0][0]
        candidate_idx=[]
        for j in range(len(sorted_temp[0])):
            if y[int(sorted_temp[0][j])][0]!=current_label:
                candidate_num.append((sorted_temp[1][j-1]+sorted_temp[1][j])/2)
                candidate_idx.append(int(sorted_temp[0][j]))
                current_label=y[int(sorted_temp[0][j])][0]

        smallest_rem=1000
        target_threshold=0
        for j in range(len(candidate_num)):
            temp_threshold=candidate_num[j]

            smaller_than_threshold_count=candidate_idx[j]

            smaller_class_true_count=0
            smaller_class_false_count=0
            for k in range(smaller_than_threshold_count):
                if y[int(sorted_temp[0][k])][0]==1:
                    smaller_class_true_count+=1
                else:
                    smaller_class_false_count+=1
            smaller_entropy=0
            if smaller_class_true_count==smaller_than_threshold_count or smaller_class_true_count==0:
                smaller_entropy=0
            else:
                smaller_entropy=-(smaller_class_true_count/smaller_than_threshold_count)*math.log(smaller_class_true_count/smaller_than_threshold_count)
                smaller_entropy-=(smaller_class_false_count/smaller_than_threshold_count)*math.log(smaller_class_false_count/smaller_than_threshold_count)


            larger_than_threshold_count=data_count-smaller_than_threshold_count
            larger_class_true_count=0
            larger_class_false_count=0
            for k in range(smaller_than_threshold_count,smaller_than_threshold_count+larger_than_threshold_count):
                if y[int(sorted_temp[0][k])][0]==1:
                    larger_class_true_count+=1
                else:
                    larger_class_false_count+=1
            larger_entropy=0
            if larger_class_true_count==larger_than_threshold_count or larger_class_true_count==0:
                larger_entropy=0
            else:
                larger_entropy=-(larger_class_true_count/larger_than_threshold_count)*math.log(larger_class_true_count/larger_than_threshold_count)
                larger_entropy-=(larger_class_false_count/larger_than_threshold_count)*math.log(larger_class_false_count/larger_than_threshold_count)
            
            rem=(smaller_than_threshold_count/data_count)*smaller_entropy+(larger_than_threshold_count/data_count)*larger_entropy

            if rem<=smallest_rem:
                smallest_rem=rem
                target_threshold=temp_threshold



        return target_threshold

    def is_pure(self,y):
        target_label=y[0][0]
        for i in range(len(y)):
            if y[i][0]==target_label:
                continue
            else:
                return False
        #print("some node is pure")
        return True # true if all label are the same
    
    def majority_class(self,y):
        true_count=0
        false_count=0
        for i in range(len(y)):
            if y[i][0]==1:
                true_count+=1
            else:
                false_count+=1
            
        if true_count>=false_count:
            return 1
        else:
            return 0

    # prediction
    def predict_proba(self, X):
        node=self.tree
        while 'value' not in node:
            feature_value=X[node['feature_idx']]
            if feature_value<node['threshold']:
                node=node['left']
            else:
                node=node['right']
        return node['value']

    def predict(self, X):
        predictions=[]
        for i in range(len(X)):
            predict_y=self.predict_proba(X[i])
            predictions.append(predict_y)
        
        return predictions



    # print tree
    def print_tree(self,node=None,depth=0, max_print_depth=3):
        if node is None:
            node=self.tree

        class_counts=node.get('class_counts',{'0':0,'1':0})
        count_0,count_1=class_counts.get(0,0),class_counts.get(1,0)

        if 'value' in node:
            print(" " * depth * 4 + f"leaf node with label: {node['value']}")

        # Base case: if we've reached a leaf node or max depth
        if 'value' in node or depth >= max_print_depth:
            return

        # Print the current node's feature and threshold with class distribution
        print(" " * depth * 4 + f"[F{node['feature_idx']}] "
            f"[{count_0} 0 / {count_1} 1]")

        # Recursive call for the left and right children
        if not depth==2:
            print(" " * depth * 4 + "Left:")
            self.print_tree(node['left'], depth + 1, max_print_depth)

            print(" " * depth * 4 + "Right:")
            self.print_tree(node['right'], depth + 1, max_print_depth)

    