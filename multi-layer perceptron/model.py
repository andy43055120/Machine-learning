import numpy as np
from abc import ABC, abstractmethod

# ====== Activation funtion ====== #
# this is a placeholder class meant to implement an activation function later,
# such as sigmoid, relu, or tanh.
# Activativation functions are non-linear functions that decide whether 
# a neruron should be activated or not
class activation():

    # TODO
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    


    @staticmethod
    def relu(x):
        max_num=-10000
        min_num=10000

        for i in range(len(x[0])):
            if x[0][i]>max_num:
                max_num=x[0][i]
                continue
            if x[0][i]<min_num:
                min_num=x[0][i]
        for i in range(len(x[0])):
            x[0][i]=((x[0][i]-min_num)/(max_num-min_num))
        return np.maximum(0, x)



    @staticmethod
    def tanh(x):
        max_num=-10000
        min_num=10000

        for i in range(len(x[0])):
            if x[0][i]>max_num:
                max_num=x[0][i]
                continue
            if x[0][i]<min_num:
                min_num=x[0][i]
        for i in range(len(x[0])):
            x[0][i]=((x[0][i]-min_num)/(max_num-min_num))*(20)-10


        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    pass
    


# ====== Optimizer function ====== #
# this is a placeholder class, used to implement an optimization algorithm
# such as gradient descent. Optimizers are responsible for adjusting
# the weights of the model to minimize the loss function.
class optimizer():
    @staticmethod
    def SGD(self,gradients):
        for i in range(len(self.biases)):#3
            for j in range(len(self.biases[i][0])):
                self.biases[i][0][j]+=self.learning_rate*gradients[i][j]

        for i in range(len(self.weights)):#3 0-2
            for row in range(len(self.weights[i])):#77 0-76
                for idx in range(len(self.weights[i][row])):#64 0-63
                    if i==0:
                        self.weights[i][row][idx]+=self.learning_rate*gradients[i][idx]*self.activations[i][row]
                    else:
                        self.weights[i][row][idx]+=self.learning_rate*gradients[i][idx]*self.activations[i][0][row]
        #print("self.weight:",self.weights[0][0][0])
        pass
    @staticmethod
    def AdaGrad(self,gradients):
        for i in range(len(self.biases)):#3
            for j in range(len(self.biases[i][0])):
                self.biases[i][0][j]+=self.learning_rate*gradients[i][j]

        for i in range(len(self.weights)):#3 0-2
            for row in range(len(self.weights[i])):#77 0-76
                for idx in range(len(self.weights[i][row])):#64 0-63
                    self.weights[i][row][idx]+=(self.learning_rate/((self.G[i][row][idx]+0.00001)**0.5))*gradients[i][idx]
        #print("self.weight:",self.weights[0][0][0])

    @staticmethod
    def RMSProp(self,gradients):
        for i in range(len(self.biases)):#3
            for j in range(len(self.biases[i][0])):
                self.biases[i][0][j]+=self.learning_rate*gradients[i][j]

        for i in range(len(self.weights)):#3 0-2
            for row in range(len(self.weights[i])):#77 0-76
                for idx in range(len(self.weights[i][row])):#64 0-63
                    self.weights[i][row][idx]+=(self.learning_rate/((self.prop_now[i][row][idx]+0.00001)**0.5))*gradients[i][idx]
        #print("self.weight:",self.weights[0][0][0])

        pass


    


# Base classifier class
# this is a base class for classifiers, defined using the Abstract Base Class(ABC)
# It serves as a blue print that ensure any subclass (e.g., MLPClassifier)
# implement these  methods
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

# this class represents a multi-layer perceptron classifier,
# a type of neural network.
# it inherits from the Classifier class and implements all the required method.
class MLPClassifier(Classifier):
    def __init__(self, layers, activate_function, optimizer, learning_rate, n_epoch = 3):
        """ TODO, Initialize your own MLP class """

        self.layers = layers
        self.activate_function = activate_function
        self.optimizer = optimizer
        self.learning_rate =learning_rate
        self.n_epoch = n_epoch
        self.weight_gradients=[]
        self.G=[] # store the accumulated sum of squared gradient
        self.first_round=True
        self.prop_now=[]
        self.prop_pre=[]
        self.epison=1e-8
    
    # calculating the output of each layer useing the inputs and weights.
    def forwardPass(self, X):
        """ Forward pass of MLP """
        # TODO

        self.z_values=[] # store the weighted input
        self.activations=[X] # store the activations layer by layer


        for i in range(len(self.weights)):
            Z=np.dot(self.activations[-1],self.weights[i])+self.biases[i]
            self.z_values.append(Z)
            #print("z:",Z)
            if self.activate_function=="sigmoid":
                A=activation.sigmoid(Z)
            elif self.activate_function=="relu":
                A=activation.relu(Z)
            elif self.activate_function=="tanh":
                A=activation.tanh(Z)
            #print("A:",A)

            self.activations.append(A)

        return self.activations[-1] #return output of the last layer 


        pass
    
    # calculate the error gradients and proagates them backward through the network
    # to adjust the weight
    def backwardPass(self, target_y):
        """ Backward pass of MLP """
        # TODO


        gradients=[]

        for i in range(1,4):
            gradients.append(self.activations[i][0])


        last_output=self.activations[-1][0][0]
        gradients[2][0]=last_output*(1-last_output)*(target_y[0]-last_output)
        
        for i in range(1,-1,-1):
            for j in range(len(gradients[i])):
                output=self.activations[i+1][0][j]
                sum_downstream=0
                for k in range(len(gradients[i+1])):
                    sum_downstream+=(self.weights[i+1][j][k])*gradients[i+1][k]
                gradients[i][j]=output*(1-output)*sum_downstream


        for i in range(3):
            self.weight_gradients.append([])
            for j in range(len(self.weights[i])):
                temp_arr=[]
                if i==0:
                    output=self.activations[i][j]
                else:
                    output=self.activations[i][0][j]
                for k in range(len(self.weights[i][0])):#64, 32
                    temp_arr.append(output*gradients[i][k])
                self.weight_gradients[i].append(temp_arr)

        for i in range(3):
            for row in range(len(self.weight_gradients[i])):
                for entry in range(len(self.weight_gradients[i][row])):
                    self.G[i][row][entry]+=self.weight_gradients[i][row][entry]*self.weight_gradients[i][row][entry]
                    if self.first_round:
                        self.prop_now[i][row][entry]=self.weight_gradients[i][row][entry]*self.weight_gradients[i][row][entry]
                        self.first_round=False
                    else:
                        self.prop_now[i][row][entry]=0.1*(self.weight_gradients[i][row][entry]*self.weight_gradients[i][row][entry])+0.9*(self.prop_pre[i][row][entry])
                        
                    #print("g:",self.G[i][row][entry])
                    
                

        return gradients
        pass

    # use optimizer to update the weights based on the gradinets
    # computed during backpropagation.
    def update(self,gradients):
        """ The update method to update parameters """
        # TODO
        if self.optimizer=="SGD":
            optimizer.SGD(self,gradients)
        elif self.optimizer=="Adagrad":
            optimizer.AdaGrad(self,gradients)
        elif self.optimizer=="RMSProp":
            optimizer.RMSProp(self,gradients)
        

                    
        pass
    
    # involve running the forward pass, computing the loss, performing backpropagation
    # and update the weight for a number of epoch
    def fit(self, X_train, y_train):
        """ Fit method for MLP, call it to train your MLP model """
        # TODO

        self.weights=[]
        self.biases=[]

        for i in range(len(self.layers)-1):
            weight_matrix=np.random.randn(self.layers[i],self.layers[i+1])
            bias_vector=np.zeros((1,self.layers[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        
        y_hat=[]
        

        for i in range(3):
            self.G.append([])
            self.prop_now.append([])
            self.prop_pre.append([])
            for j in range(len(self.weights[i])):
                temp_arr=[]
                for k in range(len(self.weights[i][0])):
                    temp_arr.append(0)
                self.G[i].append(temp_arr)
                self.prop_now[i].append(temp_arr)
                self.prop_pre[i].append(temp_arr)

        for epoch in range(self.n_epoch):
            for j in range(len(X_train.T[0])):

                y_hat.append(self.forwardPass(X_train[j]))
                self.weight_gradients=[]
                
                gradients=self.backwardPass(y_train[j])
                
                # Update weights and biases
                self.update(gradients)
                self.prop_pre=self.prop_now
            #print("bias 0 length:",len(self.biases[0][0]))
            #print("bias:",len(self.biases[1][0]))
            #print("gradient:",gradients[0][0])
            #print("weight:",self.weights[0][0][0])
            #print("activation:",self.activations[0][0])
            # Print loss at intervals
            #print("y_hat:",y_hat)
            #print("y_hat y train")
            #for i in len(y_hat):
            #    print(f"{y_hat[i]}, {y_train[i]}")
            loss=np.mean((y_hat-y_train))
            
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            

        pass

    # predict the class labels for the test data, 
    # it used the output of predict_proba and converts them into
    # binary class labels (1 or 0 based on the threshold of 0.5)
    def predict(self, X_test,Y_test):
        """ Method for predicting class of the testing data """
        #print("weight:",self.weights)
        #print("weight shape:",len(self.weights[0][0]))
        y_hat = self.predict_proba(X_test)
        loss=np.mean((y_hat-Y_test))
        print("loss in test:",loss)

        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    

    
    def predict_proba(self, X_test):
        """ Method for predicting the probability of the testing data """
        #print("Xtest:",X_test[0])


        return self.forwardPass(X_test)


    