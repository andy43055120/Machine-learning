import numpy as np
from abc import ABC, abstractmethod
from math import pi, exp


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


# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {} # P(Class)
        self.likelihoods = {} # P(Feature|Class)


    def calculate_prior(self, y):
        total_samples=len(y)
        unique_classes,class_counts=np.unique(y,return_counts=True)
        for cls, count in zip(unique_classes,class_counts):
            self.priors[cls]=count/total_samples
        

    def calculate_likelihood(self, X, y):
        classes=np.unique(y)
        n_features=X.shape[1]

        for cls in classes:
            self.likelihoods[cls]={}
            X_class=X[y==cls]
            for feature_idx in range(n_features):
                feature_values=X_class[:,feature_idx]
                unique_bins,counts=np.unique(feature_values,return_counts=True)

                total_count=len(feature_values)
                self.likelihoods[cls][feature_idx]={
                    bin_value:count/total_count
                    for bin_value, count in zip(unique_bins,counts)
                }



    def fit(self, X, y):
        self.calculate_prior(y)
        self.calculate_likelihood(X,y)


    def predict(self, X):
        predictions = []
        epsilon = 1e-9

        for sample in X:
            posteriors = {}

            for cls, priors in self.priors.items():
                posterior = np.log(priors) 

                for feature_idx, bin_value in enumerate(sample):
                    likelihood = self.likelihoods[cls][feature_idx].get(bin_value, epsilon)
                    posterior += np.log(likelihood)

                posteriors[cls] = posterior

            best_class = max(posteriors, key=posteriors.get)
            predictions.append(best_class)

        return predictions

    def predict_proba(self, X):
        pass