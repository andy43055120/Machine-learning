# preprocessor.py
import numpy as np


class Preprocessor:
    """ TODO """ 
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

        self.numeric_cols=self.df.columns[:17] # f1 to f17 are numeric
        self.categorical_cols=self.df.columns[17:] # f18 to f77 are categorical
        self.missing_indices_dict={}

    def impute_numeric(self): # impute missing values for numeric features with the mean.
        for col in self.numeric_cols:
            if self.df[col].isnull().any():
                missing_indices=self.df[self.df[col].isnull()].index.tolist()
                self.missing_indices_dict[col]=missing_indices
                mean_value=self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_value)


    def impute_categorical(self): # impute missing values for categorical features with the most frequent value
        for col in self.categorical_cols:
            if self.df[col].dtype == 'object':  
                self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})

        for col in self.categorical_cols:
            if self.df[col].isnull().any():
                most_frequent_value=self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(most_frequent_value)
        

    def mice(self):
        for col,indices in self.missing_indices_dict.items():
            for idx in indices:
                b_df=self.df.drop(columns=[col])
                cols_to_keep=b_df.columns[:16]
                result_row=b_df.loc[idx,cols_to_keep]
                B=result_row.values
                temp_df=self.df.drop(columns=[col])
                temp_df=temp_df.drop(index=idx)
                last_col='F17'
                if last_col in temp_df.columns:
                    last_col='F17'
                else:
                    last_col='F16'

                last_col_index=temp_df.columns.get_loc(last_col)
 
                matrix_2d = temp_df.iloc[:, :last_col_index + 1].values

                column_data=self.df[col].drop(index=idx).values

                X=matrix_2d
                X_t=X.T
                X_t_X=np.dot(X_t,X)
                np.fill_diagonal(X_t_X, X_t_X.diagonal() + 1)
                X_t_X_inv=np.linalg.inv(X_t_X)
                X_t_b=np.dot(X_t,column_data)
                
                beta=np.dot(X_t_X_inv,X_t_b)

                result=np.sum(B*beta)

                self.df.at[idx,col]=result
               

    def standardize(self):
        for col in self.numeric_cols:
            mean_value = self.df[col].mean()
            std_dev = self.df[col].std()
            self.df[col] = (self.df[col] - mean_value) / (std_dev)
        for col in self.categorical_cols:
            mean_value = self.df[col].mean()
            std_dev = self.df[col].std()
            self.df[col] = (self.df[col] - mean_value) / (std_dev)


    def preprocess(self):
        self.impute_numeric()
        self.impute_categorical()
        self.mice()
        self.standardize()
        return self.df


