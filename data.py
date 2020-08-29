import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
class myTransformer():
    """
    A pipeline class that fit the feature transform and make them ready for training and prediction
    Fix missing values using the function fix_missing()
    Encode Categorical Datatype to an Integer
    Extract Features from Date and generate more Features
    Scale the data using RobustScaler()
    """
    def __init__(self):
        print('Initializing Binarizer......\n')
        # initialize all binarizer variables
        print('Binarizer Ready for Use!!!!!\n')
        
        # initialize the data scaler
        self.dataScaler=RobustScaler()
        print('Scaler is Ready!!')
        
     # the data will need alot of cleaning maybe not so much 
# let get started

    # fix outlier
    def fix_outlier(self, df, column):
        """
        Fix Outlier will take 2 argument
        df = dataframe
        column = column that has an outlier value(s)
        outlier will be replaced by median value of any column 
        return a series of fixed outier
        """
        df[column] = np.where(df[column] > df[column].quantile(0.95),
                                            df[column].median(),
                                            df[column])
        return df[column]
      
        
    def encode_categorical_to_integer(self, data):
        """
        convert or change object datatype into integers
        for modelling
        Function takes 1 arguments 
        data : dataframe that contains column(s) of type object
        columns: a list of columns that are of type object
        the funtion does not return object, it does it computation implicitly
        """
        # get the list of columns that are object data types
        categorical_columns = data.columns[data.dtypes == 'object'].tolist()
        # if a categorical descriptive feature has only 2 levels,
        # define only one binary variable
        for col in categorical_columns:
            n = len(data[col].unique())
            if n == 2:
                data[col] = pd.get_dummies(data[col], drop_first=True)

        # for other categorical features (with n > 2 unique values), use regular 
        # one-hot-encoding
        # if a feature is numeric, it will be untouched
        data = pd.get_dummies(data)
        return data

    # Function to calculate missing values by column
    def missing_values_table(self, df):
        """
        calculate missing values in a dataframe df
        returns: missing values table that comprise of count % of missing and their datatype
        """
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Datatypes of missing values
        mis_val_dtypes = df.dtypes
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtypes], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Data Types'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns, mis_val_table_ren_columns.index

    def fix_missing(self, df, column):
        """
        The Function Fix missing values in the data (df) passed
        df = dataframe that contains the missing columns
        column = columns that has missing values
        """
        for col in column:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('No Data')
        return df
    
    # Fit all the binarisers on the training data
    def fit(self,input_data):
        """ Check For Missing Values and fix it. A preliminary approach to handle 
            missing values.
        """
         # Check Missing
        print('Fixing Missing Values if any\n for int/float column fill with median otherwise No Data')
        _, missing_column = self.missing_values_table(input_data)
        input_data = self.fix_missing(input_data, missing_column)

    def encode_target(self, input_data, target):
        # encode target 
        input_data[target] = input_data[target].replace({'no':0, 'yes':1})
        return input_data[target]
        

    # Transform the input data using the fitted binarisers
    def transform(self, full_dataset, train=False):
        """
        Transformation on data is carried out when this function is called
        Arguments -- 
        full_dataset: data to be transformed
        target: dependent variable/feature or target variable
        steps involve for transformation include 
        1. Copy the original data so all transformation is done on a duplicate data
        2. specify target column and drop from the data
        3. Add some features this part is the feature engineering part can be improved NB only 2 
            features will be added
        4. Map some similar features into one for proper encoding more on this later 
        5. convert object data types column to interger using the One Hot Encoding Techniques
        6. Scale the Data using the robust scaler algorithm: this was choosing because it is less 
            susceptible to outlier even though outlier from the exploratory analysis is not present
            in this data.
        7. Apply Dimensionality Reduction using
        8. Add the target column back to the data
        9. Return the transformed dataframe
        """
        
        # making a copy of the input because we dont want to change the input in the main function
        input_data=full_dataset.copy()
            
        ############################ New Features #################################
        print('Generating Features..\n')
        # bin or split numeric feature age into 3 groups of young middle-aged and old since age is a good target separator
        # this was gotten from the result of analysis
        input_data['age_bin'] = pd.qcut(input_data['age'], q=3,
                                        labels=['young', 'middle-aged', 'old'])
                                        
        # transform age on education and marital status
        input_data['age_per_edu'] = input_data.groupby('education')['age'].transform('mean')
        input_data['age_per_marital'] = input_data.groupby('marital')['age'].transform('mean')
        input_data['age_per_job'] = input_data.groupby('job')['age'].transform('mean')
        input_data['age_per_contact'] = input_data.groupby('contact')['age'].transform('mean')

        # std
        input_data['age_per_edu_std'] = input_data.groupby('education')['age'].transform('std')
        input_data['age_per_marital_std'] = input_data.groupby('marital')['age'].transform('std')
        input_data['age_per_job_std'] = input_data.groupby('job')['age'].transform('std')
        input_data['age_per_contact_std'] = input_data.groupby('contact')['age'].transform('std')

        
        # transform age on education and marital status
        # input_data['duration_per_edu'] = input_data.groupby('education')['duration'].transform('mean')
        # input_data['duration_per_marital'] = input_data.groupby('marital')['duration'].transform('mean')

        # getting the position of the mistalenly labeled 'pdays'
        ind_999 = input_data.loc[(input_data['pdays'] == 999) & (input_data['poutcome'] != 'nonexistent')]['pdays'].index.values

        # Assigning NaNs instead of 999
        input_data.loc[ind_999, 'pdays'] = np.nan
        # drop the nans 
        input_data = input_data.dropna()

        # encode categorical object
        input_data = self.encode_categorical_to_integer(input_data) 
        
        print('Drop the Duration column')
        input_data = input_data.drop('duration', axis=1)
    
        # scale dataframe
        print('Scaling Data using Robust Scaler Method\n')
        input_data = pd.DataFrame(self.dataScaler.fit_transform(input_data),
                                  columns=input_data.columns)
        
        
        print(f'Shape of data is {input_data.shape}')
        print('Done!!!! Pipeline process completed')
        return input_data
    
    def fix_imbalance_data(self, data, target):
        """
        This object is an implementation of SMOTE - Synthetic Minority
        Over-sampling Technique as presented in.
        SMOTE technique is selected to fix the imbalance data because it incoporate some algorithms
        that has proven to be a good fit for this type of problem like SVC - Support Vector Classifier
        & KNearestNeighbour to generate additional data in other to balance the data.
        
        Imbalance data occur in a classification problem: When one class is significantly more than
        or greater than the other class like ratio 80 to 20 or 90 to 10.
        NB: SMOTE works better when combined with undersampling of the majority class. So we first 
        oversample on a 1:10 ration, then undersample majority on a 1:2 ratio.
        return splitted data feature_train, feature_test, target_train, target_test
        """
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        print('Fixing Imbalnced Data..')
        # separate input features and target
        X = data.drop(target, axis=1)
        y = data[target]

        # instantiate the SMOTE model and RandomUnderSampler Module
        over = SMOTE(sampling_strategy='minority')
        under = RandomUnderSampler(sampling_strategy='majority')
        
        # apply the smote model
        X, y = over.fit_resample(X, y)
        X, y = under.fit_resample(X, y)

        # Split data into train and test stratify to make an even split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2001, 
                                                            test_size=.1, stratify = y)
        
        # output shape of train and test data
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
    # apply PCA
    def apply_pca(self, X_train, X_test):
        # import pca
        from sklearn.decomposition import PCA
        # make an instance of pca
        pca = PCA(n_components=12)
        print('Starting Dimensionality Reduction process using PCA')
        # Fit PCA with our standardized data.
        pca.fit(X_train)
        # The attribute shows how much variance is explained by each of the seven individual components.
        print('Explained Variance by Our PCA ::', pca.explained_variance_ratio_)
        # apply the pca to both the train and test set
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        return X_train, X_test
