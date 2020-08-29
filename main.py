# import classes
from data import myTransformer
from model import Model
from model import Evaluate
from config import cfg

# import supporting libraries 
import pandas as pd
from sys import argv
import os
import warnings
warnings.filterwarnings('ignore')

class Main:

    """
    imports both the pre-processing and model production code and                     
    automates model production. 

    """

    def __init__(self):
        print('Automation Class')

    def data_loader(self, path):
        if not path is None:
            #If the file exists, then read the existing data from the CSV file.
            if os.path.exists(path):
                print('loading data...')
                df = pd.read_csv(path, sep=';')
                print('Data Loaded')
                print(f'Data has {df.shape[0]} rows and {df.shape[1]} columns')
                print('Starting Preprocessing')

        return df

    def preprocess(self, df):
        # initialize the transformer class
        transformer = myTransformer()
        df['y'] = transformer.encode_target(df, 'y')
        # fit the data to the transformer
        transformer.fit(df)
        # transform the data
        df = transformer.transform(df)
        # fix imbalance
        feature_train, feature_test, target_train, target_test = transformer.fix_imbalance_data(df, target='y')
        # apply dimensionality reduction
        feature_train, feature_test = transformer.apply_pca(feature_train, feature_test)
        print(feature_train.shape, feature_test.shape)
        return feature_train, feature_test, target_train, target_test        


    def train_model(self, feature_train, feature_test, target_train, target_test):
        # initilize the Model building class
        model_class = Model()
        # initialize the evaluation class
        check_performance = Evaluate()
        # train the model class
        if cfg.model == 'random_forest':
            log_model, _, log_prediction = model_class.run_random_forest(feature_train, feature_test, target_train, target_test)
            check_performance.plot_auc_curve(log_model, 'random_forest', feature_test, target_test)
            check_performance.eval_model(target_test, log_prediction)

        elif cfg.model == 'xgb':
            xgb_model, _, xgb_prediction = model_class.run_xgboost(feature_train, feature_test, target_train, target_test)
            check_performance.plot_auc_curve(xgb_model, 'XGBoost', feature_test, target_test)
            check_performance.eval_model(target_test, xgb_prediction)

        elif cfg.model == 'mlp':
            mlp_model, _, mlp_prediction = model_class.run_mlp(feature_train, feature_test, target_train, target_test)
            check_performance.plot_auc_curve(mlp_model, 'MLP', feature_test, target_test)
            check_performance.eval_model(target_test, mlp_prediction)

        elif cfg.model == 'grb':
            grb_model, _, grb_prediction = model_class.run_gradient_boost(feature_train, feature_test, target_train, target_test)
            check_performance.plot_auc_curve(grb_model, 'Gradient Boosting', feature_test, target_test)
            check_performance.eval_model(target_test, grb_prediction)

        elif cfg.model == 'svm':
            _, _, svm_prediction = model_class.run_svc(feature_train, feature_test, target_train, target_test)
            # check_performance.plot_auc_curve(svc_model, 'SVC', feature_test, target_test)
            check_performance.eval_model(target_test, svm_prediction)
        
        elif cfg.model == 'log-reg':
            log_model, _, log_prediction = model_class.run_logistic_regression(feature_train, feature_test, target_train, target_test)
            check_performance.plot_auc_curve(log_model, 'Logistic Regression', feature_test, target_test)
            check_performance.eval_model(target_test, log_prediction)
            

        # if cfg.predict == '':
        #     check_performance.plot_auc_curve(svc_model, 'SVC', feature_test, target_test)
        #     check_performance.eval_model(target_test, svm_prediction)          
            
    def automate(self, csvfile):
        df = self.data_loader(csvfile)
        feature_train, feature_test, target_train, target_test = self.preprocess(df)
        self.train_model(feature_train, feature_test, target_train, target_test)


auto = Main()
auto.automate(cfg.data)
