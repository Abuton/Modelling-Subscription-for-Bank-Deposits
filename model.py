from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle
        
class Model:
    """
    A class for Algorithms implementations and model building.
    contains all functions and classes to generate your three to five                       
    models.
    """
    
    def __init__(self):
        print('Model Class Ready')
        
            ######################## LOGISTIC REGRSSION #######################################
    def run_logistic_regression(self, X_train, X_val, Y_train, Y_val):

        """
        Logistic Model
        """
        
        model_name = '26-08-2020-20-32-31-00-log-reg.pkl'
        # initialze the kfold
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=221), list()
        # split data index to train and test
        for train, test in kfold.split(X_train):
            # specify train and test sets
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]
            
            # initialize the model
            model = LogisticRegression(random_state=27,  solver='lbfgs')
            # train
            model.fit(x_train, y_train)
            # predict for evaluation
            preds = model.predict(x_test)
            # compute f1-score
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_pred
    
        ######################## XGBOOST #######################################
    def run_xgboost(self, X_train, X_val, Y_train, Y_val):
        model_name = '26-08-2020-20-32-31-00-xgboost.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=21), list()
        pred_tot_cb = []
        print('Using 5-fold Cross Validation to fit an XGB model')
        for train, test in kfold.split(X_train):     
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = XGBClassifier(random_state=27,  n_estimators=2000,  learning_rate=0.04,
                             early_stopping_rounds=100, objective='binary:logistic')
            eval_set  = [(x_train,y_train), (x_test,y_test)]
            model.fit(x_train, y_train, eval_set=eval_set, eval_metric="auc", verbose=500)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            pred_tot_cb.append(test_pred)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_pred
        
        ######################## MULTILAYER PERCEPTRON #######################################
    def run_mlp(self, X_train, X_val, Y_train, Y_val):
        model_name = '26-08-2020-20-32-31-00-mlp.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=11), list()
        pred_tot_cb = []
        for train, test in kfold.split(X_train): 
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = MLPClassifier(hidden_layer_sizes=(200,))
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            pred_tot_cb.append(test_pred)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_pred
    
        ######################## SUPPORT VECTOR MACHINE #######################################
    def run_svc(self, X_train, X_val, Y_train, Y_val, kernal='poly'):
        from sklearn.svm import SVC
        model_name = '26-08-2020-20-32-31-00-svc.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=12), list()
        pred_tot_cb = []
        for train, test in kfold.split(X_train): 
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = SVC(kernel=kernal)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            pred_tot_cb.append(test_pred)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_pred
    
    ######################## DECISION TREE #######################################
    def run_decision_tree(self, X_train, X_val, Y_train, Y_val):
        from sklearn.tree import DecisionTreeClassifier
        model_name = '26-08-2020-20-32-31-00-tree.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=221), list()
        pred_tot_cb = []
        for train, test in kfold.split(X_train): 
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = DecisionTreeClassifier(max_depth=4)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            pred_tot_cb.append(test_pred)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_pred
    
    ##################### RANDOM FOREST CLASSIFIER ##############################
    def run_random_forest(self, X_train, X_val, Y_train, Y_val):
        from sklearn.ensemble import RandomForestClassifier
        model_name = '26-08-2020-20-32-31-00-ran-reg.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=285), list()
        pred_tot_cb = []
        for train, test in kfold.split(X_train): 
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = RandomForestClassifier(n_estimators=300)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_pred = model.predict(X_val)
            pred_tot_cb.append(test_pred)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, preds
    
    ##################### GRADIENT BOOSTING CLASSIFIER ##############################
    def run_gradient_boost(self, X_train, X_val, Y_train, Y_val):
        from sklearn.ensemble import GradientBoostingClassifier
        model_name = '26-08-2020-20-32-31-00-grb.pkl'
        kfold, scores = KFold(n_splits=5, shuffle=True, random_state=210), list()
        pred_tot_cb = []
        for train, test in kfold.split(X_train): 
            x_train, x_test = X_train[train], X_train[test]
            y_train, y_test = Y_train[train], Y_train[test]

            model = GradientBoostingClassifier(learning_rate=0.1, max_depth=5)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            score = f1_score(y_test, preds)
            scores.append(score)
            test_prediction = model.predict(X_val)
            pred_tot_cb.append(test_prediction)
            print('f1-score: ',score)
        print("Average: ", sum(scores)/len(scores))

        # save model
        pickle.dump(model, open(model_name, 'wb'))
        print(f'Model Saved {model_name}')
        return model, scores, test_prediction

###################### CLASS EVALUATION ######################

class Evaluate:
    """
    This class is responsible for all the necessary evaluation 
    the eval_model() takes 2 argument actual_values and the model_predictions
    it returns/prints the Accuracy Score, F1-Score, Recall and Precision
    it prints the confusion matrix as well as the classification report
    The Selected metric for this data is the Recall and ROC curve
    plot_auc_curve handles the plotting of the ROC curve with the model_name
    embedded into it as a label.
    """
    def __init__(self):
        print('Model Class for Evaluation')
        
    def eval_model(self, target_test_data, prediction):
        from sklearn.metrics import accuracy_score,confusion_matrix,recall_score, f1_score, precision_score
        from sklearn.metrics import classification_report
        confusion_matrix = confusion_matrix(target_test_data, prediction)
        print('Accuracy Score: ', accuracy_score(target_test_data, prediction))
        print('F1-Score: ', f1_score(target_test_data, prediction))
        print('Recall: ', recall_score(target_test_data, prediction))
        print('Precision: ', precision_score(target_test_data, prediction))
        print(confusion_matrix)
        print(classification_report(target_test_data, prediction))

    def plot_auc_curve(self, model, model_name, test_data, target_test_data):
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve

        # Visualisation
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        import seaborn as sns

        # Configure visualisations
        mpl.style.use( 'fivethirtyeight' )
        sns.set_style( 'white' )
        pylab.rcParams[ 'figure.figsize' ] = 12 , 7

        logit_roc_auc = roc_auc_score(target_test_data, model.predict(test_data))

        fpr, tpr, _ = roc_curve(target_test_data, model.predict_proba(test_data)[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (area under curve = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic ({model_name})')
        plt.legend(loc="lower right")
        plt.savefig(f'{model_name}_ROC')
        plt.show()