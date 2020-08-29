from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--model', type=str, default='log_reg',
                help='Baseline Model to train any of the following {XGBoost, Gradient Boost, SVM, Multilayer Perceptron}')
parser.add_argument('--predict', type=str, help='Test Data to predict')
parser.add_argument('--data', type=str, default='../data/bank-additional-full.csv', help='test?')

cfg = parser.parse_args()