import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.range(-100, 100, 4)

y = X+10

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

def plot_predictions(predictions, train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test):
    """ PLots trainnig data, test data and compares predictions to ground truth labels"""

    plt.figure(figsize=(10,7))
    #Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label= "Training Data")
    #Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing Data")
    #Plot model's predictions in red
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()

def print_mae(y_test, y_pred):
    
    #Calculate MAE
    mae=tf.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred) # not in same shape
    mae=tf.metrics.mean_absolute_error(y_true=y_test, y_pred=tf.squeeze(y_pred)) 
    
    #Check shapes
    print("y_test shape",y_test.shape)
    print("y_pred shape",y_pred.shape)

    #print correct MAE
    return mae

def print_mse(y_test, y_pred):
    #Calculate the MSE
    mse=tf.metrics.mean_squared_error(y_true=y_test, y_pred=tf.squeeze(y_pred))
    #Print mse
    return mse