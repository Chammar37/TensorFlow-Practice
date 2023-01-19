import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#####           EVALUATING OUR MODEL            #####

X = tf.range(-100, 100, 4)

#Make labels for our dataset
#The pattern we want our model to learn
y = X + 10
print(len(X))
print(len(y))

# plt.scatter(X, y)

#### APPLY THE 3 DATA SET METHODOLOGY ######
X_train=X[:40] #First 40 elements are for training 
y_train=y[:40]

X_test=X[40:] #Last elements after first 40 (10) are for testing
y_test=y[40:]

# plt.figure(figsize=(10,7))
# plt.scatter(X_train, y_train, c="b", label="Training Data")
# plt.scatter(X_test, y_test, c="g", label="Test data")

#Looking at our plot, 
#we want our model to learn the relationship in the training data
#SO it can PREDICT our TEST DATA 
#FEEDING X values of our test data, model can predict Y values 

# plt.legend()
# plt.show()

#1. Create Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)] # only 1 input and 1 output 
)

#2. Compile Model
model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"]     
)

#3. Fit Model
# model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100) # read tf.expand_dims documentation. It takes a tensor


#####       VISUALIZE OUR MODEL      #####

#To run model.summary() and visuazlize our model we need to either include model.fit
#or specifying the input_shape in the first layer 

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="inputer_layer1"),
    tf.keras.layers.Dense(1, name="output_layer")
], name="one_of_many_models_we_will_build")

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["mae"]
)

model.summary()

model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

# from tensorflow.keras.util import plot_model
# plot_model(model=model, show_shapes=True)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)


#Lets plot these to compare prediction to ground truth, and judge accuracy 
# y_test or y_true versus y_pred (ground truth vs model's prediction)
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_pred):
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


# plot_predictions()

#### Many different evaluation metrics to evaluate your model's performance ####
# Since we are working on a regression model, the two main metrics are:
#  MAE- mean absoute error, 'on avg, how wrong is each of my model's predictions'
#  MSE- mean square error, 'square the average errors'
#

modelEval=model.evaluate(X_test, y_test) #cause loss and metrics are the same, evaluate will return same figure

print("model evaluate function:", modelEval)

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

print("MAE calculation:", print_mae(y_test, y_pred))

print("MSE calculation:", print_mse(y_test, y_pred))
