import tensorflow as tf
import os, getpass
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from TF_Fundamentals_Functions import plot_predictions
from TF_Fundamentals_Functions import print_mae
from TF_Fundamentals_Functions import print_mse


#Starting at video 55

####        This class will focus on how we can tweak models to improve     #####
# DONE THROUGH EXPERIMENTATION.. 
# How do we experiment? With the following
# 1.Get more data - more examples, more opportunities to learn patterns and relationships
# between features and labels
# #####
# 2.Make your model larger - more complex model (more layers, more hidden units in layer)
# ####
# 3. Train for longer - give your model more of a chance to find patterns in data 

print("Get user", getpass.getuser())
print("Get os user", os.getuid())

X = tf.range(-100, 100, 4)

y = X+10

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

tf.random.set_seed(42)

                        ##### MODEL 1 #####
                        # 1 layer, 100 epochs
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, name="first_layer"),
    tf.keras.layers.Dense(1, name="last_layer")
],name="model1")

model1.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"]
)

model1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

#Predict and Plot
y_pred1 = model1.predict(X_test)
# plot_predictions(predictions=y_pred1)

#Calculate MAE of model 1 
mae1 = print_mae(y_test, y_pred1)
print("MAE of model1:", mae1.numpy())





                        ##### MODEL 2 #####
                        # 2 layers, 100 epochs #
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
],name="model2")

model2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

#Predit and Plot
y_pred2 = model2.predict(X_test)
# plot_predictions(predictions=y_pred2)

#Calculate mae of model2
mae2 = print_mae(y_test, y_pred2)
print("MAE of model2:", mae2.numpy())





                                ##### MODEL 3 #####
                                # 2 layers, 500 epochs 

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
], name="model3")

model3.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.SGD(),
               metrics=["mae"] 
)

model3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0)

#Predict and plot predictions
y_pred3 = model3.predict(X_test)
# plot_predictions(predictions=y_pred3)

#Calculate mae of model3
mae3 = print_mae(y_test, y_pred3)
print("MAE of model3:", mae3.numpy())





        #############           Compare Results of Experiemnts          #############

model_results = [['model1', mae1.numpy()],
                 ['model2', mae2.numpy()],
                 ['model3', mae3.numpy()] ]


all_results = pd.DataFrame(model_results, columns=["model", "mae"])
print(all_results)


print("The model with the best results (lowest mae)")
#Weird thing happening here when printing results. model.summary prints then
#prints None 
print(model2.summary())

## TRACKING RESULTS OF EXPERIMENTS ##
#Many tools to help us track results:
# - TensorBoard - help track modelling experiments
#       https://www.tensorflow.org/tensorboard/get_started
# - Weights & Biases - tool for tracking all kinds of ML experiemnts (plugs into TensorBoard)
#       https://wandb.ai/site/experiment-tracking
# 


                    ####### SAVING OUR MODELS #######

model2.save("best_model") 
