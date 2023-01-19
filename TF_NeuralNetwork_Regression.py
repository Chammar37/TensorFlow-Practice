import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


#House info example 
house_info = tf.constant(["bedroom", "bathroom", "garage"]) #usually numbers but just ex
house_price = tf.constant([939700])


#Predict why from np.array X example

X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])
y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])


X = tf.constant(X) #tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.constant(y)



tf.random.set_seed(42)

#Step 1: Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
### Or we can do this###
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1))
####        #####


#Step 2: Compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

#Step 3: Fit Model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)


#Try and make prediction using our model
y_pred=model.predict([17.0])
print(y_pred)

#### How to improve our model ####
#improve our model by alterating steps we took to 
#create our model
#Create a model - add more layers, 
#               - increase number of hidden units (neurons in hidden layer)
#               - change activation function
# 
#Compiling a model - change optimization function
#                  - change learning rate of optimization function
#Fitting a model - fit a model for more epochs
#                - give model more examples to learn from     


#NEXT IMPROVED MODEL
tf.random.set_seed(42)

#Create model
model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)    
])

#Compile model
model2.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"]
              )

#Fit model
model2.fit(tf.expand_dims(X, axis=-1), y, epochs=102)

#lets see if model prediction has improved by just
#increasing epochs

print("model 2 predict", model2.predict([17.0]))



#NEXT MODEL

#Adam is usually the optimizer to use. lr paramter is
#used for telling the model the rate to step/change
#while the model is learning


model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
])

model3.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["mae"])

model3.fit(tf.expand_dims(X, axis=-1), y, epochs=103)

#lets see if model prediction has improved by changing
#the optimizer to Adam

print("model 3 predict", model3.predict([17.0]))


#NEXT IMPROVED MODEL 

model4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

model4.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# model4.fit(tf.expand_dims(X,axis=-1), y, epochs=104)

# print("model 4 predict", model4.predict([17.0]))

#Above is example of OVERFITTING
#The REAL way we evaluate and identify the performance of 
#our model is the metrics we get from the values we've 
#never seen before, not the metrics from the training data. 

#NEXT IMPROVED MODEL
# Lets try combining our logic from models 3 and 4
#with a twist. Adding layers AND Adam optimizer

model5=tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation=None),
        tf.keras.layers.Dense(1)
])

model5.compile(loss=tf.keras.losses.mae,
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics=["mae"])
               

model5.fit(tf.expand_dims(X, axis=-1), y, epochs=105)

print("model 5 predict", model5.predict([17.0]))
#Whats going on here:
#We are building a model. Compiling it and then fitting it
#with the data. giving input and desired output data, 
#we learn the relationship between the input and what
#the output should be. it learns and then we can use
#that same model to 'predict' as we give new input params

