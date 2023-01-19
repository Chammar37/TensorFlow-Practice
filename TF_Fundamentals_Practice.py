import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

from tensorflow import keras
from keras import layers, models, datasets

random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))

# print(random_1)

tf.random.shuffle(random_1, seed=48)

# print(random_1)

numpy_A = np.arange(1,25, dtype=np.int32)

# print(numpy_A)

# A = tf.constant(numpy_A, shape=(2,3,4))
A = tf.constant(numpy_A, shape=(2,6,2))
B = tf.constant(numpy_A)
print("Print matrix A", A)

rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
C = tf.zeros(shape=[2, 3, 4])
D = tf.zeros(shape=[2, 3, 4, 5, 6])
# print("printing C\n", C)
# print("printing rank 4 tensor\n", rank_4_tensor)
# print("printing D\n", D)

print("Shape of our tensor C: ", rank_4_tensor.shape)
print("Number of dimensions in our tensor C: ", rank_4_tensor.ndim)
# print("Total number of elements in our tensor C: ", tf.size(C))
print("Total number of elements in our tensor C: ", tf.size(rank_4_tensor).numpy())
print("Datatype of every element in tensor C: ", rank_4_tensor.dtype)

#Get the first 2 elemtns of each dimensions
# print(rank_4_tensor[:2, :2, :2, :2])


#More practice indexing 
# print(rank_4_tensor[:1, :1, :1, :]) # or [:1, :1, :1]
# print(rank_4_tensor[:1, :1, :, :1]) # dont know how to write this in words lol
# print(rank_4_tensor[:1, :, :1, :1]) # Just look at full tensor
# print(rank_4_tensor[:, :1, :1, :1]) # : w/o number means get all. :1 means first index

# print(rank_4_tensor.ndim)


#For matrix multiplcation to work:
#
# 1. The inner dimensions must match shape(4,2) <-- shape(3,5) /// 2 != 3
# 2. The resulting matrix has the shape of the outer dimensions (4,5)

print("\n\n\n matrix multiplcation\n\n\n") 

matrixA = tf.constant(np.arange(1,7), shape=(3,2))
matrixB = tf.constant(np.arange(7,13), shape=(3,2))
print(matrixA)
try:
    tf.matmul(matrixA, matrixB) #error
except:
    print("it failed because the shapes do not align with matrix mult rule")


print("\nMatrix being multiplied by: \n", matrixA)
reshapeM = tf.reshape(matrixB, shape=(2,3))
transposeM = tf.transpose(matrixB)
print("\nOriginal Matrix (before edits): \n", matrixB)

print("\nreshapeM Matrix: \n", reshapeM)

print("\ntransposeM Matrix: \n", transposeM)

print("\nPRINTING MATRIX MULTIPLICATION\n")
print("printing reshapeM tf.matmult matrixA", tf.matmul(reshapeM, matrixA))
print("printing transposeM tf.matmult matrixA", tf.matmul(transposeM, matrixA))


#Transpose flips axes
#Use transpose to get the correct shape for matrix multiplication

print("Tensor dot function with transposed matrixB by matrix\n", tf.tensordot(transposeM, matrixA, axes=1))

#Change data type

print("Matrix A data type:", matrixA.dtype)
castA = tf.cast(matrixA, dtype=tf.float16) #lowering dtype from int64 to float16 
print("Matrix A data type after cast:", castA.dtype)


#Aggregating tensors = condensing multiple values down to a smaller amount of values

print(tf.reduce_max(A))
print(tf.reduce_min(A))
print(tf.reduce_mean(A))
print(tf.reduce_sum(A))

print("Variance is:",tf.math.reduce_variance(tf.cast(A, dtype=tf.float32)))
print("Standard deviation is:",tf.math.reduce_std(tf.cast(A, dtype=tf.float32)))

F = tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
print(F)

print("Position (index) of max element: ", tf.argmax(F))
print("Print max elmnt now that we know the index: ", F[tf.argmax(F)])

print("for position (index) of min:", tf.argmin(F), "\nmin value is: ", F[tf.argmin(F)])

# Squeezing tensor (removing all single dimensions)
tf.random.set_seed(42)
G = tf.constant(tf.random.uniform(shape=[50]), shape=(1,1,1,1,50))
print(G.shape)
G_squeezed = tf.squeeze(G)
print(G_squeezed.shape)

#One-hot encoding tensors

some_list=[0, 1, 2, 3] # could be red, green, blue, purple
print("Print one-hot encoding:", tf.one_hot(some_list, depth=4))
#can change 'on-value' and 'off-value' paramteres


H = tf.range(1,10)
print("Square tensor:",tf.square(H))

#Other functions: 
# tf.sqrt (int32 dtype wont work)
# tf.log (int32 dtype wont work)

#Practicing with NumPy compatibility 
J = tf.constant(np.array([3., 7., 10.]))
#convert numpy array to tensor^

#convert back by passing tensor:
np.array(J)
J.numpy()

numpy_J = tf.constant(np.array([3., 7., 10.]))
tensor_J = tf.constant([3., 7., 10.])
print(numpy_J.dtype, " data type vs ", tensor_J.dtype)

#How to optimize tensor operations 

print(tf.config.list_physical_devices())
#https://www.tensorflow.org/guide/gpu

#Sample image recognition code from chatGPT 

#Simple tensorflow code for image recognition using CNN

# DATADIR = "C:/Users/Siddharth/Desktop/Python/Tensorflow/PetImages"
# CATEGORIES = ["Dog", "Cat"]
# IMG_SIZE = 50

# training_data = []

# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 training_data.append([new_array, class_num])
#             except Exception as e:
#                 pass

# create_training_data()

# random.shuffle(training_data)

# X = []
# y = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# pickle_out = open("X.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)

# pickle_in = open("y.pickle", "rb")
# y = pickle.load(pickle_in)

# X = X/255.0

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = X.shape[1:]))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

# model.add(tf.keras.layers.Conv2D(64, (3,3)))
# model.add(tf.keras.layers.Activation("relu"))
# model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64))

# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Activation("sigmoid"))

# model.compile(loss = "binary_crossentropy",
#               optimizer = "adam",
#               metrics = ["accuracy"])

# model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.1)