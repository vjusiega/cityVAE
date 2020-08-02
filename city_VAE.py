import tensorflow 
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# IMPORTS FOR LHS **************************
from pyDOE import lhs
import pandas as pd
import os
from os import path
from time import time


image_size = 100 #this is size of mnist images, want to be 250 later
image_shape = (image_size, image_size, 1)
latent_dim = 5 #will want this to be bigger later

def load_numpy_data(train_filename, test_filename):
    """
    This function loads the data and splits it into training data and testing data
    Inputs: filename is a saved numpy array of the data in current dir. test_filename is not necessary
    Returns: X_train, X_test
    """
    data = np.loadtxt(train_filename).reshape((8968,image_size,image_size))
    train = data[:7600].reshape((7600,image_size,image_size))
    test = data[7600:].reshape((1368,image_size,image_size))

    X_train = (train/1776).reshape(train.shape+(1,))
    X_test = (test/1776).reshape(test.shape+(1,))
    return (X_train, X_test)

         
#Renaud's function
def sampling(args):
    z_mean, z_log_var = args
    # we're sampling an epsilon value from a normal distribution 
    # (K.shape(z_mean)[0] is the batch size)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.) 
    # with this epsilon, we can sample from the actual distribution
    return z_mean + K.exp(z_log_var)*epsilon  

#Renaud's code
class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x,z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x,z_decoded) # this the reconstruction part of the loss
        kl_loss = -5e-4 * K.mean(1+z_log_var - K.square(z_mean)-K.exp(z_log_var), axis=-1) # this is the regularization part of the loss
        return K.mean(xent_loss+kl_loss)

    def call(self,inputs):
        x=inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x,z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

#DATA LOADING & CLEANING ***************************************************************************************************

train_filename = 'raw_data/data1004/train1004.npy'
test_filename = 'raw_data/data1004/test1004.npy'
X_train, X_test = load_numpy_data(train_filename, test_filename) # load training data and test data

#ENCODER ***************************************************************************************************
input_img = keras.Input(shape=image_shape) #initiallize encoder 
x = layers.Conv2D(32,3,padding="same", activation="relu")(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu", strides=(2,2))(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu")(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu")(input_img)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x) # flatten to a vector
x = layers.Dense(32,activation="relu")(x)

#LINK BETWEEN ENCODER & DECODER -> LATENT VECTOR *********************************************************************
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# wrap the function in custom layer
z = layers.Lambda(sampling)([z_mean,z_log_var])

encoder = Model(input_img, z)
encoder.summary()


#DECODER **************************************************************************************************
decoder_input = layers.Input(K.int_shape(encoder.layers[-1].output)[1:]) #K.int_shape(z)[1:] is the input shape (K.int_shape(z)[0] si the batch size)
x = layers.Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input) # Upsampling | np.prod returns the product of numbers in an array > here returns the array size needed
x = layers.Reshape(shape_before_flattening[1:])(x)
x = layers.Conv2DTranspose(32,3, padding='same', activation='relu', strides=(2,2))(x) # also called deconvolution
x = layers.Conv2D(1,3,padding='same',activation='sigmoid', strides=2)(x)


decoder = Model(decoder_input,x) # decoder model which takes decoder_input and turns it into image

z_decoded = decoder(z) # apply decoder to sampled latent vector from encoder: our models are now chained

decoder.summary()

#VAE ************************************************************************************************

y = CustomVariationalLayer()([input_img,z_decoded]) # call the custom layer on the input and decoded output to obtain final model output
vae = Model(input_img,y) # Final model

#FIT & TRAIN VAE************************************************************************************************

class PlotLatentSpace(keras.callbacks.Callback):    # plots latent space
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}): # NOTE: This function does not work if the latent_dim > 2
        n = 10 # number of images per row
        z = norm.ppf(np.linspace(0.05, 0.95, n))
        Z1,Z2,Z3,Z4,Z5= np.meshgrid(z,z,z,z,z)
        Z = np.vstack([Z1.flatten(),Z2.flatten(),Z3.flatten(),Z4.flatten(),Z5.flatten()]).T
        print(Z.shape)
        ims = decoder.predict(Z)
        fig, axes = plt.subplots(n,n,figsize=(15,15))
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        count = 0
        for k in range(0,ims.shape[0]):
            i,j = int(count/n), int(count%n)
            axes[i,j].imshow(ims[k,:,:,0])
            axes[i,j].axis("off")
            count +=1
        plt.show()

plot_space = PlotLatentSpace()


vae.compile(optimizer='rmsprop', loss = None)
print(vae.fit(x=X_train,y=None, epochs=4, validation_data=(X_test,None)))
# print(vae.fit(x=X_train,y=None, epochs=4, validation_data=(X_test,None), callbacks=[plot_space]))

#vae.save("models/vae_1.h5")


# PERFORMANCE METRICS FOR PCA ***********************************************************************************
def compute_standard_deviation(np_array):
    return np.std(np_array, axis=1)

def average_building_height(data):
    ahhh = np.true_divide(data.sum(1),(data!=0).sum(1))/np.count_nonzero(data, axis=1)
    return ahhh

def open_space(data):
    non_zero = np.count_nonzero(data, axis=1)
    return non_zero*-1+(image_size*image_size)

#LATIN HYPERCUBE SAMPLING ******************************************************************************************

def sample_latent_space(num_samples):
    n = num_samples # number of samples from the latent space
    sample = lhs(latent_dim, n) # latin hypercube sampling
    sample = sample-0.5
    sample = sample * np.ptp(norm.ppf(np.linspace(0.05, 0.95))) # do arithmetic on nparray to fit latent space range

    results = np.zeros(shape=(n, image_size**2))

    for i in range(n): # for each latent vector sample, store decoder output
        z1,z2,z3,z4,z5 = sample[i]
        z1 = float(z1)
        z2 = float(z2)
        z3 = float(z3)
        z4 = float(z4)
        z5 = float(z5)
        Z = np.array([z1,z2,z3,z4,z5]).reshape((1,5))
        r = decoder.predict(Z) #decoder output from random latent vector Z
        results[i] = r.reshape((image_size**2))

        if i == 0: # export the first random sampling, and two other nearby latent vectors
            # export the first random sampling
            np.savetxt("output/output1004/z_original.txt", Z, delimiter=" ", fmt="%f")
            np.savetxt("output/output1004/results_original.txt", r.reshape((image_size,image_size)), delimiter=" ", fmt="%f")
            # first nearby latent vector 
            Z1 = Z 
            Z[0][0] = Z[0][0] + 0.2 # positive small step in variable z1
            Z[0][3] = Z[0][3] + 0.1 # positive small step in variable z4
            r1 = decoder.predict(Z)
            np.savetxt("output/output1004/z1.txt", Z, delimiter=" ", fmt="%f")
            np.savetxt("output/output1004/results_1.txt", r1.reshape((image_size,image_size)), delimiter=" ", fmt="%f")
            # second nearby latent vector 
            Z1[0][2] = Z1[0][2] - 0.1 # negative small step in variable z3
            Z1[0][4] = Z1[0][4] + 0.2 # negative small stelp in variabl z5
            r2 = decoder.predict(Z1)
            np.savetxt("output/output1004/z2.txt", Z1, delimiter=" ", fmt="%f")
            np.savetxt("output/output1004/results_2.txt", r2.reshape((image_size,image_size)), delimiter=" ", fmt="%f")

    # export 
    ex1_input = (X_train[0]).reshape((image_size,image_size))
    ex1_output = (vae.predict([X_train[0].reshape(-1, image_size,image_size, 1)])).reshape((image_size,image_size))
    np.savetxt("output/output1004/ex1_input.txt", ex1_input, delimiter=" ", fmt="%s")
    np.savetxt("output/output1004/ex1_output.txt", ex1_output.reshape((image_size,image_size)), delimiter=" ", fmt="%s")



    y_avg_height = average_building_height(results)
    y_std_height = compute_standard_deviation(results)
    y_openspace = open_space(results)
    # print("std: ")
    # print(y_std_height.shape)

    if not os.path.exists("data1004"):
        os.makedirs("data1004")
    df = pd.DataFrame({'z1':sample[:,0], 'z2':sample[:,1], 'z3':sample[:,2], 'z4':sample[:,3], 'z5':sample[:,4], 'y_avg':y_avg_height, 'y_std':y_std_height, 'y_opn':y_openspace})

    df.to_pickle("data1004/samples") # save data 
    print("saved data")
    df.to_csv("data1004/samples.csv") # save data to csv (optional)

num_samples = 50
sample_latent_space(num_samples)


#TESTING **************************************************************************************************

# input_image = (X_train[0]).reshape((image_size,image_size))
# # dec_input = encoder.predict(input_image.reshape(-1, image_size, image_size,1))
# # #output_image = (decoder.predict([X_train[0].reshape(-1, image_size,image_size, 1)])).reshape((image_size,image_size))
# output_image = decoder.predict(input_image.reshape((-1, image_size, image_size, 1)))

# print("This is an example image: ", input_image)
# print("This is an example prediction: ", output_image)

# # img_1 = Image.fromarray(input_image)
# # img_1.save("output/output1004/input_1.png")
# # img_1.show()


# np.savetxt("output/output1004/input_1.npy", input_image, fmt="%f")
# np.savetxt("output/output1004/output_1.npy", output_image, fmt="%f")

# input_image.save("output/output1004/input_1.npy")
# output_image.save("output/output1004/output_1.npy")

# np.save('g:\test.csv', input_image)


# print("This is an example image: ", X_train[0])
# print("This is an example prediction: ", (vae.predict([X_train[0].reshape(-1, image_size,image_size, 1)])).reshape((image_size,image_size))


