import tensorflow 
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
import zipfile
import h5py


#from scipy.stats import norm
#import matplotlib.pyplot as plt

image_size = 28 #this is size of mnist images, want to be 250 later
image_shape = (image_size, image_size, 1)
latent_dim = 2 #will want this to be bigger later

xX_train = np.zeros(shape=(2000,image_size,image_size)) #a list of numpy matrices for every neighborhood
xX_test = np.zeros(shape=(682, image_size, image_size))

# with zipfile.ZipFile('/content/gdrive/My Drive/Violetta + Jackie Unite/Data/asciix5_1.zip', 'r') as myzip:
#     with myzip.open('asciix5_1/ascii_x5_1_0.tif.txt', 'r') as myfile:
#         print(str(myfile.read()).replace(r'\r\n', ''))

def make_matrix_from_data(data):
    index = data.index("-9999") + len("-9999\r\n") + 2 #not best way but works soooooo
    #print(data[index:])
    out = []
    col = 0
    start = index
    for row in range(0, image_size):
        sub_out = []
        col = 0
        while(col < image_size):
            if(index >= len(data) or data[index] == " "):
                try:
                    sub_out.append(float(data[start:index+1]))
                    # if(float(data[start:index+1]) > my_max):
                    #     my_max = float(data[start:index+1])
                except:
                    sub_out.append(0.0)
                start = index + 1
                col += 1
            index += 1
        # print("I got through an entire row! " + str(row))
        out.append(sub_out)
    return(out)

def make_data():
    prefix = 'asciix5_1/ascii_x5_1_'
    suffix = '.tif.txt'
    found = False
    i = 0 
    index = 0
    with zipfile.ZipFile('/Users/jackielin/Dropbox (MIT)/2019S/4.S42/Final Project/Data/asciix5_1.zip', 'r') as myzip:
        #check if this number is correct!
        while(i <= 3500): #this is specific to this dataset and and the number of the last file in the set  
            #found = False
            file_name = prefix + str(i) + suffix
            try: #we have to do this because the way the files were saved there are not all consecutive (but they are in order)
                with myzip.open(file_name, 'r') as myfile:
                    data = str(myfile.read()).replace(r'\r\n', '')
                    found = True
            except:
                found = False
            if(found):
                #if(i == 0):
                    #print(data)
                    #print(len(data))
                    #print(data.index("-9999") + len("-9999\r\n") + 2)
                neighborhood = make_matrix_from_data(data)
                a = np.array(neighborhood)
                # print(a.shape)
                if(index >= 2000):
                    xX_test[index - 2000] = a
                else:
                    
                    xX_train[index] = a

                index += 1 
            i = i + 1
    X_train = (xX_train/1776).reshape(xX_train.shape+(1,))
    X_test = (xX_test/1776).reshape(xX_test.shape+(1,))
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

#DATA CLEANING ***************************************************************************************************

# print(xX_test[0].shape)

X_train, X_test = make_data()

# print(X_train[0])

#VAE ***************************************************************************************************
input_img = keras.Input(shape=image_shape)
x = layers.Conv2D(32,3,padding="same", activation="relu")(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu", strides=(2,2))(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu")(input_img)
x = layers.Conv2D(64,3,padding="same", activation="relu")(input_img)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x) # flatten to a vector
x = layers.Dense(32,activation="relu")(x)

#OTHER THING **************************************************
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

#decoder_y = CustomVariationalLayer()([decoder_input,input_img])
decoder_y = CustomVariationalLayer()([input_img,x])

#decoder = Model(decoder_input,x) # decoder model which takes decoder_input and turns it into image
decoder = Model(decoder_input,decoder_y) 


#z_decoded = decoder(z) # apply decoder to sampled latent vector from encoder: our models are now chained

decoder.summary() #not sure if this will run properly 

#y = CustomVariationalLayer()([input_img,z_decoded]) # call the custom layer on the input and decoded output to obtain final model output
#vae = Model(input_img,y) # Final model

#vae.compile(optimizer='rmsprop', loss = None)
#print(vae.fit(x=X_train,y=None, epochs=3, validation_data=(X_test,None)))

encoded_X_train = encoder.predict(X_train)
encoded_X_test = encoder.predict(X_test)

decoder.compile(optimizer='rmsprop', loss = None)
print(decoder.fit(x=encoded_X_train,y=X_train, epochs=3, validation_data=(encoded_X_test,X_test)))

#vae.save("models/vae_1.h5")

#TESTING **************************************************************************************************

input_image = (X_train[0]).reshape((image_size,image_size))
dec_input = encoder.predict(input_image)
#output_image = (decoder.predict([X_train[0].reshape(-1, image_size,image_size, 1)])).reshape((image_size,image_size))
output_image = (decoder.predict(dec_input))

print("This is an example image: ", input_image)
print("This is an example prediction: ", output_image)

# img_1 = Image.fromarray(input_image)
# img_1.save("output/input_1.png")
# img_1.show()


np.savetxt("output/input_1.npy", input_image, fmt="%f")
np.savetxt("output/output_1.npy", output_image, fmt="%f")

# input_image.save("output/input_1.npy")
# output_image.save("output/output_1.npy")

# np.save('g:\test.csv', input_image)


# print("This is an example image: ", X_train[0])
# print("This is an example prediction: ", (vae.predict([X_train[0].reshape(-1, image_size,image_size, 1)])).reshape((image_size,image_size))


