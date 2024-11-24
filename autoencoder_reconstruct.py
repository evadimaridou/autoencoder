import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import keras
import pandas as pd
import sklearn.datasets, sklearn.decomposition
import time

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
    train_data = train_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)

    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    train_data = train_data / 255   
    test_data = test_data / 255
    return (train_data, test_data)

def add_gaussian_noise(image, mean=0, std=0.1):

    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise

    # Clip values to ensure they are within valid image range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

def pca_mnist(data):

    mu = np.mean(data, axis=0)

    pca = sklearn.decomposition.PCA()

    pca.fit(data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = explained_variance_ratio.cumsum()
    n_components = np.argmax(cumulative_variance_ratio >= 0.75) + 1
    #print(n_components)

    nComp = n_components
    Xhat = np.dot(pca.transform(data)[:,:nComp], pca.components_[:nComp,:])
    Xhat += mu

    # Save the PCA reconstructed using np.save
    #np.save('reconstructed_PCA_data_75.npy', Xhat)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return 1.0 * (x > 0)


class layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        #weights are property of the layer before
        self.weights = self.initialize_weights()    

        #biases are property of the layer before
        scale = np.sqrt(2.0 / self.input_size)
        self.bias = np.random.randn(self.output_size) * scale  

    #he initialization better for relu
    def initialize_weights(self):    #the weights[0][1] refers to the connnection of first element of output and second of input               
        scale = np.sqrt(2.0 / self.input_size)
        weights = np.random.randn(self.output_size, self.input_size) * scale
    
        return weights       
    
class autoencoder:
    def __init__(self, hid_layers, input_size):
        self.encoder = []
        self.decoder = []
        #initialize the input layer (connects the input to h1)
        self.first_layer = layer(input_size, hid_layers[0])

        self.output_size = input_size
        self.hid_num = len(hid_layers)

        self.bottle_neck = int(self.hid_num / 2) - 1
        
        self.encoder_num = int(self.hid_num / 2) #without input layer
        self.decoder_num = self.hid_num - self.encoder_num

        for i in range(self.encoder_num):
            self.encoder.append(layer(hid_layers[i], hid_layers[i + 1]))

        for i in range(self.encoder_num, self.hid_num - 1):
            self.decoder.append(layer(hid_layers[i], hid_layers[i + 1]))

        #initialize the last layer
        self.decoder.append(layer(hid_layers[self.hid_num - 1], self.output_size))    
        
        self.train_loss = []
        self.train_accur = []

        self.val_loss = []
        self.val_accur = []

        self.items = []
        self.reconstructed = []
        self.trained_model = []

        
    def feed_forward(self, data):  
        post_activation = []
        pre_activation = []

        my_pre_activation = np.dot(self.first_layer.weights, data) + self.first_layer.bias
        pre_activation.append(my_pre_activation)
        post_activation.append(relu(my_pre_activation))

        index = 0        
        for _, layer in enumerate(self.encoder):   
            my_pre_activation = np.dot(layer.weights, post_activation[index]) + layer.bias
            pre_activation.append(my_pre_activation)
            if index == self.bottle_neck:
                post_activation.append(my_pre_activation)
            else:
                post_activation.append(relu(my_pre_activation))
            index += 1

        for _, layer in enumerate(self.decoder):   
            my_pre_activation = np.dot(layer.weights, post_activation[index]) + layer.bias
            pre_activation.append(my_pre_activation)
            if index == self.hid_num - 1:
                post_activation.append(sigmoid(my_pre_activation))
            else:
                post_activation.append(relu(my_pre_activation))
            index += 1


        self.pre_activation = pre_activation
        self.post_activation = post_activation

    def compute_error_o(self, target):
        self.error_o = target - self.post_activation[self.hid_num]
       
    def back_propagation(self):        #compute delta
        delta_list = []
        delta = self.error_o * sigmoid_deriv(self.pre_activation[self.hid_num])  #this is the delta of output layer
        delta_list.append(delta)

        index = self.hid_num - 1
        for _, layer in reversed(list(enumerate(self.decoder))):
            if index == self.bottle_neck + 1:
                delta = np.dot(delta, layer.weights)
            else:
                delta = np.dot(delta, layer.weights)
                delta = delta * relu_deriv(self.pre_activation[index])
            delta_list.append(delta)
            index -= 1
            

        for _, layer in reversed(list(enumerate(self.encoder))):
            delta = np.dot(delta, layer.weights)
            delta = delta * relu_deriv(self.pre_activation[index])
            delta_list.append(delta)
            index -= 1
            
        
        delta_list.reverse()
        self.delta = delta_list

    def plot_loss(self, epochs):

        epochs = [i+1 for i in range(epochs)]

        plt.figure(1)
        plt.plot(epochs, self.train_loss, label = "Training Loss", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_loss, label = "Validation Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        # Show the plots
        plt.show()
        return 
    
    def gradient_descent(self, train_data, eta):    #update weights and biases

        index = self.hid_num - 1
        for _, layer in reversed(list(enumerate(self.decoder))):
            layer.weights += eta * np.dot(self.delta[index + 1].reshape(-1, 1), self.post_activation[index].reshape(1, -1))
            layer.bias += eta * self.delta[index + 1]
            index -= 1

        for _, layer in reversed(list(enumerate(self.encoder))):
            layer.weights += eta * np.dot(self.delta[index + 1].reshape(-1, 1), self.post_activation[index].reshape(1, -1))
            layer.bias += eta * self.delta[index + 1]
            index -= 1

        self.first_layer.weights += eta * np.dot(self.delta[0].reshape(-1, 1), train_data.reshape(1, -1))
        self.first_layer.bias += eta * self.delta[0]
    
    def train_network(self, train_data, val_data, epoch_count, eta):
        for epoch in range(epoch_count):
            start_time = time.time()
            loss = 0
            print(f"Epoch {epoch + 1} ")
            
            for sample in range(train_data.shape[0]):
                self.feed_forward(train_data[sample])
                self.compute_error_o(train_data[sample])

                #compute MSE
                loss += np.mean(np.square(self.error_o))/2

                self.back_propagation()
                self.gradient_descent(train_data[sample], eta)
           
            end_time = time.time()

           
            elapsed_time = end_time - start_time

            
            print(f"Elapsed Time: {elapsed_time} seconds")

            print(f"Training loss is {loss/train_data.shape[0]}")       
            self.train_loss.append(loss/train_data.shape[0])
            print(loss/train_data.shape[0])    
            self.test_network(val_data, "val")

    def test_network(self, data, keyword):
        loss = 0

        for sample in range(data.shape[0]):
            
            self.feed_forward(data[sample])  
            self.compute_error_o(data[sample])

            #compute MSE
            loss += np.mean(np.square(self.error_o))/2

            if (keyword == "test"):
                self.items.append((data[sample], self.post_activation[self.hid_num]))
                self.reconstructed.append(self.post_activation[self.hid_num])

                self.trained_model = []
                self.trained_model.append((self.first_layer.weights, self.first_layer.bias))
                for layer in self.encoder:
                    self.trained_model.append((layer.weights, layer.bias))
                for layer in self.decoder:
                    self.trained_model.append((layer.weights, layer.bias))

        
        if(keyword == "val"):
            print(f"Validation loss is {loss/data.shape[0]}")  
            self.val_loss.append(loss/data.shape[0])
            

if __name__ == "__main__":
    
    (train_data, test_data) = load_mnist() #keras

    #sample size
    input_size = train_data.shape[1]

    epoch_count = 10
    layers = [128, 64, 36, 18, 9, 18, 36, 64, 128]
    eta = 1e-3

    print(layers)
    print(eta)

    my_autoencoder = autoencoder(layers, input_size = input_size)
    my_autoencoder.train_network(train_data, test_data, epoch_count = epoch_count, eta = eta)
    my_autoencoder.test_network(test_data, "test")
    my_autoencoder.plot_loss(epoch_count)

    reconstructed_data = np.array(my_autoencoder.reconstructed)

    np.save('reconstructed_data_2.npy', reconstructed_data)

    original_reconstructed_data = np.array(my_autoencoder.items)

    np.save('original_reconstructed_data.npy', original_reconstructed_data)

    # Load numpy array from the date
    loaded_reconstructed = np.load('mnist_data.npy')

    with open("trained_model", "wb") as fp:   #Pickling
            pkl.dump(my_autoencoder.trained_model, fp)

    with open('trained_model','rb') as f:
        trained_model = pd.read_pickle(r'trained_model')


    for i, item in enumerate(my_autoencoder.items):
        image = item[0].reshape(-1, 28, 28)
        reconstructed = item[1].reshape(-1, 28, 28)
        plt.imshow(image[0], cmap = 'gray')
        plt.show()
        plt.imshow(reconstructed[0], cmap = 'gray')
        plt.show()
    