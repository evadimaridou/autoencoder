import numpy as np
import matplotlib.pyplot as plt
import keras
from itertools import cycle
import time

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
    train_data = train_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)

    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    train_data = train_data / 255   
    test_data = test_data / 255

    return (train_data, train_labels), (test_data, test_labels)

def show_mnist_picture(picture):
    adjusted_pic = picture.reshape(28, 28)
    plt.imshow(adjusted_pic, cmap = 'gray')
    plt.show()


def seperate_digits(data, labels):
    my_data = [[] for _ in range(10)] 
    for index, sample in enumerate(data):
        my_data[labels[index]].append(sample)

    return my_data

def expected_output(data):
    expected_nums = []

    for index, _ in enumerate(data):
        if (index != len(data) - 1):
            expected_nums.append(data[index + 1][0])
        else:
            expected_nums.append(data[0][0])

    return expected_nums


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
            delta = delta * relu_deriv(self.pre_activation[index])   #we dont have the input layer weights, it begins from the layers that have fault
            delta_list.append(delta)
            index -= 1
            
        
        delta_list.reverse()
        self.delta = delta_list

    def plot_accur_loss(self, epochs):

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
    
    def train_network(self, train_data, test_data, second_digits, epoch_count, eta):

        
        for epoch in range(epoch_count):
            start_time = time.time()
            loss = 0
            print(f"Epoch {epoch + 1}")
            
            self.second_digits = second_digits
    
            pairs = []

            for i in range(10):
                pairs.append(cycle(train_data[i]))

            length = len(train_data[0])
            for index in range(1, 10):
                if (length < len(train_data[index])):
                    length = len(train_data[index])

            length = 10 * length
            for i in range(length):
                num_pair = i % 10
                first_digit = next(pairs[num_pair])
                
                second_digit = self.second_digits[num_pair]

                self.feed_forward(first_digit)
                self.compute_error_o(second_digit)

                #compute MSE
                loss += np.mean(np.square(self.error_o))/2

                self.back_propagation()
                self.gradient_descent(first_digit, eta)
            
            end_time = time.time()

            
            elapsed_time = end_time - start_time

            
            print(f"Elapsed Time: {elapsed_time} seconds")

            num = length
                  
            self.train_loss.append(loss/num)
            print(f"Training loss is {loss/num}")    
            self.test_network(test_data, "val")

    def test_network(self, test_data, keyword):
        loss = 0
        
        for index, number in enumerate(test_data):
            for first_digit in number:
                i = 4
                second_digit = self.second_digits[index]
                self.feed_forward(first_digit)
                self.compute_error_o(second_digit)

                #compute MSE
                loss += np.mean(np.square(self.error_o))/2
                if(keyword == "test" and i > 0):
                    i -= 1
                    self.items.append(((first_digit, second_digit), self.post_activation[self.hid_num]))

        num = 0
        for i in range(10):            
            num += len(test_data[i])

        if(keyword == "val"):
            self.val_loss.append(loss/num)
            print(f"Validation loss is {loss/num}")
            

if __name__ == "__main__":

    (train_data, train_labels), (test_data, test_labels) = load_mnist() #keras

    train_data = seperate_digits(train_data, train_labels)
    test_data = seperate_digits(test_data, test_labels) 

    second_digits = expected_output(train_data)

    #sample size
    input_size = train_data[0][0].shape[0]

    epoch_count = 20
    layers = [256, 128, 64, 128, 256]
    eta = 1e-3

    print(layers)
    print(eta)

    my_autoencoder = autoencoder(layers, input_size = input_size)
    my_autoencoder.train_network(train_data, test_data, second_digits, epoch_count = epoch_count, eta = eta)
    my_autoencoder.test_network( test_data, "test")
    my_autoencoder.plot_accur_loss(epoch_count)


    for i, item in enumerate(my_autoencoder.items):
        first_digit = item[0][0]
        show_mnist_picture(first_digit)

        second_digit = item[0][1]
        show_mnist_picture(second_digit)

        constructed_second = item[1]
        show_mnist_picture(constructed_second)