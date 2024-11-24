import numpy as np
import matplotlib.pyplot as plt
import keras
import time


def one_hot_encoding(a):
    b = np.zeros((len(a), max(a) + 1))
    b[np.arange(len(a)), a] = 1
    return b

def one_hot_encoding_num(a):
    b = np.zeros(10)
    b[a] = 1
    return b

#returns 1*output_size array, all probabilities sum to 1
def softmax(x):
    #preventing large numbers from going to exp
    exp_x = np.exp(x - np.max(x))
    sum_x = np.sum(exp_x)

    return exp_x/sum_x

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
    train_data = train_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)

    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    train_data = train_data / 255   
    test_data = test_data / 255
    return (train_data, train_labels), (test_data, test_labels)


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
        self.encoder_input = layer(input_size, hid_layers[0])

        self.output_size = input_size
        self.hid_num = len(hid_layers)

        self.bottle_neck = int(self.hid_num / 2) - 1
        
        self.encoder_num = int(self.hid_num / 2) #without input layer
        self.decoder_num = self.hid_num - self.encoder_num

        for i in range(self.encoder_num):
            self.encoder.append(layer(hid_layers[i], hid_layers[i + 1]))

        self.decoder_input_layer = layer(hid_layers[self.encoder_num], hid_layers[self.encoder_num + 1])
        for i in range(self.encoder_num + 1, self.hid_num - 1):
            self.decoder.append(layer(hid_layers[i], hid_layers[i + 1]))

        #initialize the last layer
        self.decoder.append(layer(hid_layers[self.hid_num - 1], self.output_size))    
        
        self.train_encoder_loss = []
        self.train_encoder_accuracy = []
        self.train_decoder_loss = []

        self.val_encoder_loss = []
        self.val_encoder_accuracy = []
        self.val_decoder_loss = []

        self.items = []
        
    def feed_forward_encoder(self, data):  
        post_activation = []
        pre_activation = []

        my_pre_activation = np.dot(self.encoder_input.weights, data) + self.encoder_input.bias
        pre_activation.append(my_pre_activation)
        my_post_activation = relu(my_pre_activation)
        post_activation.append(my_post_activation)

        for index, layer in enumerate(self.encoder):   
            my_pre_activation = np.dot(layer.weights, post_activation[index]) + layer.bias
            pre_activation.append(my_pre_activation)
            if index == self.bottle_neck:
                my_post_activation = softmax(my_pre_activation)
                post_activation.append(my_post_activation)
            else:
                my_post_activation = relu(my_pre_activation)
                post_activation.append(my_post_activation)
        
        self.encoder_output = my_post_activation
        

        self.pre_activ_encod = pre_activation
        self.post_activ_encod = post_activation
    def feed_forward_decoder(self, decoder_input):
        self.decoder_input = decoder_input
        post_activation = []
        pre_activation = []

        my_pre_activation = np.dot(self.decoder_input_layer.weights, decoder_input) + self.decoder_input_layer.bias
        pre_activation.append(my_pre_activation)
        my_post_activation = relu(my_pre_activation)
        post_activation.append(my_post_activation)

        for index, layer in enumerate(self.decoder):   
            my_pre_activation = np.dot(layer.weights, post_activation[index]) + layer.bias
            pre_activation.append(my_pre_activation)
            if index == len(self.decoder) - 1:
                my_post_activation = sigmoid(my_pre_activation)
                post_activation.append(my_post_activation)
            else:
                my_post_activation = relu(my_pre_activation)
                post_activation.append(my_post_activation)

        self.decoder_output = my_post_activation

        self.pre_activ_decod = pre_activation
        self.post_activ_decod = post_activation


    def compute_error_decoder(self, target):
        self.error_decoder = target - self.decoder_output
    
    def compute_error_encoder(self, target):
        self.error_encoder = target - self.encoder_output
       
    def back_propagation_decoder(self):        #compute delta
        delta_list = []
        delta = self.error_decoder * sigmoid_deriv(self.pre_activ_decod[-1])    #this is the delta of output layer
        delta_list.append(delta)

        for index, layer in reversed(list(enumerate(self.decoder))):
            delta = np.dot(delta, layer.weights)
            delta = delta * relu_deriv(self.pre_activ_decod[index])
            delta_list.append(delta)
        
        delta_bottle_neck = np.dot(delta, self.decoder_input_layer.weights) 
        delta_list.append(delta_bottle_neck)

        delta_list.reverse()
        self.delta_decod = delta_list
        
    def back_propagation_encoder(self):
        delta_list = []
        delta = self.error_encoder #this is the delta of output layer
        delta_list.append(delta)

        index = self.hid_num - 1
       
        for index, layer in reversed(list(enumerate(self.encoder))):   #except the last element
            delta = np.dot(delta, layer.weights)
            delta = delta * relu_deriv(self.pre_activ_encod[index])   #we dont have the input layer weights, it begins from the layers that have fault
            delta_list.append(delta)
            
            
        delta_list.reverse()
        self.delta_encod = delta_list


    def plot_accur_loss(self, epochs):

        epochs = [i+1 for i in range(epochs)]

        plt.figure(1)
        plt.plot(epochs, self.train_decoder_loss, label = "Training Decoder Loss", color = "green")  # Plot the first curve
        #plt.plot(epochs, self.val_decoder_loss, label = "Validation Decoder Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(2)
        plt.plot(epochs, self.train_encoder_loss, label = "Training Encoder Loss", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_encoder_loss, label = "Validation Encoder Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(3)
        plt.plot(epochs, self.train_encoder_accuracy, label = "Training Encoder Accuracy", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_encoder_accuracy, label = "Validation Encoder Accuracy", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()  # Show labels for each curve

        # Show the plots
        plt.show()
        return 
    
    def gradient_descent_decoder(self, eta):    #update weights and biases
        
        for index, layer in reversed(list(enumerate(self.decoder))):
            layer.weights += eta * np.dot(self.delta_decod[index + 2].reshape(-1, 1), self.post_activ_decod[index].reshape(1, -1))
            layer.bias += eta * self.delta_decod[index + 2]
            index -= 1
        
        self.decoder_input_layer.weights += eta * np.dot(self.delta_decod[1].reshape(-1, 1), self.decoder_input.reshape(1, -1))
        self.decoder_input_layer.bias += eta * self.delta_decod[1]
        

    def gradient_descent_encoder(self, train_data, eta):

        for index, layer in reversed(list(enumerate(self.encoder))):
            layer.weights += eta * np.dot(self.delta_encod[index + 1].reshape(-1, 1), self.post_activ_encod[index].reshape(1, -1))
            layer.bias += eta * self.delta_encod[index + 1]
            
        self.encoder_input.weights += eta * np.dot(self.delta_encod[0].reshape(-1, 1), train_data.reshape(1, -1))
        self.encoder_input.bias += eta * self.delta_encod[0]


    
    def train_network(self, train_data, train_labels, val_data, val_labels, epoch_count, eta):
        for epoch in range(epoch_count):
        
            loss_decoder = 0
            loss_encoder = 0
            correct_classif_count = 0
            print(f"Epoch is {epoch + 1}")
            start_time = time.time()
            
            for index, sample in enumerate(train_data):
                self.feed_forward_encoder(sample)
                self.feed_forward_decoder(train_labels[index])
                self.compute_error_decoder(sample)
                self.compute_error_encoder(train_labels[index])

                #compute MSE
                loss_decoder += np.mean(np.square(self.error_decoder))/2

                #compute CEL
                loss_encoder += np.sum(-train_labels[index]*np.log(self.encoder_output))

                self.back_propagation_decoder()
                self.back_propagation_encoder()

                self.gradient_descent_decoder(eta)
                self.gradient_descent_encoder(sample, eta)

                prediction = np.argmax(self.encoder_output) 
                if prediction == np.argmax(train_labels[index]):
                    correct_classif_count += 1

            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            print(f"Elapsed Time: {elapsed_time} seconds")       
            self.train_encoder_loss.append(loss_encoder/train_data.shape[0])
            self.train_decoder_loss.append(loss_decoder/train_data.shape[0])

            print(f"Training encoder accuracy:")
            print(f"{(correct_classif_count * 100) / train_data.shape[0]} % \n")
            self.train_encoder_accuracy.append((correct_classif_count * 100) / train_data.shape[0])

            print(f"Training Encoder loss: {loss_encoder/train_data.shape[0]}")    
            print(f"Training Decoder loss: {loss_decoder/train_data.shape[0]}")    

            self.test_network(val_data, val_labels, "val")

    def test_network(self, data, labels, keyword):
        loss_decoder = 0
        loss_encoder = 0
        correct_classif_count = 0

        for index, sample in enumerate(data):
            self.feed_forward_encoder(sample)

            prediction_next_digit = np.argmax(self.encoder_output) 
            if (prediction_next_digit != 9):
                prediction_next_digit += 1
            else:
                prediction_next_digit = 0
            
            if (np.argmax(labels[index]) != 9):
                actual_next_digit = np.argmax(labels[index]) + 1
            else:
                actual_next_digit = 0
            
            
            if prediction_next_digit == actual_next_digit:
                correct_classif_count += 1

            decoder_input = one_hot_encoding_num(prediction_next_digit)

            self.feed_forward_decoder(decoder_input)

            self.compute_error_decoder(sample)
            self.compute_error_encoder(decoder_input)
            
            #compute MSE, we don't compute it
            #loss_decoder += np.mean(np.square(self.error_decoder))/2

            #compute CEL
            loss_encoder += np.sum(-labels[index]*np.log(self.encoder_output))

            if(keyword =="test"):
                self.items.append((sample, self.decoder_output))


        if(keyword == "val"):
            self.val_encoder_loss.append(loss_encoder/data.shape[0])
            self.val_decoder_loss.append(loss_decoder/data.shape[0])
            print(f"Validation encoder accuracy:")
            print(f"{correct_classif_count * 100 / data.shape[0]} %")
            self.val_encoder_accuracy.append(correct_classif_count * 100 / data.shape[0])
            print(f"Validation Encoder loss: {loss_encoder/data.shape[0]}")
            

if __name__ == "__main__":

    (train_data, train_labels), (test_data, test_labels) = load_mnist() #keras

    train_labels = one_hot_encoding(train_labels)
    test_labels = one_hot_encoding(test_labels)
    #sample size
    input_size = train_data.shape[1]

    epoch_count = 10
    layers = [128, 64, 36, 10, 36, 64, 128]
    eta = 1e-3

    print(layers)
    print(eta)

    my_autoencoder = autoencoder(layers, input_size = input_size)
    my_autoencoder.train_network(train_data, train_labels, test_data, test_labels, epoch_count = epoch_count, eta = eta)
    my_autoencoder.test_network(test_data, test_labels, "test")
    my_autoencoder.plot_accur_loss(epoch_count)


    for i, item in enumerate(my_autoencoder.items):
        image = item[0].reshape(-1, 28, 28)
        reconstructed = item[1].reshape(-1, 28, 28)
        plt.imshow(image[0], cmap = 'gray')
        plt.show()
        plt.imshow(reconstructed[0], cmap = 'gray')
        plt.show()
    