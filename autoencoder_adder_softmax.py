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

def show_mnist_picture(picture):
    adjusted_pic = picture.reshape(28, 28)
    plt.imshow(adjusted_pic, cmap = 'gray')
    plt.show()

def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
    train_data = train_data.reshape(-1, 784)
    test_data = test_data.reshape(-1, 784)

    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    train_data = train_data / 255   
    test_data = test_data / 255
    return (train_data, train_labels), (test_data, test_labels)

def separate_digits(n):
    # Convert the integer to a string
    str_number = str(n)

    # If the number has only one digit, add a leading zero
    if len(str_number) == 1:
        str_number = '0' + str_number

    # Extract the individual digits
    digit1 = int(str_number[0])
    digit2 = int(str_number[1])

    return digit1, digit2

def create_input_2(data, labels):
    new_data = []
    new_labels = []
    for index, sample in enumerate(data[:-1]):
        new_data.append(np.concatenate((sample, data[index + 1])))
        new_labels.append((labels[index], labels[index + 1]))
    return new_data, new_labels

def create_specific_decoder_digits(data, labels):
    my_data = [[] for _ in range(10)] 
    for index, sample in enumerate(data):
        my_data[labels[index]].append(sample)

    expected_nums = [] 

    for i in range(10):
        expected_nums.append(my_data[i][0])

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
        
        self.train_encoder_first_loss = []
        self.train_encoder_second_loss = []

        self.train_decoder_loss = []

        self.val_encoder_first_loss = []
        self.val_encoder_second_loss = []

        self.val_decoder_loss = []

        self.train_first_accuracy = []
        self.train_second_accuracy = []

        self.val_first_accuracy = []
        self.val_second_accuracy = []

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
                my_post_activation = np.zeros(20)
                my_post_activation[:10] = softmax(my_pre_activation[:10])
                my_post_activation[10:] = softmax(my_pre_activation[10:])
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
        plt.plot(epochs, self.val_decoder_loss, label = "Validation Decoder Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(2)
        plt.plot(epochs, self.train_encoder_first_loss, label = "Training Encoder First Digit Loss", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_encoder_first_loss, label = "Validation Encoder First Digit Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(3)
        plt.plot(epochs, self.train_encoder_second_loss, label = "Training Encoder Second Digit Loss", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_encoder_second_loss, label = "Validation Encoder Second Digit Loss", color = "blue")  # Plot the second curve

        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(4)
        plt.plot(epochs, self.train_first_accuracy, label = "Training Encoder First Digit Acccuracy", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_first_accuracy, label = "Validation Encoder First Digit Accuracy", color = "blue")  # Plot the second curve
        
        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()  # Show labels for each curve

        plt.figure(5)
        plt.plot(epochs, self.train_second_accuracy, label = "Training Encoder Second Digit Acccuracy", color = "green")  # Plot the first curve
        plt.plot(epochs, self.val_second_accuracy, label = "Validation Encoder Second Digit Accuracy", color = "blue")  # Plot the second curve
        
        # Set labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()  # Show labels for each curve

        # Show the plots
        plt.show()
        return 
    
    def gradient_descent_decoder(self, train_data, eta):    #update weights and biases
        
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


    
    def train_network(self, train_data, train_labels, val_data, val_labels, expected_nums, epoch_count, eta):
        for epoch in range(epoch_count):
            
            loss_decoder = 0
            loss_encoder_first = 0
            loss_encoder_second = 0

            
            correct_classif_count_first = 0
            correct_classif_count_second = 0
            start_time = time.time()
            print(f"Epoch is {epoch + 1}")

            
            for index, sample in enumerate(train_data):
                self.feed_forward_encoder(sample)

                two_one_hot_encoded_labels = np.concatenate((train_labels[index][0], train_labels[index][1]))
                self.feed_forward_decoder(two_one_hot_encoded_labels)


                prediction_first = np.argmax(self.encoder_output[:10])
                prediction_second = np.argmax(self.encoder_output[10:])

                actual_first_digit = np.argmax(train_labels[index][0])
                actual_second_digit = np.argmax(train_labels[index][1])

                if (actual_first_digit == prediction_first):
                    correct_classif_count_first += 1

                if (actual_second_digit == prediction_second):
                    correct_classif_count_second += 1
                
                expected_output = np.concatenate((expected_nums[actual_first_digit], expected_nums[actual_second_digit]))
                self.compute_error_decoder(expected_output)

                self.compute_error_encoder(two_one_hot_encoded_labels)

                #compute MSE
                loss_decoder += np.mean(np.square(self.error_decoder))/2

                #compute CEL
                loss_encoder_first += np.sum(-train_labels[index][0]*np.log(self.encoder_output[:10]))
                loss_encoder_second += np.sum(-train_labels[index][1]*np.log(self.encoder_output[10:]))

                self.back_propagation_decoder()
                self.back_propagation_encoder()

                self.gradient_descent_decoder(sample, eta)
                self.gradient_descent_encoder(sample, eta)

               

                prediction_first = one_hot_encoding_num(prediction_first)
                prediction_second = one_hot_encoding_num(prediction_second)

                prediction = np.concatenate((prediction_first, prediction_second))

            end_time = time.time()

            elapsed_time = end_time - start_time

            print(f"Elapsed Time: {elapsed_time} seconds")   
            num_data = len(train_data)
            self.train_encoder_first_loss.append(loss_encoder_first/num_data)
            self.train_encoder_second_loss.append(loss_encoder_second/num_data)
            self.train_decoder_loss.append(loss_decoder/num_data)

            print(f"Training encoder accuracy first:")
            print(f"{correct_classif_count_first * 100 / num_data} %")
            self.train_first_accuracy.append((correct_classif_count_first * 100 )/ num_data)

            print(f"Training encoder accuracy second:")
            print(f"{correct_classif_count_second * 100 / num_data} %")
            self.train_second_accuracy.append((correct_classif_count_second * 100 )/ num_data)


            
            print(f"Training Encoder First Digit loss: {loss_encoder_first/num_data}")   
            print(f"Training Encoder Second Digit loss: {loss_encoder_second/num_data}")  
            print(f"Training Decoder loss: {loss_decoder/num_data}")    

            self.test_network(val_data, val_labels, expected_nums, "val")

    def test_network(self, data, labels, expected_nums, keyword):
        loss_decoder = 0
        loss_encoder_first = 0
        loss_encoder_second = 0

        correct_classif_count_first = 0
        correct_classif_count_second = 0

        for index, sample in enumerate(data):
            self.feed_forward_encoder(sample)

            prediction_first_digit = np.argmax(self.encoder_output[:10]) 
            prediction_second_digit = np.argmax(self.encoder_output[10:]) 

            prediction_sum = prediction_first_digit + prediction_second_digit

            actual_first_digit = np.argmax(labels[index][0])
            actual_second_digit = np.argmax(labels[index][1])

            if (actual_first_digit == prediction_first_digit):
                correct_classif_count_first += 1

            if (actual_second_digit == prediction_second_digit):
                correct_classif_count_second += 1


            digit1, digit2 = separate_digits(prediction_sum)

            digit1_one_hot, digit2_one_hot = one_hot_encoding_num(digit1), one_hot_encoding_num(digit2)
            prediction_sum = np.concatenate((digit1_one_hot, digit2_one_hot))

            decoder_input = prediction_sum

            self.feed_forward_decoder(decoder_input)

            expected_output = np.concatenate((expected_nums[actual_first_digit], expected_nums[actual_second_digit]))
            self.compute_error_decoder(expected_output)

            self.compute_error_encoder(decoder_input)
            
            #compute MSE
            loss_decoder += np.mean(np.square(self.error_decoder))/2

            #compute CEL
            loss_encoder_first += np.sum(-labels[index][0]*np.log(self.encoder_output[:10]))
            loss_encoder_second += np.sum(-labels[index][1]*np.log(self.encoder_output[10:]))

            if(keyword == "test"):
                self.items.append((sample, self.decoder_output))

        num_data = len(data)
        if(keyword == "val"):
            self.val_encoder_first_loss.append(loss_encoder_first/num_data)
            self.val_encoder_second_loss.append(loss_encoder_second/num_data)
            self.val_decoder_loss.append(loss_decoder/num_data)

            print(f"Validation Encoder Accuracy First:")
            print(f"{correct_classif_count_first * 100 / num_data} %")
            self.val_first_accuracy.append((correct_classif_count_first * 100) / num_data)

            print(f"Validation Encoder Accuracy Second:")
            print(f"{correct_classif_count_second * 100 / num_data} %")
            self.val_second_accuracy.append((correct_classif_count_second * 100) / num_data)

            print(f"Validation Encoder First Digit loss: {loss_encoder_first/num_data}")
            print(f"Validation Encoder Second Digit loss: {loss_encoder_second/num_data}")
            print(f"Validation Decoder loss: {loss_decoder/num_data}")

            

if __name__ == "__main__":
    
    #to ensure pseudo-randomness        
    np.random.seed(0)   

    (train_data, train_labels), (test_data, test_labels) = load_mnist() #keras

    expected_nums = create_specific_decoder_digits(train_data, train_labels)

    train_labels = one_hot_encoding(train_labels)
    test_labels = one_hot_encoding(test_labels)

    train_data, train_labels = create_input_2(train_data, train_labels)
    test_data, test_labels = create_input_2(test_data, test_labels)


    #sample size
    input_size = train_data[0].shape[0]

    epoch_count = 20
    layers = [128, 64, 36, 20, 36, 64, 128]
    eta = 1e-3

    print(layers)
    print(eta)

    my_autoencoder = autoencoder(layers, input_size = input_size)
    my_autoencoder.train_network(train_data, train_labels, test_data, test_labels, expected_nums, epoch_count = epoch_count, eta = eta)
    my_autoencoder.test_network(test_data, test_labels, expected_nums, "test")
    my_autoencoder.plot_accur_loss(epoch_count)


    for i, item in enumerate(my_autoencoder.items):
        digit_1 = item[0][:784]
        show_mnist_picture(digit_1)

        digit_2 = item[0][784:]
        show_mnist_picture(digit_2)

        sum_1 = item[1][:784]
        show_mnist_picture(sum_1)

        sum_2 = item[1][784:]
        show_mnist_picture(sum_2)