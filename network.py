import random
import time
import json
from matplotlib import pyplot as plt

class Network:
    def __init__(self, size):
        self.__size = size
        self.weights = []
        self.biases = []
        self.e = 2.718281828459
        for index, layer_size in enumerate(self.__size):
            if index != len(self.__size)-1:
                connecting_layer_size = self.__size[index+1]
                self.biases.append([random.random()
                                    for _ in range(connecting_layer_size)])
                self.weights.append([[random.random() for _ in range(
                    layer_size)] for _ in range(connecting_layer_size)])

    def __Calculate_squared_error(self, target_outputs, computed_outputs):
        return sum([(computed_output-target_output)**2 for target_output, computed_output in zip(target_outputs, computed_outputs)])

    def __Calculate_error(self, target_outputs, computed_outputs):
        return [2*(computed_output-target_output) for target_output, computed_output in zip(target_outputs, computed_outputs)]

    def __Sigmoid(self, x):
        return 1 / (1 + self.e**-x)

    def __Derivative(self, x):
        return x*(1-x)

    def __Backpropagate(self, inputs=[], targets=[]):
        outputs = inputs
        activations = [inputs]
        for layer_weights, layer_bias in zip(self.weights, self.biases):
            outputs = [self.__Sigmoid(sum([neuron_weight*inpt for neuron_weight, inpt in zip(
                neuron_weights, outputs)])+bias) for neuron_weights, bias in zip(layer_weights, layer_bias)]
            activations.append(outputs)

        error = self.__Calculate_error(targets, outputs)
        errors = [error]

        for layer in self.weights[::-1]:
            error = [sum(pre_neuron_erros) for pre_neuron_erros in zip(
                *[[(n/sum(pre_neurons))*neuron_error for n in pre_neurons]for neuron_error, pre_neurons in zip(error, layer)])]
            errors.append(error)

        delta_biases = []
        delta_weights = []

        for layer_activation, pre_layer_activation, layer_errors in zip(activations[::-1], activations[-2::-1], errors):
            delta_layer_biases = []
            delta_layer_weights = []
            for neuron_activation, neuron_errors in zip(layer_activation, layer_errors):
                e = self.__Derivative(neuron_activation)*neuron_errors
                delta_neuron_biases = e
                delta_neuron_weights = [
                    e*pre_neuron_activation for pre_neuron_activation in pre_layer_activation]
                delta_layer_biases.append(delta_neuron_biases)
                delta_layer_weights.append(delta_neuron_weights)
            delta_biases.append(delta_layer_biases)
            delta_weights.append(delta_layer_weights)

        delta_biases = delta_biases[::-1]
        delta_weights = delta_weights[::-1]

        return delta_weights, delta_biases

    def Feedforward(self, inputs):
        outputs = inputs
        for layer_weights, layer_bias in zip(self.weights, self.biases):
            outputs = [self.__Sigmoid(sum([neuron_weight*inpt for neuron_weight, inpt in zip(
                neuron_weights, outputs)])+bias) for neuron_weights, bias in zip(layer_weights, layer_bias)]
        return outputs

    def Save_Weights_and_Biases(self, path="Data"):
        savefile = open(path, "w")
        savefile.write(str([self.weights, self.biases, self.__size]))
        savefile.close()
        print("Saved Savefile '{}'".format(path))

    def Load_Weights_and_Biases(self, path="Data"):
        savefile = json.load(open(path, "r"))
        self.weights = savefile[0]
        self.biases = savefile[1]
        self.__size = savefile[2]
        print("Loaded Savefile '{}'".format(path))

    def Train(self, input_data=[], target_data=[], save_path="Data", batch_size=1, epoch=1, learning_rate=1, show_results=True, print_progress=True, autosave=True):

        results = []
        time_buffer = []
        start_time = time.time()

        for _ in range(epoch):
            batch_delta_weights = []
            batch_delta_biases = []

            if show_results:
                random_index = random.randint(0, (len(input_data)-1))
                error = self.__Calculate_squared_error(
                    target_data[random_index], self.Feedforward(input_data[random_index]))
                time_buffer.append(time.time()-start_time)
                results.append(error)

            if print_progress:
                random_index = random.randint(0, (len(input_data)-1))
                print(self.Feedforward(
                    input_data[random_index]), target_data[random_index])

            for _ in range(batch_size):
                random_index = random.randint(0, (len(input_data)-1))
                delta_weights, delta_biases = self.__Backpropagate(
                    input_data[random_index], target_data[random_index])
                batch_delta_weights.append(delta_weights)
                batch_delta_biases.append(delta_biases)

            self.weights = [[[single_neoron_weigth-learning_rate*(sum(new_single_neoron_weight)/len(new_single_neoron_weight)) for single_neoron_weigth, new_single_neoron_weight in zip(neoron_weights, zip(
                *new_neoron_weights))] for neoron_weights, new_neoron_weights in zip(layer_weights, zip(*new_layer_weights))] for layer_weights, new_layer_weights in zip(self.weights, zip(*batch_delta_weights))]
            self.biases = [[neoron_bias-learning_rate*(sum(new_neoron_bias)/len(new_neoron_bias)) for neoron_bias, new_neoron_bias in zip(
                layer_bias, zip(*new_layer_bias))] for layer_bias, new_layer_bias in zip(self.biases, zip(*batch_delta_biases))]

        if autosave:
            self.Save_Weights_and_Biases(path=save_path)

        if show_results:
            plt.xlabel("time in Seconds")
            plt.ylabel("Squared Error")
            plt.plot(time_buffer, results)
            plt.show()
