import network
import json


size = [25, 3, 1]

inputs = [[0, 0, 0, 0, 0,
           0, 1, 1, 1, 0,
           0, 1, 0, 1, 0,
           0, 1, 1, 1, 0,
           1, 0, 0, 0, 0], [0, 1, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 1, 0, 1, 0,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 0], [0, 1, 0, 0, 0,
                                             0, 1, 0, 1, 0,
                                             0, 1, 0, 0, 0,
                                             0, 0, 1, 1, 0,
                                             1, 0, 0, 0, 0]]
targets = [[1.0], [1.0], [0.0]]

test_inputs = [[0, 0, 0, 1, 0,
                0, 0, 0, 1, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 1, 0,
                1, 0, 0, 0, 0], [1, 0, 0, 1, 0,
                                 0, 1, 1, 1, 0,
                                 0, 1, 0, 1, 0,
                                 0, 1, 1, 1, 0,
                                 1, 0, 0, 0, 0]]


nt = network.Network(size)
# nt.Load_Weights_and_Biases("Data")
nt.Train(epoch=10000, input_data=inputs, target_data=targets,
         batch_size=3, learning_rate=0.1, show_results=True)
print(nt.Feedforward(test_inputs[0]), nt.Feedforward(test_inputs[1]))
