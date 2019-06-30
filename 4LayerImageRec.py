from numpy import exp, array, random, dot
from PIL import Image
import numpy as np

import torch
print(torch.cuda.get_device_name(0),torch.cuda.is_available())

#https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons))-1
        self.bias = 1

class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3, layer4):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2, output_from_layer_3, output_from_layer_4 = self.think(training_set_inputs)

            layer4_error = training_set_outputs - output_from_layer_4
            layer4_delta = layer4_error * self.__sigmoid_derivative(output_from_layer_4)

            # Calculate the error for layer 3 (The difference between the desired output
            # and the predicted output).
            
            layer3_error = layer4_delta.dot(self.layer4.synaptic_weights.T)
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)
            
            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            
            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
            
            #print(training_set_outputs, output_from_layer_2,layer2_error)
            
            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = 0.1 * training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = 0.1 * output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = 0.1 * output_from_layer_2.T.dot(layer3_delta)
            layer4_adjustment = 0.1 * output_from_layer_3.T.dot(layer4_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            self.layer3.synaptic_weights += layer3_adjustment
            self.layer4.synaptic_weights += layer4_adjustment

            ## Adjust the bias
            self.layer1.bias += 0.1 * np.sum(layer1_delta)
            self.layer2.bias += 0.1 * np.sum(layer2_delta)
            self.layer3.bias += 0.1 * np.sum(layer3_delta)
            self.layer4.bias += 0.1 * np.sum(layer4_delta)

            if iteration == (round(number_of_training_iterations * 0.01)):
                print("Progress: 1%")
            if iteration == (round(number_of_training_iterations * 0.25)):
                print("Progress: 25%")
            if iteration == (round(number_of_training_iterations * 0.5)):
                print("Progress: 50%")
            if iteration == (round(number_of_training_iterations * 0.75)):
                print("Progress: 75%")
            if iteration == (round(number_of_training_iterations * 1)):
                print("Progress: 100%")



            posaim =  0.00001
            negaim = -0.00001
            if np.sum(layer1_adjustment) < posaim and np.sum(layer1_adjustment) > negaim:
                if np.sum(layer2_adjustment) < posaim and np.sum(layer2_adjustment) > negaim:
                    if np.sum(layer3_adjustment) < posaim and np.sum(layer3_adjustment) > negaim:
                        if np.sum(layer4_adjustment) < posaim and np.sum(layer4_adjustment) > negaim:
                            print(True,iteration)
                            print("error:", layer1_error,layer2_error,layer3_error,layer4_error)
                            print("adjustments:", layer1_adjustment,layer2_adjustment,layer3_adjustment,layer4_adjustment)
                            break
        print("never made it")
        print("error:", layer1_error,layer2_error,layer3_error,layer4_error)
        print("adjustments:", layer1_adjustment,layer2_adjustment,layer3_adjustment,layer4_adjustment)



    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights)+self.layer1.bias)
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights)+self.layer2.bias)
        output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights)+self.layer2.bias)
        output_from_layer4 = self.__sigmoid(dot(output_from_layer3, self.layer4.synaptic_weights)+self.layer3.bias)
        
        return output_from_layer1, output_from_layer2, output_from_layer3,output_from_layer4

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (4 neurons, each with 3 inputs): ")
        print(self.layer1.synaptic_weights)
        print("L1 bias", self.layer1.bias)
        print("    Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.synaptic_weights)
        print("L2 bias", self.layer2.bias)

        print("    Layer 3 (1 neuron, with 4 inputs):")
        print(self.layer3.synaptic_weights)
        print("L3 bias", self.layer3.bias)

        print("    Layer 4 (1 neuron, with 4 inputs):")
        print(self.layer4.synaptic_weights)
        print("L4 bias", self.layer4.bias)
                
if __name__ == "__main__":
    #   o
    #o<   \
    #   o
    #o<   >  o - data 1
    #   o
    #o<   >  o - data 2
    #   o
    #o<   >  o - data 3
    #   o
    #o<   >  o - data 4
    #   o
    #o    /
    #   o
    
    
    imgArray = [] #final 


    #append pixles for inputs
    for s in range(0,13+1): #0-13 results
        for t in range(0,1): #3 test images for every expected result
            im = Image.open("images/images/Akkad/"+str(s)+"."+str(t)+".png")
            im1 = np.asarray(im)
            img1 = []
            
            for i in range(len(im1)): #extract image pixel data
                for j in range(len(im1[i])):
                    img1.append(im1[i][j][0]);
            imgArray.append(img1) #append image to array of images

    
    
    
    #Seed the random number generator
    random.seed(1)    

    #state connections between neurons

    #note find a way for nodes to test and adjust after training to find the optimale number of neurons
        #and connections.
    layerInput_1 = len(img1) #all images must be the same pixel size 45,14,6,14
    layer1_2 = 200
    layer2_3 = 100
    layer3_4 = 50
    layer4_out = 14
    
    # Create layer 1 (120 neurons, each with 6 inputs)
    layer1 = NeuronLayer(layer1_2, layerInput_1) 

    # Create layer 2 (5 neurons with 120 inputs)
    layer2 = NeuronLayer(layer2_3, layer1_2)

    # Create layer 3 (8 neurons with 5 inputs)
    layer3 = NeuronLayer(layer3_4, layer2_3)

    # Create layer 3 (8 neurons with 5 inputs)
    layer4 = NeuronLayer(layer4_out, layer3_4)
    
    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3, layer4)


    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.

    imgArray = array(imgArray)
    

    #image rec outputs
    #note need more data to train each node more than once
    #also need to find a way to not let a mess with training outputs appear. 
                                    #  0 1 2 3 4 5 6 7 8 9 10111213
                                    #  v v v v v v v v v v v v v v
    img_training_set_outputs = array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                      
                                      [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                          
                                      [0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                          
                                      [0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1]
                                      ]).T
    
    print("Test Inputs: ", len(img1))
    print(imgArray)

    print("Test Outputs: ")
    print(img_training_set_outputs)
    


    # Train the neural network using the training set.
    # Do it 100,000 times and make small adjustments each time.
    neural_network.train(imgArray, img_training_set_outputs, 60000)
    
    print("Stage 2) New synaptic weights after training for single stage output: ")
    neural_network.print_weights()

    check = "images/images/Akkad/1.0"
    while True:
        inputt = str(input())
        if " " != inputt != "" : #find image to check
            check = "images/images/Akkad/"+inputt+".0"
        im = Image.open(check+".png")
        imcheck = np.asarray(im) #convert img to array
        
        checkimg = []
        for i in range(len(imcheck)):#reformat check image
            for j in range(len(imcheck[i])):
                checkimg.append(imcheck[i][j][0]);
                
        # Test the neural network with a new situation.     
        print("Stage 3) Considering a new situation", array(checkimg), "-> ?: ")
        hidden_state1,hidden_state2, hidden_state3, output = neural_network.think(array(checkimg))
        print("original: ", output)
        results = [round(output.tolist()[0]),round(output.tolist()[1]),
              round(output.tolist()[2]), round(output.tolist()[3]),
              round(output.tolist()[4]),round(output.tolist()[5]),
              round(output.tolist()[6]), round(output.tolist()[7]),
              round(output.tolist()[8]), round(output.tolist()[9]),
              round(output.tolist()[10]), round(output.tolist()[11]),
              round(output.tolist()[12]), round(output.tolist()[13])]
        print("Rounded Result: ",results)
        if " " != inputt != "":
            print("what it should be: ", img_training_set_outputs[int(float(inputt))])
