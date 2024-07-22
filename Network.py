import random


class NeuralNetwork:
    def __init__(self, name, numberOfHiddenLayers, numberOfInputNeurons, numberOfHiddenLayerNeurons,
                 numberOfOutputNeurons, randomize=True):
        if numberOfInputNeurons == 0 or numberOfHiddenLayerNeurons == 0 or numberOfOutputNeurons == 0:
            print("Error: Missing neurons")
            return
        if numberOfHiddenLayers <= 0:
            print("Error: Invalid number of hidden layers")
            return
        self.name = name
        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.numberOfInputNeurons = numberOfInputNeurons
        self.numberOfHiddenLayerNeurons = numberOfHiddenLayerNeurons
        self.numberOfOutputNeurons = numberOfOutputNeurons
        self.__createLayers()
        self.__createWeights()
        self.__createBiases()
        if randomize:
            self.randomizeAll()
        print(f"{self.name} successfully created")

    def __createLayers(self):
        self.inputLayer = [0] * self.numberOfInputNeurons
        self.hiddenLayers = [[0] * self.numberOfHiddenLayerNeurons] * self.numberOfHiddenLayers
        self.outputLayer = [0] * self.numberOfOutputNeurons
        print("Layers created")

    def __createWeights(self):
        self.inputToHiddenWeights = [[0] * self.numberOfHiddenLayerNeurons] * self.numberOfInputNeurons
        self.hiddenToHiddenWeights = ([[[0] * self.numberOfHiddenLayerNeurons] * self.numberOfHiddenLayerNeurons] * (
            self.numberOfHiddenLayers))
        self.hiddenToOutputWeights = [[0] * self.numberOfOutputNeurons] * self.numberOfHiddenLayerNeurons
        print("Weights created")

    def __createBiases(self):
        self.hiddenBiases = [[0] * self.numberOfHiddenLayerNeurons] * self.numberOfHiddenLayers
        self.outputBiases = [0] * self.numberOfOutputNeurons
        print("Biases created")

    def randomizeAll(self):
        self.randomizeWeights()
        self.randomizeBiases()
        print("Randomized all")

    def randomizeWeights(self):
        self.inputToHiddenWeights = [[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                                     range(self.numberOfInputNeurons)]
        self.hiddenToHiddenWeights = [[[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                                       range(self.numberOfHiddenLayerNeurons)] for k in
                                      range(self.numberOfHiddenLayers)]
        self.hiddenToOutputWeights = [[random.random() for i in range(self.numberOfOutputNeurons)] for j in
                                      range(self.numberOfHiddenLayerNeurons)]
        print("Randomized weights")

    def randomizeBiases(self):
        self.hiddenBiases = [[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                             range(self.numberOfHiddenLayers)]
        self.outputBiases = [random.random() for i in range(self.numberOfOutputNeurons)]
        print("Randomized biases")

    def __del__(self):
        print(f"{self.name} destroyed")
