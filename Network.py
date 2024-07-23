import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, name, numberOfHiddenLayers, numberOfInputNeurons, numberOfHiddenLayerNeurons,
                 numberOfOutputNeurons, randomize=True):

        if numberOfInputNeurons <= 0 or numberOfHiddenLayerNeurons <= 0 or numberOfOutputNeurons <= 0:
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
        self.outputBiases = None
        self.hiddenBiases = None
        self.hiddenToOutputWeights = None
        self.hiddenToHiddenWeights = None
        self.inputToHiddenWeights = None
        self.expectedOutput = None
        self.cost = None
        self.inputsDataset = None
        self.outputsDataset = None
        self.inputLayer = None
        self.hiddenLayers = None
        self.outputLayer = None
        self.output = None
        self.numberCorrect = None
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
        self.inputToHiddenWeights = [[0] * self.numberOfInputNeurons] * self.numberOfHiddenLayerNeurons
        self.hiddenToHiddenWeights = ([[[0] * self.numberOfHiddenLayerNeurons] * self.numberOfHiddenLayerNeurons] * (
            self.numberOfHiddenLayers))
        self.hiddenToOutputWeights = [[0] * self.numberOfHiddenLayerNeurons] * self.numberOfOutputNeurons
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
        self.inputToHiddenWeights = [[random.random() for i in range(self.numberOfInputNeurons)] for j in
                                     range(self.numberOfHiddenLayerNeurons)]
        self.hiddenToHiddenWeights = [[[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                                       range(self.numberOfHiddenLayerNeurons)] for k in
                                      range(self.numberOfHiddenLayers)]
        self.hiddenToOutputWeights = [[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                                      range(self.numberOfOutputNeurons)]
        print("Randomized weights")

    def randomizeBiases(self):
        self.hiddenBiases = [[random.random() for i in range(self.numberOfHiddenLayerNeurons)] for j in
                             range(self.numberOfHiddenLayers)]
        self.outputBiases = [random.random() for i in range(self.numberOfOutputNeurons)]
        print("Randomized biases")

    def feedForward(self):
        for i in range(self.numberOfHiddenLayerNeurons):
            self.hiddenLayers[0][i] = np.dot(self.inputLayer, self.inputToHiddenWeights[i]) + self.hiddenBiases[0][i]
            self.hiddenLayers[0][i] = sigmoid(self.hiddenLayers[0][i])
        for i in range(1, self.numberOfHiddenLayers):
            for j in range(self.numberOfHiddenLayerNeurons):
                self.hiddenLayers[i][j] = np.dot(self.hiddenLayers[i - 1], self.hiddenToHiddenWeights[i][j]) + \
                                          self.hiddenBiases[i][j]
                self.hiddenLayers[i][j] = sigmoid(self.hiddenLayers[i][j])
        for i in range(self.numberOfOutputNeurons):
            self.outputLayer[i] = np.dot(self.hiddenLayers[self.numberOfHiddenLayers - 1],
                                         self.hiddenToOutputWeights[i]) + self.outputBiases[i]
            self.outputLayer[i] = sigmoid(self.outputLayer[i])
        print("Feed Forward Complete")

    def calculateCost(self):
        self.cost = 0
        for i in range(self.numberOfOutputNeurons):
            self.cost += 0.5 * (self.outputLayer[i] - self.expectedOutput[i]) ** 2
        print(f"Cost: {self.cost}")

    def setExpectedOutput(self, expectedOutput):
        self.expectedOutput = expectedOutput
        print("Expected output set")

    def backPropagate(self, learning_rate=0.1):
        # Calculate output layer error
        output_errors = [0] * self.numberOfOutputNeurons
        for i in range(self.numberOfOutputNeurons):
            output_errors[i] = (self.expectedOutput[i] - self.outputLayer[i]) * self.outputLayer[i] * (
                        1 - self.outputLayer[i])

        # Calculate hidden layer error
        hidden_errors = [[0] * self.numberOfHiddenLayerNeurons for _ in range(self.numberOfHiddenLayers)]
        for i in range(self.numberOfHiddenLayers - 1, -1, -1):
            if i == self.numberOfHiddenLayers - 1:
                for j in range(self.numberOfHiddenLayerNeurons):
                    hidden_errors[i][j] = sum(self.hiddenToOutputWeights[k][j] * output_errors[k] for k in
                                              range(self.numberOfOutputNeurons)) * self.hiddenLayers[i][j] * (
                                                      1 - self.hiddenLayers[i][j])
            else:
                for j in range(self.numberOfHiddenLayerNeurons):
                    hidden_errors[i][j] = sum(self.hiddenToHiddenWeights[i + 1][k][j] * hidden_errors[i + 1][k] for k in
                                              range(self.numberOfHiddenLayerNeurons)) * self.hiddenLayers[i][j] * (
                                                      1 - self.hiddenLayers[i][j])

        # Update output layer weights and biases
        for i in range(self.numberOfOutputNeurons):
            for j in range(self.numberOfHiddenLayerNeurons):
                self.hiddenToOutputWeights[i][j] += learning_rate * output_errors[i] * self.hiddenLayers[-1][j]
            self.outputBiases[i] += learning_rate * output_errors[i]

        # Update hidden layer weights and biases
        for i in range(self.numberOfHiddenLayers - 1, -1, -1):
            if i > 0:
                for j in range(self.numberOfHiddenLayerNeurons):
                    for k in range(self.numberOfHiddenLayerNeurons):
                        self.hiddenToHiddenWeights[i][j][k] += learning_rate * hidden_errors[i][j] * \
                                                               self.hiddenLayers[i - 1][k]
                    self.hiddenBiases[i][j] += learning_rate * hidden_errors[i][j]
            else:
                for j in range(self.numberOfHiddenLayerNeurons):
                    for k in range(self.numberOfInputNeurons):
                        self.inputToHiddenWeights[j][k] += learning_rate * hidden_errors[i][j] * self.inputLayer[k]
                    self.hiddenBiases[i][j] += learning_rate * hidden_errors[i][j]

        print("Back Propagation Complete")

    def loadInputData(self, dataset):
        self.inputsDataset = dataset
        print("Input Dataset loaded")

    def loadOutputData(self, dataset):
        self.outputsDataset = dataset
        print("Output Dataset loaded")

    def fullCycle(self, datasetIteration):
        if self.inputsDataset is None or self.outputsDataset is None:
            print("Error: Dataset(s) missing")
            return
        self.inputLayer = self.inputsDataset[datasetIteration]
        self.setExpectedOutput(self.outputsDataset[datasetIteration])
        self.feedForward()
        self.calculateCost()
        self.backPropagate()
        print("Full Cycle Complete")

    def getDatasetLength(self):
        return len(self.inputsDataset)

    def storeModel(self, path):
        with open(path, "w",) as file:
            file.writelines([f"{self.name}\r\n", f"{self.numberOfHiddenLayers}\r\n", f"{self.numberOfInputNeurons}\r\n",
                             f"{self.numberOfHiddenLayerNeurons}\r\n", f"{self.numberOfOutputNeurons}\r\n",
                             f"{self.inputToHiddenWeights}\r\n", f"{self.hiddenToHiddenWeights}\r\n",
                             f"{self.hiddenToOutputWeights}\r\n", f"{self.hiddenBiases}\r\n", f"{self.outputBiases}\r\n",
                             f"{self.expectedOutput}\r\n", f"{self.cost}\r\n", f"{list(self.inputsDataset)}\r\n",
                             f"{self.outputsDataset}\r\n"])

            print("Model stored")

    def loadModel(self, path):
        with open(path, "r") as file:
            data = file.readlines()
            self.name = data[0]
            self.numberOfHiddenLayers = int(data[1])
            self.numberOfInputNeurons = int(data[2])
            self.numberOfHiddenLayerNeurons = int(data[3])
            self.numberOfOutputNeurons = int(data[4])
            self.inputToHiddenWeights = eval(data[5])
            self.hiddenToHiddenWeights = eval(data[6])
            self.hiddenToOutputWeights = eval(data[7])
            self.hiddenBiases = eval(data[8])
            self.outputBiases = eval(data[9])
            self.expectedOutput = eval(data[10])
            self.cost = float(data[11][:-2])
            self.inputsDataset = eval(data[12])
            self.outputsDataset = eval(data[13])

        print("Model loaded")

    def chooseOutput(self):
        self.output = self.outputLayer.index(max(self.outputLayer))
        if self.expectedOutput[self.output] == 1:
            self.numberCorrect += 1
            print(f"Output: {self.output} is correct")
        else:
            print(f"Output: {self.output} is incorrect")

    def calculateAccuracy(self):
        return self.numberCorrect / len(self.outputsDataset) * 100

    def testModel(self, datasetIteration):
        if self.inputsDataset is None or self.outputsDataset is None:
            print("Error: Dataset(s) missing")
            return
        self.inputLayer = self.inputsDataset[datasetIteration]
        self.setExpectedOutput(self.outputsDataset[datasetIteration])
        self.feedForward()
        self.calculateCost()
        print("Test Complete")

    def __del__(self):
        print(f"{self.name} destroyed")
