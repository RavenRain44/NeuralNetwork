import random
import numpy as np
import pickle as pkl


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
        self.numberCorrect = 0
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
        print(f'Input: {self.inputLayer}')
        self.setExpectedOutput(self.outputsDataset[datasetIteration])
        print(f'Expected Output: {self.expectedOutput}')
        self.feedForward()
        print(f'Output: {self.outputLayer}')
        self.chooseOutput()
        print(f'Selected Output: {self.output}')
        self.calculateCost()
        self.backPropagate()
        print("Full Cycle Complete")

    def getDatasetLength(self):
        return len(self.inputsDataset)

    def storeModel(self, path):
        storeFile = open(path, "ab")
        pkl.dump(self, storeFile)
        storeFile.close()

        print("Model stored")

    def loadModel(self, path):
        loadFile = open(path, "rb")
        loadedModel = pkl.load(loadFile)
        self.importModel(loadedModel)
        loadFile.close()

        print("Model loaded")

    def importModel(self, model):
        self.name = model.name
        self.numberOfHiddenLayers = model.numberOfHiddenLayers
        self.numberOfInputNeurons = model.numberOfInputNeurons
        self.numberOfHiddenLayerNeurons = model.numberOfHiddenLayerNeurons
        self.numberOfOutputNeurons = model.numberOfOutputNeurons
        self.outputBiases = model.outputBiases
        self.hiddenBiases = model.hiddenBiases
        self.hiddenToOutputWeights = model.hiddenToOutputWeights
        self.hiddenToHiddenWeights = model.hiddenToHiddenWeights
        self.inputToHiddenWeights = model.inputToHiddenWeights
        self.expectedOutput = model.expectedOutput
        self.cost = model.cost
        self.inputsDataset = model.inputsDataset
        self.outputsDataset = model.outputsDataset
        self.inputLayer = model.inputLayer
        self.hiddenLayers = model.hiddenLayers
        self.outputLayer = model.outputLayer
        self.output = model.output
        self.numberCorrect = model.numberCorrect

    def chooseOutput(self):
        self.output = self.outputLayer.index(max(self.outputLayer))
        if self.numberCorrect is None:
            self.numberCorrect = 0
        if self.expectedOutput == self.outputLayer:
            self.numberCorrect += 1
            print(f"Output: {self.output} is correct")
        else:
            print(f"Output: {self.output} is incorrect")

    def calculateAccuracy(self):
        return f"Accuracy: {self.numberCorrect / len(self.outputsDataset) * 100}"

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
