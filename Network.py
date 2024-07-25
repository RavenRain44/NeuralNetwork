import numpy as np
import pickle as pkl


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivation(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivation(x):
    return 1 - np.tanh(x) ** 2


def forceLog(message):
    print(message)


class NeuralNetwork:
    def __init__(self, name, number_of_hidden_layers, number_of_input_neurons, number_of_hidden_layer_neurons,
                 number_of_output_neurons, randomize=True):

        if number_of_input_neurons <= 0 or number_of_hidden_layer_neurons <= 0 or number_of_output_neurons <= 0:
            raise ValueError("Error: Missing neurons")
        if number_of_hidden_layers <= 0:
            raise ValueError("Error: Invalid number of hidden layers")

        self.name = name
        self.numberOfHiddenLayers = number_of_hidden_layers
        self.numberOfInputNeurons = number_of_input_neurons
        self.numberOfHiddenLayerNeurons = number_of_hidden_layer_neurons
        self.numberOfOutputNeurons = number_of_output_neurons

        self.outputBiases = np.zeros(self.numberOfOutputNeurons)
        self.hiddenBiases = [np.zeros(self.numberOfHiddenLayerNeurons) for _ in range(self.numberOfHiddenLayers)]
        self.hiddenToOutputWeights = np.random.rand(self.numberOfOutputNeurons, self.numberOfHiddenLayerNeurons)
        self.hiddenToHiddenWeights = [np.random.rand(self.numberOfHiddenLayerNeurons, self.numberOfHiddenLayerNeurons)
                                      for _ in range(self.numberOfHiddenLayers - 1)]
        self.inputToHiddenWeights = np.random.rand(self.numberOfHiddenLayerNeurons, self.numberOfInputNeurons)

        self.expectedOutput = None
        self.cost = None
        self.inputsDataset = None
        self.outputsDataset = None
        self.inputLayer = None
        self.hiddenLayers = [np.zeros(self.numberOfHiddenLayerNeurons) for _ in range(self.numberOfHiddenLayers)]
        self.outputLayer = np.zeros(self.numberOfOutputNeurons)
        self.output = None
        self.numberCorrect = 0
        self.logValues = True

        if randomize:
            self.randomizeAll()
        self._log(f"{self.name} successfully created")

    def _log(self, message):
        if self.logValues:
            print(message)

    def randomizeAll(self):
        self.randomizeWeights()
        self.randomizeBiases()
        self._log("Randomized all")

    def randomizeWeights(self):
        self.inputToHiddenWeights = np.random.rand(self.numberOfHiddenLayerNeurons, self.numberOfInputNeurons)
        self.hiddenToHiddenWeights = [np.random.rand(self.numberOfHiddenLayerNeurons, self.numberOfHiddenLayerNeurons)
                                      for _ in range(self.numberOfHiddenLayers - 1)]
        self.hiddenToOutputWeights = np.random.rand(self.numberOfOutputNeurons, self.numberOfHiddenLayerNeurons)
        self._log("Randomized weights")

    def randomizeBiases(self):
        self.hiddenBiases = [np.random.rand(self.numberOfHiddenLayerNeurons) for _ in range(self.numberOfHiddenLayers)]
        self.outputBiases = np.random.rand(self.numberOfOutputNeurons)
        self._log("Randomized biases")

    def feedForward(self):
        self.hiddenLayers[0] = tanh(np.dot(self.inputToHiddenWeights, self.inputLayer) + self.hiddenBiases[0])
        for i in range(1, self.numberOfHiddenLayers):
            self.hiddenLayers[i] = tanh(
                np.dot(self.hiddenToHiddenWeights[i - 1], self.hiddenLayers[i - 1]) + self.hiddenBiases[i])
        self.outputLayer = sigmoid(np.dot(self.hiddenToOutputWeights, self.hiddenLayers[-1]) + self.outputBiases)
        self._log("Feed Forward Complete")

    def calculateCost(self):
        self.cost = np.sum((self.outputLayer - self.expectedOutput) ** 2)
        self._log(f"Cost: {self.cost}")

    def setExpectedOutput(self, expected_output):
        self.expectedOutput = expected_output
        self._log("Expected output set")

    def backPropagate(self, learning_rate=0.1):
        # Ensure that all required variables are NumPy arrays
        self.inputLayer = np.array(self.inputLayer)
        self.hiddenLayers = [np.array(layer) for layer in self.hiddenLayers]

        # Calculate output layer error
        output_errors = (self.expectedOutput - self.outputLayer) * tanh_derivation(self.outputLayer)

        # Calculate hidden layer errors
        hidden_errors = [np.zeros(self.numberOfHiddenLayerNeurons) for _ in range(self.numberOfHiddenLayers)]
        hidden_errors[-1] = np.dot(self.hiddenToOutputWeights.T, output_errors) * tanh_derivation(
            self.hiddenLayers[-1])
        for i in range(self.numberOfHiddenLayers - 2, -1, -1):
            hidden_errors[i] = np.dot(self.hiddenToHiddenWeights[i].T, hidden_errors[i + 1]) * tanh_derivation(
                self.hiddenLayers[i])

        # Update output layer weights and biases
        self.hiddenToOutputWeights += learning_rate * np.dot(output_errors[:, np.newaxis],
                                                             self.hiddenLayers[-1][np.newaxis, :])
        self.outputBiases += learning_rate * output_errors

        # Update hidden layer weights and biases
        for i in range(self.numberOfHiddenLayers - 1, 0, -1):
            self.hiddenToHiddenWeights[i - 1] += learning_rate * np.dot(hidden_errors[i][:, np.newaxis],
                                                                        self.hiddenLayers[i - 1][np.newaxis, :])
            self.hiddenBiases[i] += learning_rate * hidden_errors[i]
        self.inputToHiddenWeights += learning_rate * np.dot(hidden_errors[0][:, np.newaxis],
                                                            self.inputLayer[np.newaxis, :])
        self.hiddenBiases[0] += learning_rate * hidden_errors[0]

        self._log("Back Propagation Complete")

    def loadInputData(self, dataset):
        self.inputsDataset = dataset
        self._log("Input Dataset loaded")

    def loadOutputData(self, dataset):
        self.outputsDataset = dataset
        forceLog("Output Dataset loaded")

    def fullCycle(self, dataset_iteration, learning_rate=0.1):
        if self.inputsDataset is None or self.outputsDataset is None:
            self._log("Error: Dataset(s) missing")
            return
        self.inputLayer = self.inputsDataset[dataset_iteration]
        self.setExpectedOutput(self.outputsDataset[dataset_iteration])
        self._log(f'Expected Output: {self.expectedOutput}')
        self.feedForward()
        self._log(f'Output: {self.outputLayer}')
        self.chooseOutput()
        self._log(f'Selected Output: {self.output}')
        self.calculateCost()
        self.backPropagate(learning_rate)
        self._log("Full Cycle Complete")

    def epoch(self, learning_rate=0.1, epochs=1):
        for i in range(epochs):
            for j in range(len(self.inputsDataset)):
                self.fullCycle(j, learning_rate)
            self.randomizeData()
            forceLog("Epoch Complete")

    def getDatasetLength(self):
        return len(self.inputsDataset)

    def storeModel(self, path):
        with open(path, "wb") as storeFile:
            pkl.dump(self, storeFile)
        self._log("Model stored")

    def loadModel(self, path):
        with open(path, "rb") as loadFile:
            loadedModel = pkl.load(loadFile)
            self.importModel(loadedModel)
        self._log("Model loaded")

    def importModel(self, model):
        self.__dict__.update(model.__dict__)

    def chooseOutput(self):
        self.output = np.argmax(self.outputLayer)
        if self.numberCorrect is None:
            self.numberCorrect = 0
        if self.output == np.argmax(self.expectedOutput):
            self.numberCorrect += 1
            self._log(f"Output: {self.output} is correct")
        else:
            self._log(f"Output: {self.output} is incorrect")

    def calculateAccuracy(self):
        return f"Accuracy: {self.numberCorrect / len(self.inputsDataset) * 100}"

    def testModel(self, test_inputs=None, test_outputs=None):
        if test_inputs is not None and test_outputs is not None:
            self.inputsDataset = test_inputs
            self.outputsDataset = test_outputs
        self.randomizeData()
        self.numberCorrect = 0
        for i in range(len(self.inputsDataset)):
            if self.inputsDataset is None or self.outputsDataset is None:
                self._log("Error: Dataset(s) missing")
                return
            self.inputLayer = self.inputsDataset[i]
            self.setExpectedOutput(self.outputsDataset[i])
            self.feedForward()
            self.chooseOutput()
        self._log("Test Complete")

    def randomizeData(self):
        indices = np.random.permutation(len(self.inputsDataset))
        self.inputsDataset = np.array(self.inputsDataset)[indices].tolist()
        self.outputsDataset = np.array(self.outputsDataset)[indices].tolist()

    def showLog(self, log_values):
        self.logValues = log_values
        self._log(f"Log values set to {self.logValues}")

    def __del__(self):
        self._log(f"{self.name} destroyed")
