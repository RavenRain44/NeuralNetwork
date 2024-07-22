import Network

newNetwork = Network.NeuralNetwork("Test Network", 2, 2, 2, 2)
print(newNetwork.inputLayer)
print(newNetwork.hiddenLayers)
print(newNetwork.outputLayer)
print(newNetwork.inputToHiddenWeights)
print(newNetwork.hiddenToHiddenWeights)
print(newNetwork.hiddenToOutputWeights)
print(newNetwork.hiddenBiases)
print(newNetwork.outputBiases)