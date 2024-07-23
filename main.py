import Network
import random
import numpy as np
from scipy.io import loadmat
mnist = loadmat("./input/mnist-original.mat")

data = mnist["data"].T
labels = mnist["label"][0]

indices = np.random.permutation(len(data))

mnistTestingData = data[indices]
mnistTestingLabel = labels[indices]

mnistTestingLabelList = []
for i in range(len(mnistTestingLabel)):
    mnistTestingLabelList.append([])
    for j in range(10):
        mnistTestingLabelList[i].append(1 if mnistTestingLabel[i] == j else 0)

testNetwork = Network.NeuralNetwork("Test Network", 2, 784, 16, 10)

testNetwork.loadInputData(mnistTestingData)
testNetwork.loadOutputData(mnistTestingLabelList)
i = 0
for i in range(round(len(mnistTestingData)/2)):
    if -len(mnistTestingData) <= i * 2 < len(mnistTestingData):
        if i == 1:
            pass
        i += 1
        testNetwork.fullCycle(i * 2)
testNetwork.storeModel("StoreFile.txt")

# testNetwork.loadModel("StoreFile.txt")
# testNetwork.loadInputData(mnistTestingData)
# testNetwork.loadOutputData(mnistTestingLabelList)
# for i in range(round(len(mnistTestingData)/2)):
#     if -len(mnistTestingData) <= 1 + i * 2 < len(mnistTestingData):
#         testNetwork.setExpectedOutput(testNetwork.outputsDataset[1 + i * 2])
#         testNetwork.feedForward()
#         testNetwork.chooseOutput()
# print(testNetwork.calculateAccuracy())
