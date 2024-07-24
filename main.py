import Network
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
mnist = loadmat("./input/mnist-original.mat")

data = mnist["data"].T
data = data / 255.0
labels = mnist["label"][0]

mnistTestingData = data
mnistTestingLabel = labels

mnistTestingLabelList = []
for i in range(len(mnistTestingLabel)):
    mnistTestingLabelList.append([])
    for j in range(10):
        mnistTestingLabelList[i].append(1 if mnistTestingLabel[i] == j else 0)

image_array = np.array(mnistTestingData[0]).reshape(28, 28)

plt.imshow(image_array, cmap='gray', vmin=0, vmax=1)
plt.title(f'Label: {mnistTestingLabel[0]}')
plt.axis('off')
plt.show()

testNetwork = Network.NeuralNetwork("Test Network", 2, 784, 16, 10)

testNetwork.loadInputData(mnistTestingData)
testNetwork.loadOutputData(mnistTestingLabelList)

testNetwork.randomizeData()

testNetwork.epoch(0.1, 10)
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
