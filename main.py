import Network
from scipy.io import loadmat
mnist = loadmat("./input/mnist-original.mat")
mnistTestingData = mnist["data"].T
mnistTestingLabel = mnist["label"][0]
mnistTestingLabelList = []
for i in range(len(mnistTestingLabel)):
    mnistTestingLabelList.append([])
    for j in range(10):
        mnistTestingLabelList[i].append(1 if mnistTestingLabel[i] == j else 0)

print(mnistTestingLabelList[0])

testNetwork = Network.NeuralNetwork("Test Network", 2, 784, 16, 10)

# testNetwork.loadInputData(mnistTestingData)
# testNetwork.loadOutputData(mnistTestingLabelList)
# for i in range(len(mnistTestingData)):
#     testNetwork.fullCycle(i)
# testNetwork.storeModel("StoreFile.txt")

testNetwork.loadModel("StoreFile.txt")
testNetwork.loadInputData(mnistTestingData)
testNetwork.loadOutputData(mnistTestingLabelList)
for i in range(len(mnistTestingData)):
    testNetwork.testModel(i)
print(testNetwork.calculateAccuracy())
