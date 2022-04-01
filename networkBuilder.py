import pickle
import numpy as np
from numpy.core.fromnumeric import mean
import neuralNetwork as nn
import matplotlib.pyplot as plt
from os import listdir
from multiprocessing import cpu_count
import concurrent.futures as cf
from random import shuffle

#IMPORTS DATA
numCores = round(cpu_count()/4)
pickles = listdir("Chessboard Ratings/chessDataPickles")
pickles = pickles[:]
shuffle(pickles)

#Building the network
learningVal = 0.0025
keepRate = 0.8
batchSize = 128
numEpochs = 5 
numErrSamples = 2500

network = []
network.append(nn.Dense(768, 1048,learningVal))
network.append(nn.ReLU())
network.append(nn.Dense(1048, 500,learningVal))
network.append(nn.ReLU())
network.append(nn.Dense(500, 50,learningVal))
network.append(nn.ReLU())
network.append(nn.Dense(50, 1,learningVal))
network.append(nn.Linear())

'''This is for if you already have a network built
file = open("DNN_three.pickle", "rb")
network = pickle.load(file)
file.close()'''

#Running the tests
teE = []
trE = []
meanTrainingError = []
meanTestingError = []
t = 1
executor = cf.ThreadPoolExecutor(numCores)
threadSize = round(batchSize/numCores)

for epoch in range(1, numEpochs+1): 

    #Load the pickle
    index = 0
    for pic in pickles:
        index += 1
        print("Processing Pickle", index)
        file = open("Chessboard Ratings/chessDataPickles/" + str(pic), "rb")
        data = pickle.load(file)
        file.close()
    
    #TRAIN, TEST, SPLIT
        dataLen = round((len(data)/5)-0.5)

        X_train = np.zeros((dataLen*4, 768))
        y_train = np.zeros(dataLen*4)

        X_test = np.zeros((dataLen, 768))
        y_test = np.zeros(dataLen)
        for x in range(dataLen*5):
            if x < dataLen:
                X_test[x] = data[x][0]
                y_test[x] = data[x][1]
            else:
                X_train[x-dataLen] = data[x][0] 
                y_train[x-dataLen] = data[x][1]

        #Minibatch Training
        for x_batch,y_batch in nn.iterate_minibatches(X_train,y_train,batchsize=batchSize,shuffle=True):
            deltaW = None
            deltaB = None
            threads = []
            #Creates threads
            for i in range(numCores):
                future = executor.submit(nn.train, network,x_batch[i*threadSize:(i+1)*threadSize],y_batch[i*threadSize:(i+1)*threadSize], keepRate)
                threads.append(future) 
            #As each thread is completed it adds the deltaW and deltaB           
            for thread in cf.as_completed(threads):
                result = thread.result()
                if deltaW is None:
                    deltaW = result[0]
                    deltaB = result[1]
                else:
                    for i in range(len(deltaW)):
                        deltaW[i] += result[0][i]
                        deltaB[i] += result[1][i]
            #Update the layers weights
            for layer in range(len(network))[::-1]:
                if isinstance(network[layer], nn.Dense):
                    network[layer].update(t, deltaW[layer], deltaB[layer])
            #print(deltaW[layer][3])
            t += 1           
        trE.append(np.mean(np.sqrt(np.square(np.subtract(nn.predict(network, X_train).flatten(), y_train)))))
        teE.append(np.mean(np.sqrt(np.square(np.subtract(nn.predict(network, X_test).flatten(), y_test)))))
        meanTrainingError.append(mean(trE[-len(pickles):]))
        meanTestingError.append(mean(teE[-len(pickles):]))
    #Gets the error

    file = open("DNN_three.pickle", "wb")
    pickle.dump(network, file)
    file.close()

    print("Epoch",epoch)
    print("Training Error: " + str(mean(trE[-len(pickles):])) + ", Validation Error: "+ str(mean(teE[-len(pickles):])))

#Saving the Network
file = open("model.pickle", "wb")
pickle.dump(network, file)
file.close()

#Showing 
# \the graph of Error
plt.plot(meanTestingError, label='Validation Error')
plt.plot(meanTrainingError, label="Training Error")
plt.legend(loc='best')
plt.grid()
plt.show()
