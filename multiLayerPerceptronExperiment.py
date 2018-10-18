########################################
#									   #
#  Multi-layer Perceptron Experiments  #
#									   #
########################################

# NOTE: Purely for experimental sake and implementations are not done with efficiency in mind.


import csv
import time
import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
# show matplotlib output inline when using jupyter notebook
%matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class expFFNN:
    
    def __init__(self):
        
        # number of layers in each hidden layer
        self.defNodeAmount = 100
        self.nNodesHidL1 = self.defNodeAmount
        self.nNodesHidL2 = self.defNodeAmount
        
        # number of class/output nodes
        self.nClass = 10
        # batch size used to train each time
        self.batch_size = 256

        self.x = tf.placeholder('float', ([None, 784]))
        self.y = tf.placeholder('float')
        
        self.weightSeed = None
        self.h1Weights = tf.Variable(tf.truncated_normal(np.array([784, self.nNodesHidL1]), seed=self.weightSeed, stddev=0.1), name="h1W")
        self.h2Weights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL1, self.nNodesHidL2]), seed=self.weightSeed, stddev=0.1), name="h2W")
        self.outWeights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL2, self.nClass]), seed=self.weightSeed, stddev=0.1), name="outW")        
        
        self.h1Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL1]), name="h1B")
        self.h2Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL2]), name="h2B")
        self.outBiases = tf.Variable(tf.constant(0.1, shape=[self.nClass]), name="outB")

        self.finH1W = np.array([])
        self.finH2W = np.array([])    
        self.finOutW = np.array([])

        self.finH1B = np.array([])
        self.finH2B = np.array([])
        self.finOutB = np.array([])
        
        self.oriAccRec = [] # to store performance from a trained network
        self.accRec = [] # to store performance of using pretrained weights/biases
        self.accMixRec = [] # to store performance loss of paired swapping network layers experiment
        
        self.nodeRec = [] # to store the order of nodes being manipulated
        self.accDic = [] # dictionary of nodes randomised:accuracy
        
        self.epochAcc = []
        self.epochLoss = []
        
        # learning rate for optimisation algorithms
        self.learningRate = 0.001
        
    # Modelling the network    
    def neural_network_model(self, data):
        h1Layer = {'weights':self.h1Weights,
                   'biases':self.h1Biases}

        h2Layer = {'weights':self.h2Weights,
                   'biases':self.h2Biases}

        outputLayer = {'weights':self.outWeights,
                   'biases':self.outBiases}

        layer1 = tf.add(tf.matmul(data, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        output = tf.add(tf.matmul(layer2, outputLayer['weights']), outputLayer['biases']) 

        return output    
    
    
    # Training the network
    def trainNeuralNetwork(self, x):
        y = self.y
        prediction = self.neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(cost)

        hmEpochs = 15
        
        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hmEpochs):
                epochLoss = 0
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                for _ in range(int(mnist.train.num_examples/self.batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict = {self.x:epoch_x, self.y:epoch_y})
                    epochLoss += c
                print ('Epoch:', (epoch + 1), 'completed out of', hmEpochs, '; loss:', epochLoss, '; Accuracy:', accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels}))
                self.epochLoss.append(epochLoss)
                self.epochAcc.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels}))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x:mnist.test.images, self.y:mnist.test.labels}))
            
            self.oriAccRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100)
            self.accRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100) 
            
            # Retrieving trained values
            variableNames = [v.name for v in tf.trainable_variables()]
            wBValues = sess.run(variableNames)
            
            # Updating final weights and biases
            self.finH1W = wBValues[-6]
            self.finH2W =  wBValues[-5]
            self.finOutW = wBValues[-4]
            
            self.finH1B = wBValues[-3]
            self.finH2B = wBValues[-2]
            self.finOutB = wBValues[-1]
            
            # to print variables and its values
            #for k, v in zip(variableNames[-8:], wBValues[-8:]):
             #   print("Variable: ", k)
             #   print("Shape: ", sess.run(tf.shape(v)))
             #   print(v)
  

    # Attempt to use predefined weights and biases to run the network
    def nnPre(self, x):
        # Modelling and Initialisation using stored weights and variables

        h1Layer = {'weights':self.finH1W,
                   'biases':self.finH1B}

        h2Layer = {'weights':self.finH2W,
                   'biases':self.finH2B}

        outputLayer = {'weights':self.finOutW,
                       'biases':self.finOutB}

        layer1 = tf.add(tf.matmul(x, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        output = tf.add(tf.matmul(layer2, outputLayer['weights']), outputLayer['biases']) 


        # Running the network
        y = self.y
        prediction = output
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            epochLoss = 0

            for _ in range(int(mnist.train.num_examples/self.batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
                c = sess.run(cost, feed_dict = {self.x:epoch_x, self.y:epoch_y})
                epochLoss += c
            print ('Predefined Weights/Biases -> Epoch loss:', epochLoss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Predefined Weights/Biases -> Accuracy:', accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels}))
            # store as accuracies as percentage into an array
            self.accRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100)    
    
    
    def shuffleWeights(self, layer, nNodesRdm):
        """
        TEST TO CHECK IF CHANGES WERE MADE 
        
        print("Length: ", len(self.finH2W))        
        print("Original full: ", self.finH2W)
        print("original", self.finH2W[0])
        rd.shuffle(self.finH2W[0])
        print("shuffled: ", self.finH2W[0])
        print("Full shuff: ", self.finH2W[0])
        print(len(self.finH2W[0]))
        print(max(self.finH2W[0]))
        print(min(self.finH2W[0]))
        """
        for n in range(0,nNodesRdm):
            #print("Before: ", layer[n][0])
            rd.shuffle(layer[n])
            #print("After: ", layer[n][0])
            print("Shuffled node", n+1, ":")
            self.nnPre(self.x)
            

    # Swapping layers around
    def mixNN(self, x, h1W, h1B, h2W, h2B, outW, outB):
        # Modelling and Initialisation using stored weights and variables
        h1Layer = {'weights':h1W,
                   'biases':h1B}

        h2Layer = {'weights':h2W,
                   'biases':h2B}

        outputLayer = {'weights':outW,
                   'biases':outB}

        layer1 = tf.add(tf.matmul(x, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        output = tf.add(tf.matmul(layer2, outputLayer['weights']), outputLayer['biases']) 

        # Running the network
        y = self.y
        prediction = output
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            epochLoss = 0

            for _ in range(int(mnist.train.num_examples/self.batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
                c = sess.run(cost, feed_dict = {self.x:epoch_x, self.y:epoch_y})
                epochLoss += c
            print ('Pre-Mix Layers -> Epoch loss:', epochLoss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Pre-Mix Layers -> Accuracy:', accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels}))
            
            self.accMixRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100)
            
            
    # truncated version of randomising
    def randomiseWeights(self, layer, nNodesRdm):
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.normal(0, 0.1, len(layer[n]))
            layer[n] = rdVals
            print("Randomised node:", n+1)
            self.nnPre(self.x)
            print("---------------------------------")
    
    
    # non truncated version of randomising
    def randomiseWeights2(self, layer, nNodesRdm):
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.random(len(layer[0]))
            layer[n] = rdVals
            print("Randomised node v2", n+1, ":")
            self.nnPre(self.x)   
            print("---------------------------------")
    
    
    # non sequential randomisation method of randomiseWeights()
    def nonSeqRand(self, layer, nNodesRdm):
        # a way to force non repeating random numbers so eventually all nodes in layer will be manipulated
        # shuffling an arranged list of number so simulate non repeating randomised numbers
        rdNumList = np.arange(0,len(layer))
        rd.shuffle(rdNumList)
        
        self.nodeRec = np.copy(rdNumList)
        for n in range(0,nNodesRdm):
            rdVals = np.random.normal(0, 0.1, len(layer[rdNumList[n]]))
            layer[rdNumList[n]] = rdVals
            print("Non-Seq-Randomised node:", rdNumList[n])
            print(n+1, "out of", len(layer), "completed.")
            self.nnPre(self.x)
            print("---------------------------------")            
    
    
    # sequentially randomise nodes and reverting the previous one back to original when moving to next node
    def rdRevertWeights(self, layer, nNodesRdm):
        # to remember the orginal weight values for reverting purposes
        # need to use np.copy or else the memory reference will be same and changing one will affect the other as well
        originLayer = np.copy(layer)
              
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.normal(0, 0.1, len(layer[n]))
            layer[n] = rdVals
            print("RANDOMISED Node:", n+1)
            self.nnPre(self.x)
            layer[n] = originLayer[n]
            print("REVERTED Node:", n+1)
            print("---------------------------------")
        
        
    # non sequential version of rdRevertWeights()
    def nonSeqRevert(self, layer, nNodesRdm):
        originLayer = np.copy(layer)
        rdNumList = np.arange(0,len(layer[0]))
        rd.shuffle(rdNumList)
        self.nodeRec = np.copy(rdNumList)
        
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.normal(0, 0.1, len(layer[rdNumList[n]]))
            layer[rdNumList[n]] = rdVals
            print("Non-Seq-RANDOMISED Node:", rdNumList[n])
            self.nnPre(self.x)
            layer[rdNumList[n]] = originLayer[rdNumList[n]]
            print("NS-REVERTED Node:", rdNumList[n])
            print("---------------------------------")        
        
        
    # generate a dictionary containing performance when a specific node or number of nodes has been manipulated        
    def generateAccDict(self, numNodes, recordList):
        acc = recordList
        nodes = range(0,numNodes)
        self.accDic = dict(zip(nodes, acc))
        print(self.accDic)
		

# Example usage

    # network = expFFNN()
    # network.trainNeuralNetwork(network.x)