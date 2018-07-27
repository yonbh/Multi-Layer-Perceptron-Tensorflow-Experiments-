"""Multi-layer Perceptron Network Experiments"""

# uncomment %matplotlib inline if using jupyter notebook to view plots inline
# %matplotlib inline

import time
import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Using MNIST datasets of images of numbers from 0-9
# 28 x 28 pixel size, so 784 total pixels per images
# 60000 training samples, 10000 testing samples
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class expFFNN:
    
    def __init__(self):
        
        # number of nodes in each hidden layer
        self.defNodeAmount = 100
        self.nNodesHidL1 = self.defNodeAmount
        self.nNodesHidL2 = self.defNodeAmount
        self.nNodesHidL3 = self.defNodeAmount
        self.nNodesHidL4 = self.defNodeAmount
        
        # number of class/output nodes
        self.nClass = 10
        # go through batches of 100 images and feed them into the network at a time batches by batches
        self.batch_size = 100
		
		# placeholders are just some place at any given time for some data to be shoved through the network
		# matrix = height x width
		# height = none, width is 784 as flattened out the 28x28
        self.x = tf.placeholder('float', ([None, 784]))
        self.y = tf.placeholder('float')

        self.h1Weights = tf.Variable(tf.truncated_normal(np.array([784, self.nNodesHidL1]), stddev=0.1, seed=1), name="h1W")
        self.h2Weights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL1, self.nNodesHidL2]), stddev=0.1, seed=1), name="h2W")
        self.h3Weights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL2, self.nNodesHidL3]), stddev=0.1, seed=1), name="h3W")
        self.h4Weights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL3, self.nNodesHidL4]), stddev=0.1, seed=1), name="h4W")
        self.outWeights = tf.Variable(tf.truncated_normal(np.array([self.nNodesHidL4, self.nClass]), stddev=0.1, seed=1), name="outW")        
        
        self.h1Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL1]), name="h1B")
        self.h2Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL2]), name="h2B")
        self.h3Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL3]), name="h3B")
        self.h4Biases = tf.Variable(tf.constant(0.1, shape=[self.nNodesHidL4]), name="h4B")
        self.outBiases = tf.Variable(tf.constant(0.1, shape=[self.nClass]), name="outB")

        self.finH1W = np.array([])
        self.finH2W = np.array([])
        self.finH3W = np.array([])
        self.finH4W = np.array([])        
        self.finOutW = np.array([])

        self.finH1B = np.array([])
        self.finH2B = np.array([])
        self.finH3B = np.array([])
        self.finH4B = np.array([])
        self.finOutB = np.array([])
        
        self.oriAccRec = [] # to store performance from a trained network
        self.accRec = [] # to store performance of using pretrained weights/biases
        self.accLossRec = [] # to store performance loss of paired swapping network layers experiment

        
    # Modelling the network    
    def neural_network_model(self, data):
		# using dictionary to define the shape of the hidden layers and shape of weights in line with the hidden layer
        h1Layer = {'weights':self.h1Weights,
                   'biases':self.h1Biases}

        h2Layer = {'weights':self.h2Weights,
                   'biases':self.h2Biases}

        h3Layer = {'weights':self.h3Weights,
                   'biases':self.h3Biases}
        
        h4Layer = {'weights':self.h4Weights,
                   'biases':self.h4Biases}

        outputLayer = {'weights':self.outWeights,
                   'biases':self.outBiases}
		
		# relu is like your threshold function
        layer1 = tf.add(tf.matmul(data, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        layer3 = tf.add(tf.matmul(layer2, h3Layer['weights']), h3Layer['biases']) 
        layer3 = tf.nn.relu(layer3)
        
        layer4 = tf.add(tf.matmul(layer3, h4Layer['weights']), h4Layer['biases']) 
        layer4 = tf.nn.relu(layer4)

        output = tf.add(tf.matmul(layer4, outputLayer['weights']), outputLayer['biases']) 

        return output    
    
    
    # Training the network
	# x as input data
    def trainNeuralNetwork(self, x):
        y = self.y
        prediction = self.neural_network_model(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
		
		# how many epochs; feed forward + backpropogation = epoch (a cycle)
        hmEpochs = 5
        
        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hmEpochs):
                epochLoss = 0

                for _ in range(int(mnist.train.num_examples/self.batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict = {self.x:epoch_x, self.y:epoch_y})
                    epochLoss += c
                print ('Epoch:', (epoch + 1), 'completed out of', hmEpochs, '; loss:', epochLoss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:', accuracy.eval({x:mnist.test.images, self.y:mnist.test.labels}))
            
            self.oriAccRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100)
            
            # Retrieving trained values
            variableNames = [v.name for v in tf.trainable_variables()]
            wBValues = sess.run(variableNames)
            
            # Updating final weights and biases
            self.finH1W = wBValues[-10]
            self.finH2W =  wBValues[-9]
            self.finH3W = wBValues[-8]
            self.finH4W = wBValues[-7]
            self.finOutW = wBValues[-6]
            
            self.finH1B = wBValues[-5]
            self.finH2B = wBValues[-4]
            self.finH3B = wBValues[-3]
            self.finH4B = wBValues[-2]
            self.finOutB = wBValues[-1]
 
 
    # Attempt to use predefined weights and biases to run the network
    def nnPre(self, x):
        # Modelling and Initialisation using stored weights and variables

        h1Layer = {'weights':self.finH1W,
                   'biases':self.finH1B}

        h2Layer = {'weights':self.finH2W,
                   'biases':self.finH2B}

        h3Layer = {'weights':self.finH3W,
                   'biases':self.finH3B}
        
        h4Layer = {'weights':self.finH4W,
                   'biases':self.finH4B}

        outputLayer = {'weights':self.finOutW,
                   'biases':self.finOutB}

        layer1 = tf.add(tf.matmul(x, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        layer3 = tf.add(tf.matmul(layer2, h3Layer['weights']), h3Layer['biases']) 
        layer3 = tf.nn.relu(layer3)
        
        layer4 = tf.add(tf.matmul(layer3, h4Layer['weights']), h4Layer['biases']) 
        layer4 = tf.nn.relu(layer4)

        output = tf.add(tf.matmul(layer4, outputLayer['weights']), outputLayer['biases']) 


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
 
 
	# Shuffling weights of a layer node by node
    def shuffleWeights(self, layer, nNodesRdm):

        for n in range(0,nNodesRdm):
            #print("Before: ", layer[n][0])
            rd.shuffle(layer[n])
            #print("After: ", layer[n][0])
            print("Shuffled node", n+1, ":")
            self.nnPre(self.x)
 
 
	# Randomising weights of a layer node by node with predefined means and standard deviations (truncated)
    def randomiseWeights(self, layer, nNodesRdm):
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.normal(0, 0.1, len(layer[n]))
            layer[n] = rdVals
            print("Randomised node:", n+1)
            self.nnPre(self.x)
            print("---------------------------------")
 
 
	# Randomising weights of a layer node by node using default means and std deviation 
    def randomiseWeights2(self, layer, nNodesRdm):
        for n in range(0,nNodesRdm):
            # using len() as a form of validation check
            rdVals = np.random.random(len(layer[0]))
            layer[n] = rdVals
            print("Randomised node v2", n+1, ":")
            self.nnPre(self.x)   
            print("---------------------------------")            

			
    # Sequentially randomise a node and revert it back to previous values when moving on to the next node
	# The purpose of this method is to potentially identify in which portion of a layer would be more impact if manipulated
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
            
    
    # Swapping layers around
    def mixNN(self, x, h1W, h1B, h2W, h2B, h3W, h3B, h4W, h4B, outW, outB):
        # Modelling and Initialisation using stored weights and variables
        h1Layer = {'weights':h1W,
                   'biases':h1B}

        h2Layer = {'weights':h2W,
                   'biases':h2B}

        h3Layer = {'weights':h3W,
                   'biases':h3B}
        
        h4Layer = {'weights':h4W,
                   'biases':h4B}

        outputLayer = {'weights':outW,
                   'biases':outB}

        layer1 = tf.add(tf.matmul(x, h1Layer['weights']), h1Layer['biases']) 
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.add(tf.matmul(layer1, h2Layer['weights']), h2Layer['biases']) 
        layer2 = tf.nn.relu(layer2)	

        layer3 = tf.add(tf.matmul(layer2, h3Layer['weights']), h3Layer['biases']) 
        layer3 = tf.nn.relu(layer3)
        
        layer4 = tf.add(tf.matmul(layer3, h4Layer['weights']), h4Layer['biases']) 
        layer4 = tf.nn.relu(layer4)

        output = tf.add(tf.matmul(layer4, outputLayer['weights']), outputLayer['biases']) 


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
            
            self.accLossRec.append(accuracy.eval({self.x:mnist.test.images, y:mnist.test.labels})*100)
    
	
	# Generate a dictionary which maps node manipulated progress and performance after that specific manipulation
    def generateAccDict(self, numNodes, recordList):
        acc = recordList
        nodes = range(1,numNodes+1)
        accDic = dict(zip(nodes, acc))
        print(accDic)
		

# Example usage	(sequentially randomise nodes without reverting)	
"""
a = expFFNN()
a.trainNeuralNetwork(a.x)
now = time.time()
print("\n")
a.randomiseWeights(a.finH2W, len(a.finH2W))
then = time.time()
print("Time taken: {:0.2f}".format((then-now)/60/60), "hours.")

a.generateAccDict(a.defNodeAmount, a.accRec)

nr = a.defNodeAmount
randWFig = plt.figure()
plt.rcParams["figure.dpi"] = 200
plt.style.use("seaborn-pastel")
plt.plot(range(1,nr+1), a.accRec, linewidth=1, figure=randWFig)
plt.title("Randomise and Revert Weights of Nodes Of Layer 2")
plt.ylabel("Accuracies %")
plt.xlabel("Number of nodes shuffled")
plt.axis([0, nr, 0, 100])
plt.xticks(np.arange(0, nr+1, 5), fontsize=7)
plt.yticks(np.arange(0, 101, 10), fontsize=7)
plt.grid(linewidth=0.4, alpha=0.3)
randWFig.savefig("MLP_rdRevertWeights_L1.png", dpi=400)
plt.show()
"""
