import random
import numpy as np
import matplotlib.pyplot as plt

class neuralnet(object):

    def __init__(self, layers):
        self.layers = layers
        self.reset()

    def reset(self):
        self.b = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.w = [np.random.randn(y, x) for x,y in zip(self.layers[:-1], self.layers[1:])]
        self.epochs = []
        self.trainResult = []
        self.testResult = []
        self.ne = 0

    def output(self, input_layer):
        phi = input_layer
        layer = 0
        for b, w in zip(self.b, self.w):
            phi = self.sigmoid(np.dot(w, phi) + b)
        return phi

    def vector(self, y):
        vector = np.zeros((10,1))
        vector[y] = 1
        return vector
        
    def vectorize(self, data):
        results = [self.vector(y) for y in data]
        return results

    def fit(self, x, y, epochs, batch_size, eta, tx = None, ty = None):
        n = len(x)
        dy = self.vectorize(y)
        data = list(zip(x,dy))
        
        self.epochs.append(self.ne)
        self.trainResult.append(self.test(x, y)/len(data))      
                    
        if tx:
            self.testResult.append(self.test(tx, ty)/len(tx))

        for j in range(epochs):
            random.shuffle(data)
            batches = [data[k:k + batch_size] for k in range(0, n, batch_size)]
            for group in batches:
                self.update(group, eta)
            self.ne += 1
            self.epochs.append(self.ne)
            self.trainResult.append(self.test(x,y)/len(data))      
            if tx:
                self.testResult.append(self.test(tx, ty)/len(tx))
            
            print("... {}".format(j+1), end="")
                
    def update(self, data, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]
        
        for x, y in data:
            delta_nabla_b, delta_nabla_w = self.backPropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.w = [w-(eta/len(data))*nw for w, nw in zip(self.w, nabla_w)]
        self.b = [b-(eta/len(data))*nb for b, nb in zip(self.b, nabla_b)]
        
    def backPropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]
        nLayers = len(self.layers)
        
        activation = x
        activations = [x] 
        zs = [] 
        
        for b, w in zip(self.b, self.w):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
       
        delta = self.cost(activations[-1], y) * self.dsigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, nLayers):
            z = zs[-l]
            sp = self.dsigmoid(z)
            delta = np.dot(self.w[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost(self, output_activations, y):
        return (output_activations-y)

    def test(self, x,y, vectorized = False):
        data = list(zip(x,y))
        
        test_results = [(np.argmax(self.output(x)), y) for (x, y) in data]
            
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))
        
    def dsigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def plot(self):
        fig = plt.scatter(self.epochs, self.trainResult, color='green', alpha=0.8, label='Train')
        fig = plt.scatter(self.epochs, self.testResult, color='magenta', alpha=0.8, label='Test')
        plt.title("Acurácia para cada Epoch", fontsize=14)
        plt.xlabel('Epochs')
        plt.legend(loc='lower right')
        return fig
        