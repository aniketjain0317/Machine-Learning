import math
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def g(x):
    return 1/(1+ math.exp(-x))

class NeuralNetwork:
    def __init__(self , feature_no , class_no , layerArray=None, reg_val=0):
        if layerArray is None:
            self.layerArray = [feature_no, feature_no + 1, class_no]
            self.L =3
        else:
            self.layerArray=layerArray
            self.L = len(layerArray)
        self.feature_no = feature_no
        self.class_no=class_no
        self.theta = self.constructTheta()
        self.reg_val=reg_val

    #This just puts out random data for now, will change later
    def processData(self,data,train=1):
        example_no = data.shape[0]
        label= data.columns[-1]
        df = pd.get_dummies(data,columns = [data.columns[-1]])
        y_values =  np.unique(data.iloc[:,-1])

        X = df.iloc[:,:self.feature_no].to_numpy()
        Y = df.iloc[:,self.feature_no:].to_numpy()
        if train==1:
            self.X = X
            self.Y = Y
            self.example_no = example_no
            self.y_values = y_values
            self.label = label
        return X,Y, example_no

    #Constructs a theta according to the layers and is initialized with random values from the range: (-initial_range, initial_range)
    def constructTheta(self , initial_range=1):
        theta=[]
        for l in range(1,self.L):
            theta.append((2*np.random.rand(self.layerArray[l],self.layerArray[l-1]+1) - 1)*initial_range)
        return theta

    #constructs the activation values array, initialized to one.
    #x is the feature values (layer 1)
    #the value of bias determines whether a one is added in the beginning of the arrays or not (i.e the bias unit which is always one)
    def constructNodes(self,x=None,bias=1):
        a=[]
        for l in range(self.L):
            a.append(np.ones(self.layerArray[l]+bias))
        if not (x is None):
            for i in range(self.feature_no):
                a[0][i+1]=x[i]
        return a

    #this is the hypothesis function
    #given x (the values of the feature nodes), it returns the final layer values
    #bias just removes the bias units in each layer
    #give_a makes it return the entire activation value array instead of only the hypothesis
    def forwardProp(self,x, theta=None,give_a=False,bias=1):
        if theta is None:
            theta=self.theta
        a = self.constructNodes(x,bias=1)
        for l in range(1,self.L):
                z=(theta[l-1] @ a[l-1])
                a[l][1:]= np.array([g(k) for k in z])
        hyp = a[self.L-1][1:]
        if not bias:
            for l in range(self.L):
                a[l] = a[l][1:]
        if give_a:
            ans = a
        else:
            ans = hyp
        return ans

    # NOT SURE IF CORRECT
    #this is the cost function
    def costFn(self, theta=None, test_data=None, AddToSelf=1, meanSE=0):
        if theta is None:
            theta = self.theta
        if test_data is None:
            X, Y, example_no = self.X, self.Y, self.example_no
        else:
            X, Y, example_no = self.processData(test_data)
        jval=0
        for i in range(example_no):
            x=X[i]
            y=Y[i]
            h = self.forwardProp(x,theta)
            if meanSE:
                h = np.array([1/self.class_no] * self.class_no)
            for k in range(self.class_no):
                jval -= np.sum(y@np.log(h) + (1-y)*np.log(1-h))
        for l in range(self.L-1):
            jval += self.reg_val * np.sum(theta[l][:,1:]**2) / 2
        jval /= example_no


        if AddToSelf and not meanSE:
            self.cost = jval
        return jval

    def errorSE(self,theta=None,test_data=None):
        accuracy = 1 - (self.costFn(theta,test_data) / self.costFn(theta,test_data,meanSE = 1))
        return accuracy
    #this calculates the gradient through backpropogation
    #there is some subtle problem in it, which makes it a bit more inefficient than it should be
    def calcGradient(self, theta=None, data=None):
        if theta is None:
            theta = self.theta
        if data is None:
            X, Y, example_no = self.X, self.Y, self.example_no
        else:
            X, Y, example_no = self.processData(data)
        delta = self.constructTheta(initial_range = 0)
        for i in range(example_no):
            x=X[i]
            y=Y[i]
            a = self.forwardProp(x,theta,give_a = True)
            hyp = a[self.L-1][1:]
            rho = self.constructNodes(x)
            rho[self.L-1] = hyp - y
            for l in range(self.L-2,0,-1):
                gz = np.multiply(a[l],1-a[l])
                rho[l] = np.multiply(theta[l].T @ rho[l+1],gz)[1:]
                delta[l] = delta[l] + np.outer(rho[l+1],a[l])
            delta[0] = delta[0] + np.outer(rho[1],a[0])

        grad = copy.deepcopy(delta)
        for l in range(self.L-1):
            for i in range(self.layerArray[l+1]):
                grad[l][i][0] /= example_no
                for j  in range(1,self.layerArray[l]+1):
                    grad[l][i][j] /= example_no
                    grad[l][i][j] += self.reg_val * theta[l][i][j]
        return grad


    #standard gradient descent algo
    def trainGradientDescent(self, data=None, theta=None, alpha=0.1, epsilon=0.1):
        if theta is None:
           theta =self.theta
        if data is None:
            X, Y, example_no = self.X, self.Y, self.example_no
        else:
            X, Y, example_no = self.processData(data)
        steps=1
        cost_prev=10000
        while True:
            # print(steps)
            if steps % 100 == 0:
                cost = self.costFn(theta,data)
                print(cost)
                if abs(cost_prev - cost) < epsilon:
                    break
                cost_prev=cost
            grad = self.calcGradient(theta)
            # correct = self.numGradCheck(theta,data)
            # print("theta",theta)
            # print("grad" , grad)
            # print("correct", correct)
            for l in range(self.L-1):
                theta[l] = theta[l] - alpha*grad[l]
            steps+=1
            # print("grad ", grad)
        self.theta=theta
        print("Steps Taken: ",steps)
        print("Theta: ",theta)
        return steps


    #this is a way of calculating gradients directly by measuring the change in cost function
    #unfortunately, this is also quite slow
    #epsilon is the change made in x (dx) in order to measure the change in y (dy)
    def numGradCheck(self,theta,data, epsilon = 0.001):
        correct =  self.constructTheta(0)
        for l in range(self.L-1):
            a,b = theta[l].shape
            for i in range(a):
                for j in range(b):
                    tplus=copy.deepcopy(theta)
                    tminus= copy.deepcopy(theta)
                    tplus[l][i][j] = tplus[l][i][j] + epsilon
                    tminus[l][i][j] = tminus[l][i][j] - epsilon
                    correct[l][i][j]= (self.costFn(tplus,data) - self.costFn(tminus,data))/ (2 * epsilon)
        return correct

    def test(self, test_data, th=None):
        if th is None:
            theta=self.theta
        else:
            theta=th
        X,Y, test_no = self.processData(test_data)
        for i in range(test_no):
            x = X[i]
            y= Y[i]
            hyp = self.forwardProp(x)
            print("Id: ",i+1)
            print("Hyp: ",hyp)
            print("Actual: ", y)
        accuracy = self.errorSE(test_data=test_data)
        print("Accuracy: ",accuracy*100,"%")


# model = NeuralNetwork(2,1)
# t = model.theta
# print(t)
# h = model.forwardProp([],t)
# print(h)
# c = model.costFn(t,[],1)
# print(c)
# g = model.calcGradient(t,[])
# print("g",g)
# thet = model.trainGradientDescent([1,2],t)
# print(thet)
