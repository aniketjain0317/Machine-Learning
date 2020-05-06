#INCOMPLETE:
#   - test()
#   - showTrainResult()
#   - showTestResult()
#   - regularization
#
#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def g(x):
    return 1 / (1+ math.exp(-x))

class LearningModel:
    def __init__(self,dfTrain,label):
        self.label=label
        self.dfTrain=dfTrain
        self.X, self.y, self.feature_no, self.example_no = self.processData(self.dfTrain)
        self.trained = 0
        self.scaled = False
        self.tested = False


    def processData( self, df ):
        feature_no = df.shape[1]
        example_no = df.shape[0]
        X = np.append(np.ones((example_no, 1)), df.drop(columns=[self.label]), axis=1)

        minArr = np.amin(X[:, 1:], axis=0)
        maxArr = np.amax(X[:, 1:], axis=0)
        X[:, 1:] = X[:, 1:] / (maxArr - minArr)

        y = df[self.label].to_numpy()

        return X, y, feature_no, example_no

    def costFn( self,theta,test_data):
        X, y, feature_no, example_no = self.processData(test_data)
        h = np.array([g(x) for x in (X @ theta.T)])
        cost = -((y@np.log(h))+(1-y)@np.log(1 - h))
        return cost

    def trainGradientDescent( self, alpha=10, epsilon=0.001 ):
        self.theta = np.array([1 for i in range(self.feature_no)])
        self.steps = 0
        self.alpha = alpha
        self.epsilon = epsilon

        grad = self.theta
        while (abs(grad) > self.epsilon).any():
            h = np.array([g(x) for x in (self.X @ self.theta.T)])
            self.steps += 1
            grad = (self.X.T @ (h - self.y)) / self.example_no
            self.theta = self.theta - self.alpha * grad

        self.trained = 1
        self.costTrain=(self.costFn(self.theta,self.dfTrain))

    def test( self, dfTest ):
        self.dfTest = dfTest
        pass

    def showTrainResult( self ):
        pass

    def showTestResult( self ):
        pass
x1 = np.array([[0,1],[1,1],[2,1],[3,0],[4,0],[5,0]])
x2= np.array([[0,0],[1,0],[2,0],[3,1],[4,1],[5,1]])
df = pd.DataFrame(x1,columns=['X','y'])
model = LearningModel(df,'y')
model.trainGradientDescent()
