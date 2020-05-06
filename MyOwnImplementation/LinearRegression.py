import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class LearningModel:
    def __init__( self, dfTrain,label='y'):
        self.label = label
        self.dfTrain = dfTrain
        self.X, self.y, self.feature_no, self.example_no = self.processData(self.dfTrain)
        self.trained = 0
        self.tested = False

    def processData( self, df):
        feature_no = df.shape[1]
        example_no = df.shape[0]
        X = np.append(np.ones((example_no, 1)),     df.drop(columns=[self.label]), axis=1)
        y = df[self.label].to_numpy()

        return X,y, feature_no, example_no

    def scale( self ):
        minArr = np.amin(self.X[:, 1:], axis=0)
        maxArr = np.amax(self.X[:, 1:], axis=0)
        self.X[:, 1:] = self.X[:, 1:] / (maxArr - minArr)


    def costFn( self, theta, test_data):

        X, y, feature_no, example_no = self.processData(test_data)
        cost = np.sum((X @ theta.T - y) ** 2) / (2 * example_no)
        return cost

    def trainGradientDescent( self, alpha=0.01, epsilon=0.0001 ):
        self.theta = np.ones(self.feature_no, )
        self.steps = 0
        self.alpha = alpha
        self.epsilon = epsilon

        grad = self.theta
        while (abs(grad) > self.epsilon).any():
            self.steps += 1
            grad = (self.X.T @ (self.X @ self.theta.T - self.y)) / self.example_no
            self.theta = self.theta - self.alpha * grad
        self.tr_cost = self.costFn(self.theta, self.dfTrain)
        self.trained = 1

    def trainNormalEqn( self ):
        self.theta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

        self.tr_cost = self.costFn(self.theta, self.dfTrain)
        self.trained = 2


    def test( self, dfTest):
        self.dfTest = dfTest
        mean =  dfTest[self.label].mean()
        meanHyp = np.concatenate(([mean],np.zeros(self.feature_no-1)))
        costMean = self.costFn(meanHyp, dfTest)
        costTest = self.costFn(self.theta, dfTest)
        self.r_squared = 1 - costTest / costMean
        self.tested = True

    def showTrainResult( self ):
        print("Theta: ", self.theta)
        if self.trained==1:
            print("Steps Taken: ",self.steps)
        print("CostFn: ", self.tr_cost)

    def showTestResult( self ):
        print("Theta: ", self.theta)
        print("Accuracy: ", self.r_squared)

        x_lin = np.linspace(0,10,100)
        y_lin = self.theta[0] + self.theta[1] * x_lin
        plt.plot(self.dfTrain.X,self.dfTrain.y,'go')
        plt.plot(self.dfTest.X,self.dfTest.y,'ro')
        plt.plot(x_lin,y_lin,'b-')
        plt.axis([0,20,0,20])
        plt.show()



# x1 = [[1, 2,3], [2, 3,4], [3,4, 5], [4, 5,6], [5, 6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,110]]
# x10 = [[1,2,4],[3,4,8],[5,6,16]]
# x11 = np.array(x10,dtype=float)
# df2 = pd.DataFrame(x11,columns=['X','Z','y'])
# x2 = np.array(x1, dtype=float)
# df = pd.DataFrame(x2,columns=['X','Z','y'])
# model = LearningModel(df,'y')
# model.trainGradientDescent()
# model.showTrainResult()
# model.test(df)
# model.showTestResult()
