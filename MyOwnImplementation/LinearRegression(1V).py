import numpy as np

class LearningModel:
    def __init__(self, trainingData):
        self.tr_label = trainingData.label
        self.tr_data = trainingData.data
        self.tr_no = self.tr_data.shape[0]
        self.X = np.append(np.ones((self.tr_no,1)),self.tr_data[:,0:1],axis=1)
        self.y = self.tr_data[:,1]
        self.trained = 0
        self.tested = False
    def costFn( self, h=np.zeros(2,),test_data=np.zeros(2,)):
        if not(h.any()):
            h=self.h
        if not(test_data.any()):
            test_data=self.tr_data

        cost = 0.0
        for example in test_data:
            features_val = np.concatenate(([1.0], example[0:1]))
            y =example[1]
            cost += (float(h.T @ features_val) - y)**2
        cost = cost / (2*self.tr_no)
        return cost

    def calcGradient(self):
        grad = np.zeros(2,dtype=float)
        for example in self.tr_data:
            features_val = np.concatenate(([1.0],example[0:1]))
            y = example[1]
            update = features_val * (float(self.h.T @ features_val) - y)
            grad += update
        grad /= self.tr_no
        return grad

    def trainGradientDescent( self,alpha=0.01,epsilon=0.0001):
        self.h = np.ones(2,)
        self.steps =0
        self.alpha=alpha
        self.epsilon = epsilon
        self.epsilonArray = np.array([epsilon for i in range(2)])

        grad = np.array([1,1])
        while (abs(grad)>self.epsilonArray).any():
            self.steps += 1
            grad = self.calcGradient()
            self.h = self.h - self.alpha * grad

        self.tr_cost = self.costFn(self.h)
        self.trained=1

    def trainNormalEqn(self):
        self.h = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

        self.tr_cost = self.costFn(self.h)
        self.trained = 2

    def test( self, test_data):
        self.test_data=test_data
        mean = test_data[:,1].average()
        meanHyp = np.array([mean,0])
        costMean = self.costFn(meanHyp,test_data)
        costTest = self.costFn(self.h,test_data)
        self.r_squared = 1 - costTest/costMean
        self.tested = True

    def showTrainResult( self ):
        print("Hypothesis: ",self.h)
        # print("Steps Taken: ",self.steps)
        print("CostFn: ",self.tr_cost)

    def showTestResult( self ):
        print("Hypothesis: ", self.h)
        print("Accuracy: ",self.r_squared)


class Data:
    def __init__(self,data,label=1):
        self.data = data
        self.label = label
        self.exampleNo = data.shape[0]

x1 = [[1,3],[2,4],[3,5],[4,6],[5,7]]
x2= np.array(x1,dtype=float)
x3 = Data(x2)
x4 = LearningModel(x3)
x4.trainNormalEqn()