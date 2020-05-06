import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MyOwnImplementation.LinearRegression as linreg
import MyOwnImplementation.NeuralNetwork as nn
import math, copy





#
# plt.plot(x, x, label='linear')  # Plot some data on the (implicit) axes.
# plt.plot(x, x ** 2, label='quadratic')  # etc.
# plt.plot(x, x ** 3, label='cubic')
# plt.xlabel('x label')
# plt.ylabel('y label')
# plt.title("Simple Plot")
# plt.legend()
# plt.show()
# # while (abs(x2)>0).any():
#     x2-=1
#     print(x2)
# print(x1)
# print(x2)
#
# x=np.array([[1,2,3],[1,5,10.]])
# a,b = x.shape
# for i in range(a):
#     for j in range(b):
#         y=x
#         y[i][j]=0.11
#         print(y)
# reg_val = 0.1 equalizes the hypothesis, i.e it always is [0.33,0.33,0.33]
model = nn.NeuralNetwork(4,3,[4,3,3],reg_val = 0)

df = pd.read_csv("Datasets/Iris/iris.csv",index_col = "Id")

x = df.loc[(df["Species"]=="Iris-setosa")]
ab = df.iloc[:35]
cd = df.iloc[50:85]
ea = df.iloc[100:135]
traindf = pd.concat([ab,cd,ea])
x1 = df.iloc[35:50]
x2 = df.iloc[85:100]
x3=df.iloc[135:150]
testdf = pd.concat([x1,x2,x3])
model.processData(traindf)
model.trainGradientDescent(alpha = 0.5,epsilon = 0.01)
model.test(testdf)
