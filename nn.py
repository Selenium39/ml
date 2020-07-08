import numpy as np


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x));

#MSE(均方误差损失)
def mse_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()


# 神经元
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # 前馈
    def feed_forward(self, inputs):
        # Weight inputs,add bias,then use actication function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


# 神经网络
class NeuralNetWork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0;
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feed_forward(self, x):
        out_h1 = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)
        out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))
        return out_o1


if __name__ == '__main__':
# weights = np.array([0, 1])
# bias = 4
# n = Neuron(weights, bias)
# inputs = np.array([2, 3])
# print(n.feed_forward(inputs))
   network=NeuralNetWork()
   print(network.feed_forward(np.array([2,3])))
   print(mse_loss(np.array([1,0,0,1]),np.array([0,0,0,0])))
