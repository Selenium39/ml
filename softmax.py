import numpy as np

#通过softmax我们可以通过概率回答分类问题
def softmax(arr):
    return np.exp(arr)/sum(np.exp(arr))

if __name__ == '__main__':
    arr=np.array([-1,0,3,5])
    print(softmax(arr))