import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

if __name__ == '__main__':
    #自定义数据集
    x = [1, 5, 1.5, 8, 1, 9]
    y = [2,8,1.8,8,0.6,11]
    plt.scatter(x,y)
    # plt.show()
    X=np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
    Y=[0,1,0,1,0,1]
    # print(X)
    #定义分类器
    clf=svm.SVC(kernel='linear',C=1.0)
    #输入数据建立模型
    clf.fit(X,Y)
    #预测
    print(clf.predict([[2,2]]))
    print(clf.predict([[8,8]]))