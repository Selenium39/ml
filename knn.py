from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    iris = load_iris()  # 加载数据集
    x = iris.data  # 数据
    y = iris.target  # 每个数据对应标签
    print(x.shape)
    k_range = range(1, 31)
    k_error = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # cv参数决定数据集划分，这里按照5:1划分训练集和测试集
        scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')
        k_error.append(1 - scores.mean())

    plt.plot(k_range, k_error)
    plt.xlabel('k')
    plt.ylabel('error')
    # plt.show()  #最合适k值为11

    # 数据预处理,分割数据为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    #特征工程:归一化处理
    std=StandardScaler()
    x_train=std.fit_transform(x_train)
    x_test=std.transform(x_test)

    #使用K-近邻算法
    knn=KNeighborsClassifier(n_neighbors=11)
    #输入训练集数据
    knn.fit(x_train,y_train)
    #输入测试集数据，查看训练结果
    result=knn.predict(x_test)
    r_result=knn.score(x_test,y_test)
    print('训练结果:',result)
    print('正确结果为:',y_test)
    print('识别成功率为:',r_result)

    #预测
    x_new=np.array([[5,2.9,1,0.2]])
    prediction=knn.predict(x_new)
    print("Predicted target name:{}".format(iris["target_names"][prediction]))