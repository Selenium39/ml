from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__ == "__main__":
    cancer = load_breast_cancer()  # 载入数据
    x = cancer.data  # 数据
    y = cancer.target  # 对应标签
    # print(x.shape) #(569, 30) 569个样本 30 个特征
    # print(y[y==1].shape)  #357个阳性样本
    # print(y[y==0].shape)  #212个阴性样本
    # print(cancer.feature_names)  #特征名

    # 区分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # 使用模型训练和评分
    model = LogisticRegression(max_iter=3000)
    # 输入训练数据
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    # print('train score',train_score)
    # print('test score',test_score)
    # 输入测试集数据查看训练结果
    result = model.predict(x_test)
    print('训练结果:', result)
    r_result = model.score(x_test, y_test)
    print('识别成功率为:', r_result)
    # 预测
    x_new = np.array([[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01, 3.001e-01,
                       1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01, 8.589e+00, 1.534e+02,
                       6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02, 3.003e-02, 6.193e-03, 2.538e+01,
                       1.733e+01, 1.846e+02, 2.019e+03, 1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01,
                       4.601e-01, 1.189e-01]])
    prediction = model.predict(x_new)
    print("Predicted target name:{}".format(cancer["target_names"][prediction]))
