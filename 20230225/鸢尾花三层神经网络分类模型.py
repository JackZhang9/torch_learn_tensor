import torch
import numpy as np
import torch.nn as nn
import pandas as pd

'''iris数据集3层神经网络预测模型'''
class My_network(nn.Module):
    def __init__(self):
        super(My_network, self).__init__()
        '''模型参数'''
        self.fc1=nn.Linear(4,50)
        self.fc2=nn.Linear(50,30)
        self.fc3 = nn.Linear(30,3)

    def forward(self,x):
        '''前向传播'''
        hid_net1=self.fc1(x)
        out_net1=torch.relu(hid_net1)
        hid_net2 = self.fc2(out_net1)
        out_net2 = torch.sigmoid(hid_net2)
        hid_net3 = self.fc3(out_net2)
        # out_net3 = torch.sigmoid(hid_net3)

        return hid_net3

def data_pre_process(file_name):
    '''数据集划分成训练集、测试集'''
    data=pd.read_csv(file_name)
    # print(data.columns)  # ['Unnamed: 0', 'Sepal.Length', 'Sepal.Width', 'Petal.Length','Petal.Width', Species]
    # 先去掉x无关的列组成新的x,把dataframe转换成numpy类型的ndarray数组
    x=data.drop(['Unnamed: 0','Species'],axis=1).values   # 通过.values取到这些numpy类型数据
    x=x.astype(dtype=np.float32)
    # y有几个种类,用集合性质去重查看
    # print(set(data['Species'].values))  # {'versicolor', 'virginica', 'setosa'}
    data[data['Species'] == 'versicolor'] = 0  # 对每个种类对应换成 0,1,2数字表示
    data[data['Species'] == 'virginica'] = 1
    data[data['Species']== 'setosa'] = 2
    y=data['Species'].values  # 通过.values取到这些numpy类型数据
    y=y.astype(dtype=np.float32)  # y的数据类型转换为np.float64
    # 将numpy类型数据使用torch.from_numpy转换成张量tensor形式
    x_train=torch.from_numpy(x[:int(0.8*len(x))])
    y_train=torch.from_numpy(y[:int(0.8*len(x))])
    x_test=torch.from_numpy(x[int(0.8*len(x)):])
    y_test=torch.from_numpy(y[int(0.8*len(y)):])

    # print('trainx={} \n triany={}\n testx={}\n testy={}'.format(x_train,y_train,x_test,y_test))
    return x_train,y_train,x_test,y_test

if __name__ == '__main__':

    '''数据导入'''
    file_name=r'D:\深兰2132\torchself220\224\iris.csv'
    x_train,y_train,x_test,y_test=data_pre_process(file_name)
    x,y=x_train,y_train
    y=torch.asarray(y,dtype=torch.long)
    total_sample=len(x)
    '''模型构建'''
    my_net=My_network()

    '''loss构建'''
    loss_fun=nn.CrossEntropyLoss()

    '''优化器构建'''
    opt=torch.optim.SGD(my_net.parameters(),lr=0.1)

    '''模型循环训练'''
    for i in range(500):

        '''前向传播'''
        y_pre=my_net(x)
        # print(y_pre,y)
        loss=loss_fun(y_pre,y)

        print('第{}轮 loss->{}'.format(i,loss))
        '''反向传播'''
        opt.zero_grad()
        loss.backward()

        '''参数更新'''
        opt.step()


    print(y)
    res=my_net(x)
    print(res.max(dim=1).indices)
    '''测试集'''
    print(my_net(x_test).max(dim=1).indices)
