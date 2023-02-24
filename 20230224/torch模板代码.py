import torch
import numpy as np
import torch.nn as nn

'''一个完整深度学习网络结构'''
class Mynetwork(nn.Module):  # 集成
    def __init__(self):
        super(Mynetwork, self).__init__()
        '''模型参数的初始化'''
        # self.w1=nn.Parameter(torch.randn((2,3)))
        # self.b1=nn.Parameter(torch.randn(3))
        # self.w2=nn.Parameter(torch.randn((3,2)))
        # self.b2=nn.Parameter(torch.randn(2))

        self.fc1 = nn.Linear(2, 3)  # 输入2，输出3
        self.fc2 = nn.Linear(3, 2)  # 输入3，输出2

    def forward(self,x):
        '''前向传播，x结构是(n,2)'''
        # hi_net1=torch.matmul(x,self.w1)+self.b1
        # out_net1=torch.sigmoid(hi_net1)
        # hi_net2 = torch.matmul(out_net1, self.w2) + self.b2
        # out_net2 = torch.sigmoid(hi_net2)

        hi_net1=self.fc1(x)
        out_net1=torch.sigmoid(hi_net1)
        hi_net2=self.fc2(out_net1)
        out_net2=torch.sigmoid(hi_net2)

        return out_net2

if __name__ == '__main__':
    n=640  # 数据总数
    batch_size=32
    '''加载数据'''
    x=torch.randn((n,2))
    y=torch.randn((n,2))  # label [n,2]
    # print('y={}'.format(y))

    '''构建网络'''
    my_net=Mynetwork()
    '''构建损失'''
    loss_func=nn.MSELoss()  # mse损失函数，回归损失函数，
    '''构建优化器'''
    optimi=torch.optim.SGD(my_net.parameters(),lr=0.01)
    # print('所有参数={}'.format(list(my_net.parameters())))
    # out_net=my_net.forward(x)
    '''循环训练'''
    for i in range(50):
        idxs=np.random.permutation(n)  # n是数据总数，生成n个随机索引
        '''按batch训练，需要加个for循环'''
        for batch in range(0,n,batch_size):  # 0~32, 32~64,
            start=batch           # 0,32
            end=batch+batch_size  # 32,64
            # print('sts={},end={}'.format(start,end))
            idxs_sub=idxs[start:end]  # 0~32, 32~64,
            # print('idx_sub=',idxs_sub,len(idxs_sub))
            x_sub=x[idxs_sub]
            # print('x_su->{} {}'.format(x_sub,len(x_sub)))
            y_sub=y[idxs_sub]
            # print('y_sub=',y_sub)
            '''前向传播'''
            out_net=my_net(x_sub)  # 前向传播得到预测值
            # print('out_net={}'.format(out_net))

            loss=loss_func(out_net,y_sub)

            '''反向传播'''
            optimi.zero_grad()
            loss.backward() # 反向传播
            # print(my_net.w1.grad)
            '''参数更新'''
            optimi.step()  # 参数更新
            # for w in my_net.parameters():  # 参数更新
            #     w.data.sub_(0.01*w.grad.data)
            print('loss->',loss)
    print('my_net->{}'.format(my_net(x)))


