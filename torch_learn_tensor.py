# _*_ Author:JackZhang9 _*_
# _*_ Time:20230220 _*_
import torch
import numpy as np

'''一个学习使用torch中tensor的类'''
class torch_tensor:
    def __init__(self):
        pass

    def gener_tensor(self):
        '''使用numpy生成tensor'''
        a=np.random.randint(0,100,(10,))
        print(a)
        '''变成tensor格式'''
        a_tensor=torch.tensor(a,dtype=torch.float32,device='cuda')
        print(a_tensor,type(a_tensor))

        '''使用torch.randint生成tensor'''
        b=torch.randint(0,100,(10,),dtype=torch.float32,requires_grad=True)
        print('b={}'.format(b))

        '''使用torch.randn生成标准正态分布tensor'''
        c=torch.randn((10,))
        print('c={}'.format(c))

    def oper_func(self):
        '''tensor的矩阵乘法'''
        x=torch.randn((3,5))
        w=torch.randn((5,2))
        b=torch.randn((2))
        y=torch.matmul(x,w)+b

        print(y.shape)
        print(y)
        '''使用激活函数，shape不变'''
        print('sigmoid={}'.format(torch.sigmoid(y)))
        print('relu={}'.format(torch.relu(y)))
        print('tanh={}'.format(torch.tanh(y)))
        print('softmax={}'.format(torch.softmax(y,dim=1)))

    def oper_grid(self):
        '''求参数梯度'''
        w_h=torch.randn((20,20),requires_grad=True)
        w_x=torch.randn((20,10),requires_grad=True)
        x=torch.randn((1,10))
        h=torch.randn((1,20))

        '''前向传播'''
        h2h=torch.matmul(w_h,h.T)
        x2x=torch.matmul(w_x,x.T)
        x_h=h2h+x2x
        x_h_tanh=torch.tanh(x_h)

        '''loss'''
        loss=x_h_tanh.sum()
        print(loss)

        '''反向传播'''
        loss.backward()
        print('w_x={}'.format(w_x.grad))
        print('w_h={}'.format(w_h.grad))
        print('w={}'.format(h.grad))
        print('x={}'.format(x.grad))

if __name__ == '__main__':
    '''生成一个实例对象'''
    tensor_example=torch_tensor()
    tensor_example.gener_tensor()
    tensor_example.oper_func()
    tensor_example.oper_grid()





