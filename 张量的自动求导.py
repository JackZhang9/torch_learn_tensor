# _*_ Author: JackZhang9 _*_
# _*_ Time: 20230223 19:08 _*_
import torch

'''张量的自动求导'''
x=torch.rand((2,3))
w=torch.rand((3,4),requires_grad=True)
b=torch.rand((4),requires_grad=True)
# print('a{}'.format(a))
# print('b{}'.format(b))
# print('c{}'.format(c))

'''前向传播'''
d1=torch.matmul(x,w)+b
print('d1',d1)
loss1=torch.relu(d1)  # 激活函数
loss1=torch.sum(loss1)
print('loss',loss1)

'''反向传播'''
loss1.backward()
for i in range(10):
    lr=0.01
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)
    print('{}轮新的w的grad={}'.format(i,w))
    print('{}轮新的b的grad={}'.format(i,b))