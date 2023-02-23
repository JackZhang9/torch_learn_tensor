# _*_ Author: JackZhang9 _*_
# _*_ Time: 20230223 16:28 _*_
import torch
import numpy as np

'''torch里面的变量是tensor，tensor的创建'''
a=torch.rand((3,4))  # 创建符合均匀分布的随机的张量
print('0-1之间均匀分布={}'.format(a))

a1=torch.rand_like(a)  # 创建和a类似的张量
print('类似的分布={}'.format(a1))

a2=torch.randn((3,4)) # 创建符合正态分布的随机的张量
print('正态分布={}'.format(a2))

'''创建固定张量'''
a3=torch.zeros((3,4))  # 创建全0张量
print('全0张量={}'.format(a3))

a4=torch.ones((3,4))  # 创建全1张量
print('全1张量={}'.format(a4))

a5=torch.zeros_like(a2)  # 创建形状类似的全0张量
print('形状类似全0={}'.format(a5))

a6=torch.ones_like(a2)  # 创建形状类似的全1张量
print('形状类似全1={}'.format(a6))

a7=torch.Tensor([[1,2],[2,3]])
print('创建简单张量={}'.format(a7))

np_array=np.random.randint(0,100,(3,2))
a8=torch.from_numpy(np_array)  # 从numpy数组创建张量
print('从numpy数组创建={}'.format(a8))

a9=torch.arange(1,10)
print('一个一维张量={}'.format(a9))

a10=torch.linspace(0,10,5)
print('等差张量={}'.format(a10))

a11=torch.logspace(0,2,2)
print('等比张量={}'.format(a11))

a12=torch.eye(3,4)  # 创建一个指定形状的对角张量
print('对角张量={}'.format(a12))

a13=torch.randint(0,100,(3,4)) # 创建指定形状的随机整数张量
print('指定形状的随机整数张量={}'.format(a13))

a14=a13.resize(2,6)
print('变形状={}'.format(a14))

a15=a13.reshape(6,2)  #
print('reshape={}'.format(a15))

a16=a13.view(4,3)
print('变形状={}'.format(a16))

a17=a13.numpy()
print('转=numpy{}={}'.format(type(a17),a17))

# a18=a13.cuda()  # 放到gpu  ,易出错
# print('{}'.format(a18))


