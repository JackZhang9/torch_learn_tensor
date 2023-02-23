# _*_ Author: JackZhang9 _*_
# _*_ Time: 20230223 17:08 _*_
import torch

'''张量的基本运算，加减乘，转置'''
a=torch.randn((2,3))
b=torch.randint(0,100,(3,2))
c=torch.randint(0,50,(3,2))
'''张量乘法'''
mul=torch.matmul(a,b.to(torch.float32))  # 用.to(torch.float32)转化为float类型数值
print('{}*{}={}'.format(a.shape,b.shape,mul.shape))
print('mul={}'.format(mul))

'''张量加法'''
ad=torch.add(b,c)
print('加法={}'.format(ad))

'''张量减法'''
div=torch.div(b,c)
print('减法={}'.format(div))


'''张量的转置'''
transpo=mul.T
print('转置={}'.format(transpo))

transpo1=torch.transpose(mul,1,0)
print('转置2={}'.format(transpo1))



