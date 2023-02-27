class ClassifyModel(object):
    def __init__(self):
        self.name='good'
        self.prob=0.9

    def interface(self, text):
        '''
        针对给定的文本调用训练好的模型进行预测，预测结果为：预测类别、预测概率
        :param text:
        :return: 预测类别，预测概率
        '''
        if len(text)==0:
            raise ValueError('模型预测要求输入参数不为空，当前输入参数为{}'.format(text))
        return '正向评论', 0.985


if __name__ == '__main__':
    model=ClassifyModel()
    print(model.interface('是的是的是的'))