import flask
from flask import request,jsonify
from model import ClassifyModel
app=flask.Flask(import_name=__name__)


# 创建模型对象
model=ClassifyModel()

@app.route('/',methods=['GET','POST'])
@app.route('/index',methods=['GET','POST'])  # 一般是post请求
def index():
    print('进入服务器内部方法')
    return 'hello,flask'

@app.route('/t1',methods=['GET','POST'])
@app.route('/t1/<sex>',methods=['GET','POST'])
def t1(sex='male'):
    _args_dict=None
    _method=request.method
    if _method == 'GET':
        _args_dict=request.args   # 字典
    elif _method == 'POST':
        _args_dict=request.form
    else:
        raise ValueError('{}方式不支持'.format(_method))
    name=_args_dict['name']
    age=_args_dict.get('age',-1)
    print('请求方式:{} 获取参数:{} {} {}'.format(_method,name,age,sex))
    print('传入参数，数据处理，调用相关代码，返回结果json，字典形式')
    result={
        'code':200,
        'msg':'操作成功',
        'data':[
            {
                'name':name,
                'age':age,
                'sex':sex
            },
            {
                'name':'{}'.format(name*2),
                'age':age
            }
        ]
    }
    return jsonify(result)

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        # 获取参数
        _args_dict = request.args if request.method == 'GET' else request.form
        text=_args_dict.get('text', ' ').strip()
        # 相关逻辑代码执行
        if len(text)==0:
            res = {
                'code': 202,
                'msg': '请求参数异常，给定有效长度的text文本参数'
            }
        else:
            pre_class,pre_prob=ClassifyModel.interface(text)
            res={
                'code':200,
                'msg':'执行成功',
                'data':{
                    'text':text,
                    'class_name':pre_class,
                    'probability':float(pre_prob)
                }
            }
    except Exception as e:
        print('EE{}'.format(e))
        res={
            'code':201,
            'msg':'服务器异常{}'.format(e),
            'data':{
                'text':text

            }

        }
    # 返回结果
    return jsonify(res)

if __name__ == '__main__':
    '''启动'''
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
    print(flask.__version__)







