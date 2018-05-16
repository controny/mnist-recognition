# mnist-recognition
Simple neural network to recognize MNIST.

## 安装
```
# 克隆仓库
git clone https://github.com/controny/mnist-recognition
# 切换到项目源码目录
cd mnist-recognition
# 新建虚拟环境并指定Python版本，最好是Python3.5以上
virtualenv venv -p python3
# 激活虚拟环境
source venv/bin/activate
# 安装第三方模块
pip install -r requirements.txt
```

## 运行

### Train
使用默认参数：
```
python train.py
```
指定参数：
```
python train.py --epochs 50 --validating_gap 5 --batch_size 10 --learning_rate 1.0 --reg_lambda 0.1 --model_name MyModel
```

### Test
测试默认模型：
```
python test.py
```
测试指定模型：
```
python test.py --model_name MyModel
```