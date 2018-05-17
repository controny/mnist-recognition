# mnist-recognition
Simple neural network to recognize MNIST.

## 环境
Linux, Python3（推荐3.5以上）

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
使用默认参数训练：
```
python train.py
```
指定参数训练，如：
```
python train.py --epochs 50 --validating_gap 5 --batch_size 10 --learning_rate 1.0 --reg_lambda 0.1 --model_name MyModel
```
各项参数的含义如下：

- `epochs`: 迭代的轮数，默认为50
- `validating_gap`: 进行验证（会输出当前的准确率并记录当前的loss）的间隔轮数，默认为1
- `batch_size`: 每个训练batch的大小，默认为10
- `learning_rate`: 学习率，默认为1.0
- `reg_lambda`: 正则化项的lambda参数, 默认为0.0
- `model_name`: 模型的名称，默认为model

### Test
测试默认模型：
```
python test.py
```
测试指定模型，如：
```
python test.py --model_name MyModel
```

### Inference
使用默认模型预测默认的`images`文件夹中的图片：
```
python inference.py
```
使用指定模型预测指定文件夹中的图片，如：
```
python inference.py --model_name MyModel --images_dir MyImages/
```

### 查看训练loss统计图
查看默认模型的loss统计图：
```
python check_summary.py
```
查看指定模型的loss统计图：
```
python check_summary.py --model_name MyModel
```

