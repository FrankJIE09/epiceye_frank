# EpicEye SDK for Python

## 使用方法
### 独立模块使用
#### 安装依赖
```shell
pip install -r requirements.txt
```

#### 运行示例程序

自动搜索相机，并选择第一个
```shell
python example.py
```

手动指定相机ip

假设ip为127.0.0.1
```shell
python example.py 127.0.0.1
```

### 安装库使用

#### 安装epiceye到python环境中
```shell
python setup.py install
```

#### 打包epiceye为whl安装包并安装
```shell
python setup.py bdist_wheel
pip install ./dist/epiceye-3.2.1.20221110-py3-none-any.whl
```

#### 在项目中引用
```python
import epiceye
```