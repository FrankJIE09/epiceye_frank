# EpicEye相机API与测试系统

本项目提供了完整的EpicEye相机API封装和全面的测试系统，包括重复精度、帧率、稳定性等测试功能。

## 文件结构

```
epiceyesdk_samples_project/
├── epiceye_camera_api.py      # 相机API封装类
├── epiceye_camera_test.py     # 全面测试程序
├── test_results_template.md   # 测试报告模板
├── README_相机测试.md         # 本文件
├── epiceye/                   # 原始SDK
├── example.py                 # 原始示例
└── test_results/              # 测试结果目录（自动创建）
```

## 功能特性

### EpicEyeCamera API类 (`epiceye_camera_api.py`)

- **完整的相机控制**: 连接、配置、拍摄、数据获取
- **多种数据格式**: 图像、点云、深度图
- **自动参数管理**: 相机矩阵、畸变参数、去畸变查找表
- **数据统计**: 点云统计信息、质量评估
- **文件保存**: 自动保存图像、点云、深度图
- **错误处理**: 完善的异常处理和日志记录
- **上下文管理**: 支持with语句自动管理连接

### 测试系统 (`epiceye_camera_test.py`)

- **基础功能测试**: 验证所有基本功能
- **帧率测试**: 测量最大帧率和稳定性
- **重复精度测试**: 评估测量重复性
- **稳定性测试**: 长时间运行稳定性
- **噪声测试**: 测量噪声水平
- **精度测试**: 绝对精度评估（需要标准目标）
- **自动报告生成**: JSON数据和Markdown报告

## 安装依赖

```bash
pip install numpy opencv-python open3d matplotlib requests
```

## 快速开始

### 1. 基础使用

```python
from epiceye_camera_api import EpicEyeCamera

# 自动搜索并连接相机
camera = EpicEyeCamera()

if camera.is_connected():
    # 拍摄并获取所有数据
    data = camera.capture_and_get_all()
    
    if data['success']:
        # 保存数据
        camera.save_data(data, "my_capture")
        
        # 显示点云统计
        if data['pointcloud'] is not None:
            stats = camera.get_point_cloud_statistics(data['pointcloud'])
            print("点云统计:", stats)
    
    camera.disconnect()
```

### 2. 指定IP连接

```python
# 使用指定IP连接
camera = EpicEyeCamera("192.168.1.100")
```

### 3. 上下文管理器

```python
# 自动管理连接
with EpicEyeCamera() as camera:
    if camera.is_connected():
        data = camera.capture_and_get_all()
        camera.save_data(data)
```

### 4. 运行全面测试

```python
from epiceye_camera_test import EpicEyeCameraTester

# 创建测试器
tester = EpicEyeCameraTester()

# 运行所有测试
results = tester.run_all_tests()

# 保存结果和生成报告
tester.save_results()
tester.generate_report()
```

### 5. 命令行运行测试

```bash
python epiceye_camera_test.py
```

## API详细说明

### EpicEyeCamera类

#### 初始化
```python
camera = EpicEyeCamera(ip=None, auto_connect=True)
```

#### 主要方法

**连接管理**
- `connect()`: 连接相机
- `disconnect()`: 断开连接
- `is_connected()`: 检查连接状态

**信息获取**
- `get_info()`: 获取相机信息
- `get_config()`: 获取相机配置
- `get_camera_matrix()`: 获取相机矩阵
- `get_distortion()`: 获取畸变参数

**拍摄和数据获取**
- `capture_frame(pointcloud=True)`: 触发拍摄
- `get_image(frame_id)`: 获取图像
- `get_point_cloud(frame_id)`: 获取点云
- `get_depth(frame_id)`: 获取深度图
- `capture_and_get_all(pointcloud=True)`: 拍摄并获取所有数据

**数据处理**
- `undistort_image(image)`: 图像去畸变
- `get_point_cloud_statistics(pointcloud)`: 点云统计
- `save_data(data, prefix)`: 保存数据

**配置管理**
- `set_config(config)`: 设置相机配置

### 测试系统

#### EpicEyeCameraTester类

**测试方法**
- `test_basic_functionality()`: 基础功能测试
- `test_frame_rate()`: 帧率测试
- `test_repeatability()`: 重复精度测试
- `test_stability()`: 稳定性测试
- `test_noise()`: 噪声测试
- `test_accuracy()`: 精度测试

**结果管理**
- `save_results(filename)`: 保存测试结果
- `generate_report(output_file)`: 生成测试报告

## 测试参数配置

可以在`EpicEyeCameraTester`类中修改测试参数：

```python
tester = EpicEyeCameraTester()
tester.test_params = {
    'frame_rate_test_duration': 10,    # 帧率测试持续时间（秒）
    'repeatability_test_count': 50,    # 重复精度测试次数
    'stability_test_duration': 60,     # 稳定性测试持续时间（秒）
    'noise_test_count': 100,           # 噪声测试次数
}
```

## 输出文件

### 测试结果文件
- `test_results/test_results_[timestamp].json`: 原始测试数据
- `test_results/test_report_[timestamp].md`: 测试报告
- `test_results/`: 测试过程中保存的图像和点云文件

### 数据文件
- `[prefix]_image_[timestamp].png`: 图像文件
- `[prefix]_pointcloud_[timestamp].ply`: 点云文件
- `[prefix]_depth_[timestamp].png`: 深度图文件

## 测试报告解读

### 关键指标

**帧率性能**
- 平均帧率: 相机的实际工作帧率
- 帧率稳定性: 帧时间的变化程度
- 成功率: 拍摄成功率

**重复精度**
- 最大偏差: 多次测量的最大偏差
- 标准差: 测量结果的标准差
- 点云密度: 有效点云的比例

**稳定性**
- 长期稳定性: 长时间运行的数据一致性
- 数据完整性: 有效数据的比例

**噪声水平**
- 平均噪声: 深度测量的噪声水平
- 噪声分布: 噪声的空间分布特征

## 故障排除

### 常见问题

1. **相机连接失败**
   - 检查网络连接
   - 确认相机IP地址
   - 检查防火墙设置

2. **拍摄失败**
   - 检查相机状态
   - 确认相机配置
   - 检查环境光照

3. **点云数据异常**
   - 检查激光功率设置
   - 确认目标表面特性
   - 检查工作距离

4. **测试程序异常**
   - 检查Python依赖
   - 确认SDK版本
   - 查看日志信息

### 日志查看

程序会输出详细的日志信息，包括：
- 连接状态
- 拍摄过程
- 错误信息
- 测试进度

## 扩展开发

### 添加新的测试项目

```python
def test_custom_function(self) -> Dict:
    """自定义测试函数"""
    # 实现测试逻辑
    return {
        'test_result': 'value',
        'metrics': {}
    }

# 在run_all_tests中添加
self.run_test(self.test_custom_function, "自定义测试")
```

### 自定义数据处理

```python
# 继承EpicEyeCamera类
class CustomCamera(EpicEyeCamera):
    def custom_processing(self, data):
        # 自定义数据处理
        pass
```

## 技术支持

如有问题，请检查：
1. 相机硬件连接
2. 网络配置
3. 软件依赖版本
4. 测试环境条件

## 版本历史

- v1.0: 初始版本，包含基础API和测试功能
- 支持EpicEye相机的基本功能
- 提供全面的测试系统
- 自动报告生成 