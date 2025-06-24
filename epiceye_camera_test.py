#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EpicEye相机全面测试程序
包括重复精度、帧率、稳定性等测试
"""

import time
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from epiceye_camera_api import EpicEyeCamera, search_cameras


class NumpyEncoder(json.JSONEncoder):
    """ 自定义编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    success: bool
    data: Dict
    error_message: Optional[str] = None
    execution_time: float = 0.0


class EpicEyeCameraTester:
    """EpicEye相机测试类"""
    
    def __init__(self, camera_ip: Optional[str] = None):
        """
        初始化测试器
        
        Args:
            camera_ip: 相机IP地址，如果为None则自动搜索
        """
        self.camera_ip = camera_ip
        self.camera: Optional[EpicEyeCamera] = None
        self.test_results: List[TestResult] = []
        
        # 设置日志
        self.logger = logging.getLogger("EpicEyeCameraTester")
        self.logger.setLevel(logging.INFO)
        
        # 创建结果目录
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 测试参数
        self.test_params = {
            'frame_rate_test_duration': 10,  # 帧率测试持续时间（秒）
            'repeatability_test_count': 50,  # 重复精度测试次数
            'stability_test_duration': 60,   # 稳定性测试持续时间（秒）
            'accuracy_test_distance': 1000,  # 精度测试距离（mm）
            'noise_test_count': 100,         # 噪声测试次数
        }
    
    def setup_camera(self) -> bool:
        """设置相机连接"""
        try:
            if self.camera_ip is None:
                cameras = search_cameras()
                if not cameras:
                    self.logger.error("未找到相机")
                    return False
                self.camera_ip = cameras[0].ip
                self.logger.info(f"使用自动搜索到的相机: {self.camera_ip}")
            
            self.camera = EpicEyeCamera(self.camera_ip)
            if not self.camera.is_connected():
                self.logger.error("相机连接失败")
                return False
            
            self.logger.info("相机连接成功")
            return True
            
        except Exception as e:
            self.logger.error(f"设置相机失败: {e}")
            return False
    
    def run_test(self, test_func, test_name: str, **kwargs) -> TestResult:
        """运行单个测试"""
        start_time = time.time()
        result = TestResult(test_name=test_name, success=False, data={})
        
        try:
            self.logger.info(f"开始测试: {test_name}")
            data = test_func(**kwargs)
            result.data = data
            result.success = True
            self.logger.info(f"测试完成: {test_name}")
            
        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"测试失败 {test_name}: {e}")
        
        result.execution_time = time.time() - start_time
        self.test_results.append(result)
        return result
    
    def test_basic_functionality(self) -> Dict:
        """基础功能测试"""
        data = {
            'camera_info': None,
            'camera_config': None,
            'camera_matrix': None,
            'distortion': None,
            'single_capture': None,
            'point_cloud_stats': None
        }
        
        # 获取相机信息
        info = self.camera.get_info()
        if info:
            data['camera_info'] = {
                'sn': info.sn,
                'ip': info.ip,
                'model': info.model,
                'alias': info.alias,
                'width': info.width,
                'height': info.height
            }
        
        # 获取相机配置
        config = self.camera.get_config()
        if config:
            data['camera_config'] = {
                'exposure_time': config.exposure_time,
                'gain': config.gain,
                'laser_power': config.laser_power,
                'laser_mode': config.laser_mode,
                'filter_mode': config.filter_mode,
                'point_cloud_mode': config.point_cloud_mode
            }
        
        # 获取相机参数
        camera_matrix = self.camera.get_camera_matrix()
        if camera_matrix is not None:
            data['camera_matrix'] = camera_matrix.tolist()
        
        distortion = self.camera.get_distortion()
        if distortion is not None:
            data['distortion'] = distortion.tolist()
        
        # 单次拍摄测试
        capture_data = self.camera.capture_and_get_all()
        if capture_data['success']:
            data['single_capture'] = {
                'frame_id': capture_data['frame_id'],
                'has_image': capture_data['image'] is not None,
                'has_pointcloud': capture_data['pointcloud'] is not None,
                'has_depth': capture_data['depth'] is not None
            }
            
            # 点云统计
            if capture_data['pointcloud'] is not None:
                stats = self.camera.get_point_cloud_statistics(capture_data['pointcloud'])
                data['point_cloud_stats'] = stats
        
        return data
    
    def test_frame_rate(self) -> Dict:
        """帧率测试"""
        duration = self.test_params['frame_rate_test_duration']
        frame_times = []
        frame_ids = []
        success_count = 0
        total_count = 0
        
        start_time = time.time()
        end_time = start_time + duration
        
        self.logger.info(f"开始帧率测试，持续时间: {duration}秒")
        
        while time.time() < end_time:
            frame_start = time.time()
            
            # 触发拍摄
            frame_id = self.camera.capture_frame(pointcloud=False)  # 不包含点云以提高速度
            
            if frame_id:
                frame_end = time.time()
                frame_times.append(frame_end - frame_start)
                frame_ids.append(frame_id)
                success_count += 1
            else:
                self.logger.warning("拍摄失败")
            
            total_count += 1
        
        # 计算统计信息
        if frame_times:
            frame_times = np.array(frame_times)
            fps = len(frame_times) / duration
            avg_frame_time = np.mean(frame_times)
            std_frame_time = np.std(frame_times)
            min_frame_time = np.min(frame_times)
            max_frame_time = np.max(frame_times)
            
            # 计算实际帧率（去除异常值）
            frame_times_filtered = frame_times[frame_times < np.percentile(frame_times, 95)]
            fps_filtered = len(frame_times_filtered) / duration
            
        else:
            fps = fps_filtered = avg_frame_time = std_frame_time = min_frame_time = max_frame_time = 0
        
        return {
            'duration': duration,
            'total_attempts': total_count,
            'successful_captures': success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'fps': fps,
            'fps_filtered': fps_filtered,
            'avg_frame_time': avg_frame_time,
            'std_frame_time': std_frame_time,
            'min_frame_time': min_frame_time,
            'max_frame_time': max_frame_time,
            'frame_times': frame_times[:100]  # 只保存前100个时间用于分析
        }
    
    def test_repeatability(self) -> Dict:
        """重复精度测试"""
        count = self.test_params['repeatability_test_count']
        point_clouds = []
        center_points = []
        success_count = 0
        
        self.logger.info(f"开始重复精度测试，测试次数: {count}")
        
        for i in range(count):
            # 拍摄并获取点云
            data = self.camera.capture_and_get_all()
            
            if data['success'] and data['pointcloud'] is not None:
                pointcloud = data['pointcloud']
                
                # 计算点云中心点（去除无效点）
                valid_points = pointcloud[~np.isnan(pointcloud).any(axis=2)]
                valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
                
                if len(valid_points) > 0:
                    center = np.mean(valid_points, axis=0)
                    center_points.append(center)
                    point_clouds.append(pointcloud)
                    success_count += 1
                
                self.logger.debug(f"重复精度测试进度: {i+1}/{count}")
            else:
                self.logger.warning(f"第{i+1}次拍摄失败")
        
        if len(center_points) < 2:
            return {'error': '有效数据不足，无法计算重复精度'}
        
        center_points = np.array(center_points)
        
        # 计算重复精度统计
        mean_center = np.mean(center_points, axis=0)
        std_center = np.std(center_points, axis=0)
        max_deviation = np.max(np.linalg.norm(center_points - mean_center, axis=1))
        
        # 计算点云密度变化
        densities = []
        for pc in point_clouds:
            valid_points = pc[~np.isnan(pc).any(axis=2)]
            valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
            density = len(valid_points) / (pc.shape[0] * pc.shape[1])
            densities.append(density)
        
        densities = np.array(densities)
        
        return {
            'test_count': count,
            'success_count': success_count,
            'success_rate': success_count / count,
            'mean_center': mean_center.tolist(),
            'std_center': std_center.tolist(),
            'max_deviation': max_deviation,
            'mean_density': np.mean(densities),
            'std_density': np.std(densities),
            'center_points': center_points.tolist(),
            'densities': densities.tolist()
        }
    
    def test_stability(self) -> Dict:
        """稳定性测试"""
        duration = self.test_params['stability_test_duration']
        interval = 1.0  # 每秒拍摄一次
        measurements = []
        start_time = time.time()
        
        self.logger.info(f"开始稳定性测试，持续时间: {duration}秒")
        
        while time.time() - start_time < duration:
            test_start = time.time()
            
            # 拍摄并获取数据
            data = self.camera.capture_and_get_all()
            
            if data['success']:
                measurement = {
                    'timestamp': time.time() - start_time,
                    'frame_id': data['frame_id'],
                    'has_image': data['image'] is not None,
                    'has_pointcloud': data['pointcloud'] is not None,
                    'has_depth': data['depth'] is not None
                }
                
                # 计算点云统计
                if data['pointcloud'] is not None:
                    stats = self.camera.get_point_cloud_statistics(data['pointcloud'])
                    measurement['point_cloud_stats'] = stats
                
                measurements.append(measurement)
            
            # 等待到下一个间隔
            elapsed = time.time() - test_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        
        # 分析稳定性
        if measurements:
            success_rate = len(measurements) / int(duration / interval)
            
            # 分析点云统计的稳定性
            if any('point_cloud_stats' in m for m in measurements):
                valid_points_counts = [m['point_cloud_stats']['valid_points'] 
                                     for m in measurements if 'point_cloud_stats' in m]
                mean_distances = [m['point_cloud_stats']['mean_distance'] 
                                for m in measurements if 'point_cloud_stats' in m]
                
                stability_metrics = {
                    'valid_points_mean': np.mean(valid_points_counts),
                    'valid_points_std': np.std(valid_points_counts),
                    'mean_distance_mean': np.mean(mean_distances),
                    'mean_distance_std': np.std(mean_distances)
                }
            else:
                stability_metrics = {}
        else:
            success_rate = 0
            stability_metrics = {}
        
        return {
            'duration': duration,
            'total_measurements': len(measurements),
            'success_rate': success_rate,
            'stability_metrics': stability_metrics,
            'measurements': measurements[:100]  # 只保存前100个测量值
        }
    
    def test_noise(self) -> Dict:
        """噪声测试"""
        count = self.test_params['noise_test_count']
        depth_maps = []
        success_count = 0
        
        self.logger.info(f"开始噪声测试，测试次数: {count}")
        
        for i in range(count):
            data = self.camera.capture_and_get_all(pointcloud=False)
            
            if data['success'] and data['depth'] is not None:
                depth_maps.append(data['depth'])
                success_count += 1
            
            if (i + 1) % 10 == 0:
                self.logger.debug(f"噪声测试进度: {i+1}/{count}")
        
        if len(depth_maps) < 2:
            return {'error': '有效数据不足，无法分析噪声'}
        
        depth_maps = np.array(depth_maps)
        
        # 计算噪声统计
        # 假设相机和场景静止，深度变化主要由噪声引起
        mean_depth = np.mean(depth_maps, axis=0)
        std_depth = np.std(depth_maps, axis=0)
        
        # 计算有效区域的噪声
        valid_mask = ~np.isnan(mean_depth) & ~np.isinf(mean_depth)
        if np.any(valid_mask):
            noise_mean = np.mean(std_depth[valid_mask])
            noise_std = np.std(std_depth[valid_mask])
            noise_max = np.max(std_depth[valid_mask])
        else:
            noise_mean = noise_std = noise_max = 0
        
        return {
            'test_count': count,
            'success_count': success_count,
            'success_rate': success_count / count,
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'noise_max': noise_max,
            'mean_depth_shape': mean_depth.shape,
            'valid_pixels_ratio': np.sum(valid_mask) / valid_mask.size if valid_mask.size > 0 else 0
        }
    
    def test_accuracy(self) -> Dict:
        """精度测试（需要标准目标）"""
        # 这个测试需要标准目标，这里提供一个基础框架
        test_count = 10
        measurements = []
        
        self.logger.info("开始精度测试")
        
        for i in range(test_count):
            data = self.camera.capture_and_get_all()
            
            if data['success'] and data['pointcloud'] is not None:
                pointcloud = data['pointcloud']
                
                # 这里应该添加标准目标的检测和测量
                # 例如：检测棋盘格角点、测量已知距离等
                
                # 临时使用点云中心作为测量点
                valid_points = pointcloud[~np.isnan(pointcloud).any(axis=2)]
                valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
                
                if len(valid_points) > 0:
                    center = np.mean(valid_points, axis=0)
                    measurements.append(center)
        
        if len(measurements) < 2:
            return {'error': '有效测量数据不足'}
        
        measurements = np.array(measurements)
        
        return {
            'test_count': test_count,
            'measurement_count': len(measurements),
            'measurements': measurements.tolist(),
            'mean_measurement': np.mean(measurements, axis=0).tolist(),
            'std_measurement': np.std(measurements, axis=0).tolist(),
            'note': '需要标准目标进行更精确的精度测试'
        }
    
    def run_all_tests(self) -> List[TestResult]:
        """运行所有测试"""
        if not self.setup_camera():
            self.logger.error("相机设置失败，无法进行测试")
            return []
        
        try:
            # 基础功能测试
            self.run_test(self.test_basic_functionality, "基础功能测试")
            
            # 帧率测试
            self.run_test(self.test_frame_rate, "帧率测试")
            
            # 重复精度测试
            self.run_test(self.test_repeatability, "重复精度测试")
            
            # 稳定性测试
            self.run_test(self.test_stability, "稳定性测试")
            
            # 噪声测试
            self.run_test(self.test_noise, "噪声测试")
            
            # 精度测试
            self.run_test(self.test_accuracy, "精度测试")
            
        finally:
            if self.camera:
                self.camera.disconnect()
        
        return self.test_results
    
    def save_results(self, filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"test_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # 准备保存的数据
        save_data = {
            'test_timestamp': time.time(),
            'camera_ip': self.camera_ip,
            'test_params': self.test_params,
            'results': []
        }
        
        for result in self.test_results:
            result_data = {
                'test_name': result.test_name,
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'data': result.data
            }
            save_data['results'].append(result_data)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        self.logger.info(f"测试结果已保存: {filepath}")
        return filepath
    
    def generate_report(self, output_file: str = None):
        """生成测试报告"""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"test_report_{timestamp}.md"
        
        filepath = self.results_dir / output_file
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# EpicEye相机测试报告\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"相机IP: {self.camera_ip}\n\n")
            
            # 测试参数
            f.write("## 测试参数\n\n")
            for key, value in self.test_params.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
            
            # 测试结果摘要
            f.write("## 测试结果摘要\n\n")
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results if r.success)
            f.write(f"- 总测试数: {total_tests}\n")
            f.write(f"- 成功测试数: {successful_tests}\n")
            f.write(f"- 成功率: {successful_tests/total_tests*100:.1f}%\n\n")
            
            # 详细结果
            f.write("## 详细测试结果\n\n")
            for result in self.test_results:
                f.write(f"### {result.test_name}\n\n")
                f.write(f"- 状态: {'成功' if result.success else '失败'}\n")
                f.write(f"- 执行时间: {result.execution_time:.2f}秒\n")
                
                if result.error_message:
                    f.write(f"- 错误信息: {result.error_message}\n")
                
                if result.success and result.data:
                    f.write("- 测试数据:\n")
                    f.write("```json\n")
                    f.write(json.dumps(result.data, indent=2, ensure_ascii=False, cls=NumpyEncoder))
                    f.write("\n```\n")
                
                f.write("\n")
        
        self.logger.info(f"测试报告已生成: {filepath}")
        return filepath


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试器
    tester = EpicEyeCameraTester()
    
    # 运行所有测试
    print("开始EpicEye相机全面测试...")
    results = tester.run_all_tests()
    
    # 保存结果
    results_file = tester.save_results()
    
    # 生成报告
    report_file = tester.generate_report()
    
    # 打印摘要
    print("\n" + "="*50)
    print("测试完成！")
    print(f"结果文件: {results_file}")
    print(f"报告文件: {report_file}")
    
    successful_tests = sum(1 for r in results if r.success)
    print(f"成功测试: {successful_tests}/{len(results)}")
    
    # 显示关键结果
    for result in results:
        if result.success and result.data:
            if result.test_name == "帧率测试":
                fps = result.data.get('fps', 0)
                print(f"帧率: {fps:.2f} FPS")
            elif result.test_name == "重复精度测试":
                max_dev = result.data.get('max_deviation', 0)
                print(f"重复精度: {max_dev:.3f} mm")


if __name__ == "__main__":
    main() 