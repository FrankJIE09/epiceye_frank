#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EpicEye相机使用示例
演示如何使用EpicEyeCamera API类进行各种操作
"""

import time
import logging
from epiceye_camera_api import EpicEyeCamera, search_cameras


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 自动搜索相机
    cameras = search_cameras()
    if not cameras:
        print("未找到相机")
        return
    
    print(f"找到 {len(cameras)} 台相机:")
    for i, camera in enumerate(cameras):
        print(f"  {i+1}. {camera.model} ({camera.ip}) - {camera.sn}")
    
    # 连接第一台相机
    camera = EpicEyeCamera(cameras[0].ip)
    
    if not camera.is_connected():
        print("相机连接失败")
        return
    
    print(f"成功连接相机: {camera.get_info().model}")
    
    # 拍摄并获取数据
    print("拍摄中...")
    data = camera.capture_and_get_all()
    
    if data['success']:
        print("拍摄成功!")
        print(f"帧ID: {data['frame_id']}")
        print(f"图像: {'有' if data['image'] is not None else '无'}")
        print(f"点云: {'有' if data['pointcloud'] is not None else '无'}")
        print(f"深度图: {'有' if data['depth'] is not None else '无'}")
        
        # 保存数据
        camera.save_data(data, "example_capture")
        
        # 显示点云统计
        if data['pointcloud'] is not None:
            stats = camera.get_point_cloud_statistics(data['pointcloud'])
            print("\n点云统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print("拍摄失败")
    
    camera.disconnect()


def example_context_manager():
    """上下文管理器示例"""
    print("\n=== 上下文管理器示例 ===")
    
    # 使用with语句自动管理连接
    with EpicEyeCamera() as camera:
        if camera.is_connected():
            print("相机连接成功")
            
            # 获取相机信息
            info = camera.get_info()
            print(f"相机型号: {info.model}")
            print(f"分辨率: {info.width} x {info.height}")
            
            # 拍摄
            data = camera.capture_and_get_all()
            if data['success']:
                print("拍摄成功")
                camera.save_data(data, "context_example")
            else:
                print("拍摄失败")
        else:
            print("相机连接失败")


def example_configuration():
    """配置管理示例"""
    print("\n=== 配置管理示例 ===")
    
    camera = EpicEyeCamera()
    
    if camera.is_connected():
        # 获取当前配置
        config = camera.get_config()
        if config:
            print("当前配置:")
            print(f"  曝光时间: {config.exposure_time} ms")
            print(f"  增益: {config.gain}")
            print(f"  激光功率: {config.laser_power}%")
            print(f"  激光模式: {config.laser_mode}")
            print(f"  滤波模式: {config.filter_mode}")
            print(f"  点云模式: {config.point_cloud_mode}")
        
        # 获取相机参数
        camera_matrix = camera.get_camera_matrix()
        distortion = camera.get_distortion()
        
        if camera_matrix is not None:
            print(f"\n相机矩阵: {camera_matrix.shape}")
        if distortion is not None:
            print(f"畸变参数: {distortion.shape}")
    
    camera.disconnect()


def example_multiple_captures():
    """多次拍摄示例"""
    print("\n=== 多次拍摄示例 ===")
    
    camera = EpicEyeCamera()
    
    if camera.is_connected():
        print("开始连续拍摄...")
        
        for i in range(5):
            print(f"第 {i+1} 次拍摄...")
            
            # 拍摄
            frame_id = camera.capture_frame()
            if frame_id:
                print(f"  帧ID: {frame_id}")
                
                # 获取图像
                image = camera.get_image(frame_id)
                if image is not None:
                    print(f"  图像尺寸: {image.shape}")
                
                # 获取点云
                pointcloud = camera.get_point_cloud(frame_id)
                if pointcloud is not None:
                    stats = camera.get_point_cloud_statistics(pointcloud)
                    print(f"  有效点数: {stats.get('valid_points', 0)}")
                
                # 保存数据
                data = {
                    'frame_id': frame_id,
                    'image': image,
                    'pointcloud': pointcloud,
                    'depth': camera.get_depth(frame_id)
                }
                camera.save_data(data, f"capture_{i+1}")
            else:
                print("  拍摄失败")
            
            time.sleep(1)  # 等待1秒
    
    camera.disconnect()


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 尝试连接不存在的IP
    print("尝试连接不存在的IP...")
    camera = EpicEyeCamera("192.168.1.999", auto_connect=False)
    
    if not camera.connect():
        print("连接失败（预期结果）")
    
    # 尝试连接真实相机
    cameras = search_cameras()
    if cameras:
        camera = EpicEyeCamera(cameras[0].ip)
        
        if camera.is_connected():
            print("连接成功，测试错误处理...")
            
            # 尝试获取无效帧ID的数据
            image = camera.get_image("invalid_frame_id")
            if image is None:
                print("正确处理了无效帧ID")
            
            camera.disconnect()


def main():
    """主函数"""
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    print("EpicEye相机使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_context_manager()
        example_configuration()
        example_multiple_captures()
        example_error_handling()
        
        print("\n所有示例运行完成!")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 