#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EpicEye相机API类
提供完整的相机控制功能，包括图像获取、点云处理、参数配置等
"""

import json
import time
import numpy as np
import cv2
import open3d as o3d
from typing import Optional, Tuple, Dict, List, Union
import logging
from dataclasses import dataclass
from enum import Enum

import epiceye


class CameraStatus(Enum):
    """相机状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class CameraInfo:
    """相机信息数据类"""
    sn: str
    ip: str
    model: str
    alias: str
    width: int
    height: int
    # 添加可选字段以兼容相机返回的完整数据
    version: Optional[str] = None
    firmwareVersion: Optional[str] = None
    hardwareVersion: Optional[str] = None
    cameraModel: Optional[int] = None
    ipv4SubnetMask: Optional[str] = None
    ipv6: Optional[str] = None
    isDhcpEnabled: Optional[bool] = None


@dataclass
class CameraConfig:
    """相机配置数据类"""
    exposure_time: float
    gain: float
    laser_power: float
    laser_mode: str
    filter_mode: str
    point_cloud_mode: str
    Selected: Optional[str] = None
    ProjectorBrightness: Optional[int] = None

    def __init__(self, **kwargs):
        self.exposure_time = kwargs.get('exposure_time')
        self.gain = kwargs.get('gain')
        self.laser_power = kwargs.get('laser_power')
        self.laser_mode = kwargs.get('laser_mode')
        self.filter_mode = kwargs.get('filter_mode')
        self.point_cloud_mode = kwargs.get('point_cloud_mode')
        self.Selected = kwargs.get('Selected')
        self.ProjectorBrightness = kwargs.get('ProjectorBrightness')
        
        # 接受所有其他未知的关键字参数
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class EpicEyeCamera:
    """EpicEye相机API类"""
    
    def __init__(self, ip: Optional[str] = None, auto_connect: bool = True):
        """
        初始化相机
        
        Args:
            ip: 相机IP地址，如果为None则自动搜索
            auto_connect: 是否自动连接相机
        """
        self.ip = ip
        self.status = CameraStatus.DISCONNECTED
        self.info: Optional[CameraInfo] = None
        self.config: Optional[CameraConfig] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.undistort_lut: Optional[np.ndarray] = None
        
        # 设置日志
        self.logger = logging.getLogger(f"EpicEyeCamera_{ip if ip else 'Auto'}")
        self.logger.setLevel(logging.INFO)
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """
        连接相机
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if self.ip is None:
                self.logger.info("自动搜索相机...")
                found_cameras = epiceye.search_camera()
                if found_cameras is None or len(found_cameras) == 0:
                    self.logger.error("未找到相机")
                    self.status = CameraStatus.ERROR
                    return False
                self.ip = found_cameras[0]["ip"]
                self.logger.info(f"找到相机: {self.ip}")
            
            # 获取相机信息
            info = epiceye.get_info(self.ip)
            if info is None:
                self.logger.error(f"无法获取相机信息: {self.ip}")
                self.status = CameraStatus.ERROR
                return False
            
            # 创建CameraInfo对象，只使用必要的字段
            self.info = CameraInfo(
                sn=info.get('sn', ''),
                ip=info.get('ip', ''),
                model=info.get('model', ''),
                alias=info.get('alias', ''),
                width=info.get('width', 0),
                height=info.get('height', 0),
                version=info.get('version'),
                firmwareVersion=info.get('firmwareVersion'),
                hardwareVersion=info.get('hardwareVersion'),
                cameraModel=info.get('cameraModel'),
                ipv4SubnetMask=info.get('ipv4SubnetMask'),
                ipv6=info.get('ipv6'),
                isDhcpEnabled=info.get('isDhcpEnabled')
            )
            self.logger.info(f"相机连接成功: {self.info.model} ({self.info.sn})")
            
            # 获取相机参数
            self._load_camera_parameters()
            
            self.status = CameraStatus.CONNECTED
            return True
            
        except Exception as e:
            self.logger.error(f"连接相机失败: {e}")
            self.status = CameraStatus.ERROR
            return False
    
    def _load_camera_parameters(self):
        """加载相机参数"""
        try:
            # 获取相机矩阵并转换为numpy数组
            cam_matrix_list = epiceye.get_camera_matrix(self.ip)
            if cam_matrix_list:
                self.camera_matrix = np.array(cam_matrix_list)
            
            # 获取畸变参数并转换为numpy数组
            distortion_list = epiceye.get_distortion(self.ip)
            if distortion_list:
                self.distortion = np.array(distortion_list)
            
            # 获取配置
            config_data = epiceye.get_config(self.ip)
            if config_data:
                self.config = CameraConfig(**config_data)
            
            # 获取去畸变查找表
            if self.info:
                self.undistort_lut = epiceye.get_undistort_lut(
                    self.ip, self.info.width, self.info.height)
                
        except Exception as e:
            self.logger.warning(f"加载相机参数失败: {e}")
    
    def disconnect(self):
        """断开相机连接"""
        self.status = CameraStatus.DISCONNECTED
        self.logger.info("相机连接已断开")
    
    def is_connected(self) -> bool:
        """检查相机是否连接"""
        return self.status == CameraStatus.CONNECTED
    
    def get_info(self) -> Optional[CameraInfo]:
        """获取相机信息"""
        return self.info
    
    def get_config(self) -> Optional[CameraConfig]:
        """获取相机配置"""
        return self.config
    
    def set_config(self, config: Dict) -> bool:
        """
        设置相机配置
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 设置是否成功
        """
        try:
            result = epiceye.set_config(self.ip, config)
            if result is not None:
                self.config = CameraConfig(**result)
                self.logger.info("相机配置更新成功")
                return True
            return False
        except Exception as e:
            self.logger.error(f"设置相机配置失败: {e}")
            return False
    
    def capture_frame(self, pointcloud: bool = True) -> Optional[str]:
        """
        触发拍摄一帧
        
        Args:
            pointcloud: 是否包含点云数据
            
        Returns:
            str: 帧ID，失败返回None
        """
        try:
            frame_id = epiceye.trigger_frame(self.ip, pointcloud)
            if frame_id:
                self.logger.debug(f"拍摄成功，帧ID: {frame_id}")
                return frame_id
            else:
                self.logger.error("拍摄失败")
                return None
        except Exception as e:
            self.logger.error(f"拍摄异常: {e}")
            return None
    
    def get_image(self, frame_id: str) -> Optional[np.ndarray]:
        """
        获取图像
        
        Args:
            frame_id: 帧ID
            
        Returns:
            np.ndarray: 图像数据，失败返回None
        """
        try:
            image = epiceye.get_image(self.ip, frame_id)
            if image is not None:
                # 转换为8位图像用于显示
                image_8bit = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
                return image_8bit
            return None
        except Exception as e:
            self.logger.error(f"获取图像失败: {e}")
            return None
    
    def get_point_cloud(self, frame_id: str) -> Optional[np.ndarray]:
        """
        获取点云数据
        
        Args:
            frame_id: 帧ID
            
        Returns:
            np.ndarray: 点云数据，失败返回None
        """
        try:
            pointcloud = epiceye.get_point_cloud(self.ip, frame_id)
            return pointcloud
        except Exception as e:
            self.logger.error(f"获取点云失败: {e}")
            return None
    
    def get_depth(self, frame_id: str) -> Optional[np.ndarray]:
        """
        获取深度图
        
        Args:
            frame_id: 帧ID
            
        Returns:
            np.ndarray: 深度图数据，失败返回None
        """
        try:
            depth = epiceye.get_depth(self.ip, frame_id)
            return depth
        except Exception as e:
            self.logger.error(f"获取深度图失败: {e}")
            return None
    
    def get_image_and_point_cloud(self, frame_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        同时获取图像和点云
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Tuple: (图像, 点云)，失败返回(None, None)
        """
        try:
            image, pointcloud = epiceye.get_image_and_point_cloud(self.ip, frame_id)
            if image is not None:
                image = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
            return image, pointcloud
        except Exception as e:
            self.logger.error(f"获取图像和点云失败: {e}")
            return None, None
    
    def capture_and_get_all(self, pointcloud: bool = True) -> Dict:
        """
        拍摄并获取所有数据
        
        Args:
            pointcloud: 是否包含点云数据
            
        Returns:
            Dict: 包含所有数据的字典
        """
        result = {
            'frame_id': None,
            'image': None,
            'pointcloud': None,
            'depth': None,
            'success': False
        }
        
        try:
            # 触发拍摄
            frame_id = self.capture_frame(pointcloud)
            if frame_id is None:
                return result
            
            result['frame_id'] = frame_id
            
            # 获取图像
            image = self.get_image(frame_id)
            result['image'] = image
            
            # 获取点云
            if pointcloud:
                pointcloud_data = self.get_point_cloud(frame_id)
                result['pointcloud'] = pointcloud_data
            
            # 获取深度图
            depth = self.get_depth(frame_id)
            result['depth'] = depth
            
            result['success'] = True
            
        except Exception as e:
            self.logger.error(f"拍摄并获取数据失败: {e}")
        
        return result
    
    def save_data(self, data: Dict, prefix: str = "epiceye"):
        """
        保存数据到文件
        
        Args:
            data: 数据字典
            prefix: 文件前缀
        """
        timestamp = int(time.time())
        
        try:
            # 保存图像
            if data.get('image') is not None:
                image_path = f"{prefix}_image_{timestamp}.png"
                cv2.imwrite(image_path, data['image'])
                self.logger.info(f"图像已保存: {image_path}")
            
            # 保存点云
            if data.get('pointcloud') is not None:
                pcd_path = f"{prefix}_pointcloud_{timestamp}.ply"
                pcd = o3d.geometry.PointCloud()
                points = data['pointcloud'].reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(pcd_path, pcd)
                self.logger.info(f"点云已保存: {pcd_path}")
            
            # 保存深度图
            if data.get('depth') is not None:
                depth_path = f"{prefix}_depth_{timestamp}.png"
                pseudo_color = cv2.applyColorMap(
                    (data['depth'] / 10.0).astype("uint8"), cv2.COLORMAP_JET)
                cv2.imwrite(depth_path, pseudo_color)
                self.logger.info(f"深度图已保存: {depth_path}")
                
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """获取相机矩阵"""
        return self.camera_matrix
    
    def get_distortion(self) -> Optional[np.ndarray]:
        """获取畸变参数"""
        return self.distortion
    
    def undistort_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        去畸变图像
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 去畸变后的图像
        """
        if self.camera_matrix is None or self.distortion is None:
            self.logger.error("相机参数未加载，无法进行去畸变")
            return None
        
        try:
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.distortion, (w, h), 1, (w, h))
            
            dst = cv2.undistort(image, self.camera_matrix, self.distortion, None, newcameramtx)
            return dst
        except Exception as e:
            self.logger.error(f"图像去畸变失败: {e}")
            return None
    
    def get_point_cloud_statistics(self, pointcloud: np.ndarray) -> Dict:
        """
        获取点云统计信息
        
        Args:
            pointcloud: 点云数据
            
        Returns:
            Dict: 统计信息
        """
        if pointcloud is None:
            return {}
        
        try:
            # 移除无效点（NaN或无穷大）
            valid_points = pointcloud[~np.isnan(pointcloud).any(axis=2)]
            valid_points = valid_points[~np.isinf(valid_points).any(axis=1)]
            
            if len(valid_points) == 0:
                return {'valid_points': 0}
            
            # 计算统计信息
            stats = {
                'total_points': pointcloud.shape[0] * pointcloud.shape[1],
                'valid_points': len(valid_points),
                'valid_ratio': len(valid_points) / (pointcloud.shape[0] * pointcloud.shape[1]),
                'mean_distance': np.mean(np.linalg.norm(valid_points, axis=1)),
                'std_distance': np.std(np.linalg.norm(valid_points, axis=1)),
                'min_distance': np.min(np.linalg.norm(valid_points, axis=1)),
                'max_distance': np.max(np.linalg.norm(valid_points, axis=1)),
                'mean_x': np.mean(valid_points[:, 0]),
                'mean_y': np.mean(valid_points[:, 1]),
                'mean_z': np.mean(valid_points[:, 2]),
                'std_x': np.std(valid_points[:, 0]),
                'std_y': np.std(valid_points[:, 1]),
                'std_z': np.std(valid_points[:, 2])
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算点云统计信息失败: {e}")
            return {}
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


def search_cameras() -> List[CameraInfo]:
    """
    搜索所有可用的相机
    
    Returns:
        List[CameraInfo]: 相机信息列表
    """
    try:
        found_cameras = epiceye.search_camera()
        if found_cameras is None:
            return []
        
        cameras = []
        for camera_data in found_cameras:
            try:
                # 只使用CameraInfo类定义的字段
                camera_info = CameraInfo(
                    sn=camera_data.get('sn', ''),
                    ip=camera_data.get('ip', ''),
                    model=camera_data.get('model', ''),
                    alias=camera_data.get('alias', ''),
                    width=camera_data.get('width', 0),
                    height=camera_data.get('height', 0),
                    version=camera_data.get('version'),
                    firmwareVersion=camera_data.get('firmwareVersion'),
                    hardwareVersion=camera_data.get('hardwareVersion'),
                    cameraModel=camera_data.get('cameraModel'),
                    ipv4SubnetMask=camera_data.get('ipv4SubnetMask'),
                    ipv6=camera_data.get('ipv6'),
                    isDhcpEnabled=camera_data.get('isDhcpEnabled')
                )
                cameras.append(camera_info)
            except Exception as e:
                logging.warning(f"处理相机数据时出错: {e}, 数据: {camera_data}")
                continue
        
        return cameras
    except Exception as e:
        logging.error(f"搜索相机失败: {e}")
        return []


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 搜索相机
    cameras = search_cameras()
    print(f"找到 {len(cameras)} 台相机")
    
    if cameras:
        # 使用第一台相机
        camera = EpicEyeCamera(cameras[0].ip)
        
        if camera.is_connected():
            # 拍摄并获取数据
            data = camera.capture_and_get_all()
            
            if data['success']:
                # 保存数据
                camera.save_data(data)
                
                # 显示点云统计信息
                if data['pointcloud'] is not None:
                    stats = camera.get_point_cloud_statistics(data['pointcloud'])
                    print("点云统计信息:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
        
        camera.disconnect() 