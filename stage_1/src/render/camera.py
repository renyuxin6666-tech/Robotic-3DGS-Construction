# -*- coding: utf-8 -*-
"""
相机控制模块

该模块提供Blender中相机的创建和姿态设置功能。
主要功能包括：
- 创建新的相机对象并配置视场角（FOV）
- 设置相机在球坐标系中的位置和朝向
- 实现相机自动朝向场景中心的功能

用途：用于3D渲染和多视角数据集生成中的相机控制
"""

import bpy
import math
import mathutils

def setup_camera(fov_deg):
    """
    创建并设置相机
    
    参数:
        fov_deg (float): 视场角度数（水平方向）
    
    返回:
        bpy.types.Object: 配置好的相机对象
    
    功能:
        1. 创建新的相机数据对象
        2. 配置相机使用视场角（FOV）模式
        3. 设置相机的视场角（转换为弧度）
        4. 创建相机对象并链接到当前场景
        5. 设置该相机为当前场景的活动相机
    """
    # 创建新的相机数据对象
    cam_data = bpy.data.cameras.new("Camera")
    
    # 设置相机镜头单位为视场角（FOV）模式
    cam_data.lens_unit = 'FOV'
    
    # 设置相机的视场角（将度数转换为弧度）
    cam_data.angle = math.radians(fov_deg)
    
    # 创建相机对象，关联相机数据
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    
    # 将相机对象链接到当前集合（场景）
    bpy.context.collection.objects.link(cam_obj)
    
    # 设置该相机为当前场景的活动相机
    bpy.context.scene.camera = cam_obj
    
    return cam_obj

def set_camera_pose(cam, radius, az_deg, el_deg):
    """
    设置相机在球坐标系中的姿态
    
    参数:
        cam (bpy.types.Object): 相机对象
        radius (float): 相机到原点的距离（球坐标系半径）
        az_deg (float): 方位角度数（水平旋转角度，0-360度）
        el_deg (float): 仰角度数（垂直角度，-90到90度）
    
    功能:
        1. 将角度转换为弧度
        2. 计算相机在球坐标系中的位置
        3. 设置相机自动朝向场景中心（原点）
    
    球坐标系转换公式:
        x = radius * cos(elevation) * cos(azimuth)
        y = radius * cos(elevation) * sin(azimuth) 
        z = radius * sin(elevation)
    """
    # 将角度转换为弧度（Blender使用弧度制）
    az = math.radians(az_deg)   # 方位角（水平方向）
    el = math.radians(el_deg)   # 仰角（垂直方向）
    
    # 计算相机在球坐标系中的位置
    # 使用球坐标系到笛卡尔坐标系的转换公式
    cam.location = (
        radius * math.cos(el) * math.cos(az),  # X坐标
        radius * math.cos(el) * math.sin(az),  # Y坐标
        radius * math.sin(el)                   # Z坐标
    )
    
    # 设置相机朝向，使其始终看向场景中心（原点）
    # 1. 计算从相机位置指向原点的向量
    # 2. 使用to_track_quat方法将向量转换为四元数（指定跟踪轴和上方向轴）
    # 3. 将四元数转换为欧拉角（Blender的标准旋转表示）
    cam.rotation_euler = (
        mathutils.Vector((0,0,0)) - cam.location  # 从相机指向原点的向量
    ).to_track_quat('-Z', 'Y').to_euler()  
    # to_track_quat参数说明:
    #   '-Z': 相机的-Z轴指向目标（Blender中相机默认朝向-Z方向）
    #   'Y': 相机的Y轴作为上方向轴