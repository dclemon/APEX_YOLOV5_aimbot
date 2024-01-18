from pynput import mouse, keyboard
import numpy as np
import pyautogui
import time
import win32print
import cv2
import pygame
import win32api, win32con, win32gui
import ctypes
from ctypes import *
import numpy as np
import os

# 控制目标检测状态的全局变量
Start_detection = False

# 旧的目标检测状态，可能在某些地方被使用
old_status = False

# 控制循环监听状态的全局变量
Listen = True

# 用于记录目标的宽度
width = 0

# 帧处理间隔
interval = 0.01

# 屏幕的尺寸（以像素为单位）
screen_size = np.array([win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)])

# 屏幕中心的坐标（以像素为单位）
screen_center = np.array(screen_size, dtype=int) // 2

# 鼠标移动的目标位置
destination = screen_center

# 上一次的目标位置
last = destination

# 后向力，可能是用于控制特殊情况的参数
backforce = 0

# 获取屏幕DPI缩放比例
hDC = win32gui.GetDC(0)
scale = win32print.GetDeviceCaps(hDC, win32con.LOGPIXELSX) / 96


#------------绘制自瞄区域-------------

gdi32 = ctypes.windll.LoadLibrary(r"C:\Windows\System32\gdi32.dll")
user32 = ctypes.windll.LoadLibrary(r"C:\Windows\System32\user32.dll")

def draw_arera(boxes,screen,hwnd,fuchsia,len_x,len_y,args):
    try:
        pygame.time.Clock().tick(60)
        screen.fill(fuchsia)
        pygame.display.update()
        hdc = windll.user32.GetDC(hwnd)
        #--------------hdc过一段时间后变成0，导致无法绘制方框，需要及时释放-----------

        pen = gdi32.CreatePen(0, 3, 255)  # 创建红色画笔  0代表实线  2是粗细  3是颜色
        pen2 = gdi32.SelectObject(hdc, pen)  # 原画笔跟红画笔交换
        Printing = gdi32.GetStockObject(5)  # 空画刷 目的是让图形透明
        Printing2 = gdi32.SelectObject(hdc, Printing)  # 交换画刷
        # 画出自瞄区间
        gdi32.Rectangle(hdc, int(0), int(0), int(300), int(300))
        # 如果没有检测到任何目标
        if len(boxes) == 0:
            gdi32.SelectObject(hdc, pen2)
            gdi32.DeleteObject(pen)  # 销毁红色画笔
            gdi32.SelectObject(hdc, Printing2)  # 交换画刷  不需要销毁 因为画刷是向系统借的
            user32.ReleaseDC(hwnd, hdc)  # 释放hdc
            return
        else:
            box_list = boxes[0][1:]
            # 创建一个包含嵌套列表的NumPy数组
            boxes = np.array([list(map(float, box_list))])
            for i, det in enumerate(boxes):
                x_min, y_min, x_max, y_max = det
                color = (0, 0, 255)  # RGB颜色（蓝色）
                # 绘制连线和敌人方框
                drawrect(hdc, x_min, y_min, x_max, y_max, boxes,len_x,len_y,args)

        gdi32.SelectObject(hdc, pen2)
        gdi32.DeleteObject(pen)  # 销毁红色画笔
        gdi32.SelectObject(hdc, Printing2)  # 交换画刷  不需要销毁 因为画刷是向系统借的
        user32.ReleaseDC(hwnd, hdc)  # 释放hdc
    except IOError as e:
        print(e)

    return

def drawrect(hdc,x1,y1,x2,y2,boxes,len_x,len_y,args):
    """
    :param hdc:windll.user32.GetDC(hwnd)的返回值
    :param x1:最左边的点
    :param y1:最上面的点
    :param x2:最右边的点
    :param y2:最下面的点
    :return:
    """
    #画敌人方框

    gdi32.Rectangle(hdc, int(x1), int(y1), int(x2), int(y2))

    if x1 != 0 and y1 != 0:
        destination = get_nearest_target(boxes,args)
        gdi32.MoveToEx(hdc, 150,150, None) #设置起始点坐标
        gdi32.LineTo(hdc, int(destination[0]), int(destination[1]))# 绘制直线到终点坐标

def get_nearest_target(boxes,args):
    global destination, width, interval, screen_size, screen_center, last
    pos = np.array(win32api.GetCursorPos(), dtype=int)  # 获取当前鼠标位置
    # 计算检测到的目标的中心点
    boxes_center = (
            (boxes[:, :2] + boxes[:, 2:]) / 2
    )
    #boxes_center[:, 1] = (boxes[:, 1] * 0.6 + boxes[:, 3] * 0.4)
    # 找到最近的目标中心点
    dis = np.linalg.norm(boxes_center - pos, axis=-1)
    min_index = np.argmin(dis)
    width = boxes[min_index, 2] - boxes[min_index, 0]
    last = destination
    destination = boxes_center[np.argmin(dis)].astype(int)  # 更新目标位置为最近的目标中心点
    return destination

def Move_Mouse(args):
    global screen_size, screen_center
    global destination, width, interval, pre_error, intergral
    global last, mouse_vector_history
    # 如果启动了目标检测
    if args.lock_mode:
        pos = np.array(win32api.GetCursorPos(), dtype=int)  # 获取当前鼠标位置
        mouse_vector = (destination - pos) / scale  # 计算鼠标目标向量
        norm = np.linalg.norm(mouse_vector)  # 计算向量长度（范数）
        #--------目标补偿计算--------

        #----------------PID平滑------------------
        # 如果启用了PID控制
        if args.pid:
            if (destination[0] == -1 and destination[1] == -1):
                if last[0] == -1:
                    pre_error = intergral = np.array([0., 0.])
                    mouse_vector = np.array([0, 0])
                    return
                else:
                    mouse_vector = np.array([0, 0])
            move = PID(args, mouse_vector)  # 使用PID算法计算鼠标移动距离
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(move[0]), int(move[1]))
            return

        # 如果目标距离很近，或者目标在屏幕中心，则不移动鼠标
        if norm <= 2 or (destination[0] == screen_center[0] and destination[1] == screen_center[1]):
            return

        # 如果目标距离较近，使用一半的移动向量移动鼠标
        if norm <= width * 2 / 3:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_vector[0] / 2), int(mouse_vector[1] / 2))
            return


        des = mouse_vector / args.smooth  # 计算平滑后的移动向量
        for i in range(int(args.smooth)):
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(des[0]), int(des[1]))
            time.sleep(0.01 / args.smooth)  # 平滑移动，每次移动间隔一小段时间

        delay_time = 1 / float(args.game_fps)  # 计算延迟时间
        time.sleep(delay_time)  # 等待一段时间，可能是为了控制移动速度
    else:
        pre_error = intergral = np.array([0., 0.])  # 如果未启动目标检测，则重置一些参数


# redirect the mouse closer to the nearest box center
def Mouse_redirection(boxes, args, tpf):
    global destination, width, interval, screen_size, screen_center, last
    # 如果没有检测到任何目标
    if len(boxes) == 0:
        last = destination
        destination = np.array([-1, -1])
        return
    else:
        box_list = boxes[0][1:]
        # 创建一个包含嵌套列表的NumPy数组
        boxes = np.array([list(map(float, box_list))])
    if boxes.shape[0] == 0:
        last = destination
        destination = np.array([-1, -1])  # 将目标位置设为无效值
        return

    interval = tpf  # 更新帧处理间隔
    pos = np.array(win32api.GetCursorPos(), dtype=int)  # 获取当前鼠标位置

    # 计算检测到的目标的中心点
    boxes_center = (
        (boxes[:, :2] + boxes[:, 2:]) / 2
    )
    boxes_center[:, 1] = (boxes[:, 1] * 0.6 + boxes[:, 3] * 0.4)

    # 将目标从图像坐标映射到屏幕坐标
    screen_center = screen_size / 2
    start_point = screen_center - screen_size[1] * args.region[1] / 2
    start_point = list(map(int, start_point))
    boxes_center[:, 0] = boxes_center[:, 0] + start_point[0]
    boxes_center[:, 1] = boxes_center[:, 1] + start_point[1]
    # 找到最近的目标中心点
    dis = np.linalg.norm(boxes_center - pos, axis=-1)
    min_index = np.argmin(dis)
    width = boxes[min_index, 2] - boxes[min_index, 0]
    last = destination
    destination = boxes_center[np.argmin(dis)].astype(int)  # 更新目标位置为最近的目标中心点
# 初始化PID控制器的前一误差和积分部分
pre_error = intergral = np.array([0., 0.])
def PID(args, error):
    global pre_error, intergral, backforce

    # 更新积分部分
    intergral += error

    # 计算导数部分
    derivative = error - pre_error
    pre_error = error  # 更新前一误差

    # 使用PID公式计算输出
    output = args.Kp * error + args.Ki * intergral + args.Kd * derivative
    output[1] += backforce  # 加入后向力，可能是一种特殊的控制手段

    pre_error = error  # 更新前一误差
    return output.astype(int)  # 返回整数形式的输出