from jd_config import read_ini
from aim_csgo.screen_inf import grab_screen_mss, grab_screen_win32, get_parameters
from aim_csgo.cs_model import load_model
import cv2
import win32gui
import win32con
import torch
import numpy as np
import pygame
import win32api
from ctypes import *
from pynput.mouse import Button
from MyListener import *






class apex:
    def __init__(self):
        super().__init__()
        self.username = read_ini('config', 'username', 'config.ini')
        self.password = read_ini('config', 'password', 'config.ini')




def move_mouse(x, y):

    windll.user32.mouse_event(
        c_uint(0x0001),
        c_uint(x),
        c_uint(y),
        c_uint(0),
        c_uint(0)
    )
def on_click(x, y, button, pressed):
    global lock_mode

    if button in lock_buttons:
        if args.hold_lock:
            if pressed:
                lock_mode = True
            else:
                lock_mode = False
        else:
            if pressed:
                lock_mode = not lock_mode
                print('lock mode', 'on' if lock_mode else 'off')

from aim_csgo.verify_args import verify_args
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import pynput
import argparse
import time
import os
from args_ import arg_init
# 解析命令行参数
args = argparse.ArgumentParser()
args = arg_init(args)  # 初始化参数设置

'------------------------------------------------------------------------------------'

verify_args(args)
cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'
args.model_path = cur_dir + args.model_path
args.lock_tag = [str(i) for i in args.lock_tag]
args.lock_choice = [str(i) for i in args.lock_choice]

device = 'cuda' if args.use_cuda else 'cpu'
half = device != 'cpu'
imgsz = args.imgsz

conf_thres = args.conf_thres
iou_thres = args.iou_thres

top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))

monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

model = load_model(args)
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

lock_mode = False
team_mode = True

# 假设 args.lock_button 是一个包含按钮名称的列表
lock_button_names = args.lock_button

# 创建一个空列表来存储枚举成员
lock_buttons = []

# 遍历按钮名称列表并将其转换为枚举成员
for button_name in lock_button_names:
    lock_button = getattr(Button, button_name)
    lock_buttons.append(lock_button)
mouse = pynput.mouse.Controller()


listener = pynput.mouse.Listener(on_click=on_click)
listener.start()



def main_fun():
    print('enjoy yourself!')
    t0 = time.time()
    cnt = 0

    width = 300
    high = 300
    pygame.init()
    screen = pygame.display.set_mode((width, high), pygame.NOFRAME)  # 创建一个无边框窗口
    fuchsia = (255, 0, 128)  # 透明色(255, 0, 128)
    hwnd = pygame.display.get_wm_info()["window"]
    dwExStyle = win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
    # 计算窗口位置使其居中
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    left = (screen_width - width) // 2
    up = (screen_height - high) // 2
    win32gui.SetWindowPos(hwnd, -1, left, up, width, high, 1)  # 窗口置顶 保持窗口大小  忽略高和宽的参数
    win32gui.SetWindowLong(hwnd, -20, dwExStyle)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(255, 0, 128), 0, 1)

    while True:

        if cnt % 20 == 0:
            top_x, top_y, x, y = get_parameters()
            len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
            top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
            monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}
            cnt = 0

        if args.use_mss:
            img0 = grab_screen_mss(monitor)
            img0 = cv2.resize(img0, (len_x, len_y))
        else:
            img0 = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
            img0 = cv2.resize(img0, (len_x, len_y))

        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.
        if len(img.shape) == 3:
            img = img[None]
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        aims = []
        aims2 = []
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # bbox:(tag, x_center, y_center, x_width, y_width)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    aim = ('%g ' * len(line)).rstrip() % line
                    aim = aim.split(' ')
                    aims.append(aim)

            if len(aims):
                for i, det in enumerate(aims):
                    tag, x_center, y_center, width, height = det
                    x_center, width = len_x * float(x_center), len_x * float(width)
                    y_center, height = len_y * float(y_center), len_y * float(height)
                    x1 = int(x_center - width / 2.)
                    y1 = int(y_center - height / 2.)
                    x2 = int(x_center + width / 2.)
                    y2 = int(y_center + height / 2.)
                    new_det = (tag, x1, y1, x2, y2)
                    # 转换坐标格式
                    aims2.append(new_det)
            if args.show_window:
                draw_arera(aims2, screen, hwnd, fuchsia, len_x, len_y, args)
                args.lock_mode = lock_mode
                if lock_mode:
                    Mouse_redirection(aims2, args, interval)  # 处理预测结果，可能是自瞄的核心部分
                    Move_Mouse(args)  # 移动鼠标N
                # move_mouse(int(top_x + x_center) - 960, int(top_y + y_center) - 540)

        cnt += 1

    return



# 主程序入口
if __name__ == "__main__":
    main_fun()





