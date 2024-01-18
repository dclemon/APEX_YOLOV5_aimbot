import os
import configparser


def write_ini(inikey, inivaluse, str, filepath):
    config = configparser.ConfigParser()
    config.read(filepath,encoding='utf-8')
    convaluse = config.set(inikey, inivaluse, str)
    config.write(open(filepath, "w"))
    return convaluse


def read_ini(inikey, inivaluse, filepath):
    config = configparser.RawConfigParser()
    config.read(filepath, encoding='utf-8')
    convaluse = config.get(inikey, inivaluse)
    return convaluse

def arg_init(parser):
    dirpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--model-path', type=str, default='weights/allies2.pt', help='模型位址 model address')
    parser.add_argument('--imgsz', type=int, default=640, help='和訓練模型时imgsz一樣')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='置信閥值')
    parser.add_argument('--iou-thres', type=float, default=0.8, help='交並比閥值')
    parser.add_argument('--use-cuda', type=bool, default=True, help='是否使用cuda')
    parser.add_argument('--show-window', type=bool, default=True,
                        help='是否顯示實時檢測窗口(debug用,若是True,不要去點右上角的X)')
    parser.add_argument('--top-most', type=bool, default=True, help='是否保持窗口置頂')
    parser.add_argument('--resize-window', type=float, default=1 / 2, help='缩放窗口大小')
    parser.add_argument('--thickness', type=int, default=5, help='邊框粗細，需大於1/resize-window')
    parser.add_argument('--show-fps', type=bool, default=False, help='是否顯示fps')
    parser.add_argument('--show-label', type=bool, default=False, help='是否顯示標籤')
    parser.add_argument('--use_mss', type=str, default=False, help='是否使用mss截屏；为False時使用win32截屏')
    parser.add_argument('--region', type=tuple, default=(0.156, 0.278),
                        help='檢測範圍；分别为x軸和y軸，(1.0, 1.0)表示全屏檢測，越低檢測範圍越小(以屏幕中心為檢測中心)')
    parser.add_argument('--hold-lock', type=bool, default=True, help='lock模式；True為按住，False為切換')
    parser.add_argument('--lock-sen', type=float, default=2.0, help='lock幅度系數,遊戲中靈敏度(建議不要調整)')

    parser.add_argument('--lock-button', type=list, default=['right', 'left'], help='lock按鍵；只支持鼠標按键')
    parser.add_argument('--head-first', type=bool, default=True, help='是否優先瞄頭')
    parser.add_argument('--lock-tag', type=list, default=[1], help='對應標籤；person(若模型不同請自行修改對應標籤)')
    parser.add_argument('--lock-choice', type=list, default=[1], help='目標選擇；决定鎖定的目標，從自己的標籤中選')
    # PID args
    parser.add_argument("--pid", type=bool, default=True, help="use pid")
    parser.add_argument("--Kp", type=float, default=float(read_ini('config', 'Kp', 'config.ini')), help="Kp")
    parser.add_argument("--Ki", type=float, default=float(read_ini('config', 'Ki', 'config.ini')),
                      help="补偿，可对移动目标做补偿但不能做提前量I太大时，会晃动，可适当调整D或者减小I应该比P小比D大  为0时不起作用")
    parser.add_argument("--Kd", type=float, default=float(read_ini('config', 'Kd', 'config.ini')),
                      help="抑制，可防止抖动D太大时，会弹开鼠标,请调小")
    parser.add_argument('--smooth', type=float, default=100, help='lock平滑系数；越大越平滑')


    args = parser.parse_args(args=[])
    return args
