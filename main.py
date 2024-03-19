import os
import tkinter as tk
import platform
from tkinter import messagebox
from tkinter import *
import subprocess
import ttkbootstrap as ttk
from PIL import ImageTk, Image
from _utils.configs import read_cfg
import torch
import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))
tr_Cfg, Ir_Cfg, ba_Cfg, Cfg = read_cfg(base_dir)

os_name = str(platform.system())
env = ba_Cfg['env_path']
absolute_env_path = os.path.abspath(env)

# 构建环境中 Python 解释器的绝对路径
if os_name == "Windows":
    python_executable = os.path.join(absolute_env_path, "Scripts", "python.exe")  # Windows
else:
    python_executable = os.path.join(absolute_env_path, "bin", "python")  # Linux/macOS
# 当前文件路径
current_dir = os.path.dirname(os.path.realpath(__file__))


def get_system_info():
    os_name = str(platform.system())
    return os_name



def online_input():
    global pose_text,ip_text,port_text,source_text,view_mode_text
    global shuru
    shuru = tk.Toplevel(root)
    shuru.geometry('400x200')
    shuru.title("请输入参数")
    def clear_handle():  # 清空事件
        pose_text.set("")  # 设置单行文本框内容
        ip_text.set("")
        port_text.set("")
        source_text.set("")
        view_mode_text.set("")
    """pose_net框架"""
    pose_frame = Frame(shuru)
    Label(pose_frame, text="pose_net：").pack(side=LEFT)  # 组件按照自左向右方向排列
    pose_text = StringVar()  # 与单行文本框绑定的变量
    Entry(pose_frame, textvariable= pose_text).pack(side=LEFT)
    pose_frame.pack(pady=5)  # 组件之间的水平间距
    """IP框架"""
    ip_frame = Frame(shuru)
    Label(ip_frame, text="IP：").pack(side=LEFT)
    ip_text = StringVar()
    Entry(ip_frame, textvariable= ip_text).pack(side=LEFT)
    ip_frame.pack(pady=5)
    """port框架"""
    port_frame = Frame(shuru)
    Label(port_frame, text="port：").pack(side=LEFT)
    port_text = StringVar()
    Entry(port_frame, textvariable=port_text).pack(side=LEFT)
    port_frame.pack(pady=5)
    """source框架"""
    source_frame = Frame(shuru)
    Label(source_frame, text="source：").pack(side=LEFT)
    source_text = StringVar()
    Entry(source_frame, textvariable=source_text).pack(side=LEFT)
    source_frame.pack(pady=5)
    """view_mode框架"""
    view_mode_frame = Frame(shuru)
    Label(view_mode_frame, text="view_mode：").pack(side=LEFT)
    view_mode_text = StringVar()
    Entry(view_mode_frame, textvariable=view_mode_text).pack(side=LEFT)
    view_mode_frame.pack(pady=5)
    """按钮框架"""
    button_frame = Frame(shuru)
    Button(button_frame, text='确定', cursor="hand2", width=5, command=press_online).pack(side=LEFT, padx=10)
    Button(button_frame, text='清空', cursor="hand2", width=5, command=clear_handle).pack(side=LEFT, padx=10)
    button_frame.pack()
    shuru.mainloop()
def offline_input():
    global platform_text,pose_text_1, source_text_1, view_mode_text_1
    global lixian
    lixian = tk.Toplevel(root)
    lixian.geometry('400x200')
    lixian.title("请输入参数")

    def clear_handle():  # 清空事件
        platform_text.set("")
        pose_text_1.set("")  # 设置单行文本框内容
        source_text_1.set("")
        view_mode_text_1.set("")

    """pose_net框架"""
    pose_frame = Frame(lixian)
    Label(pose_frame, text="pose_net：").pack(side=LEFT)  # 组件按照自左向右方向排列
    pose_text_1 = StringVar()  # 与单行文本框绑定的变量
    Entry(pose_frame, textvariable=pose_text_1).pack(side=LEFT)
    pose_frame.pack(pady=5)  # 组件之间的水平间距
    """platform框架"""
    platform_frame = Frame(lixian)
    Label(platform_frame, text="platform：").pack(side=LEFT)
    platform_text = StringVar()
    Entry(platform_frame, textvariable=platform_text).pack(side=LEFT)
    platform_frame.pack(pady=5)

    """source框架"""
    source_frame_1 = Frame(lixian)
    Label(source_frame_1, text="source：").pack(side=LEFT)
    source_text_1 = StringVar()
    Entry(source_frame_1, textvariable=source_text_1).pack(side=LEFT)
    source_frame_1.pack(pady=5)
    """view_mode框架"""
    view_mode_frame_1 = Frame(lixian)
    Label(view_mode_frame_1, text="view_mode：").pack(side=LEFT)
    view_mode_text_1 = StringVar()
    Entry(view_mode_frame_1, textvariable=view_mode_text_1).pack(side=LEFT)
    view_mode_frame_1.pack(pady=5)
    """按钮框架"""
    button_frame = Frame(lixian)
    Button(button_frame, text='确定', cursor="hand2", width=5, command=press_offline).pack(side=LEFT, padx=10)
    Button(button_frame, text='清空', cursor="hand2", width=5, command=clear_handle).pack(side=LEFT, padx=10)
    button_frame.pack()
    lixian.mainloop()


def press_offline():
    a = platform_text.get()
    b = pose_text_1.get()
    c = source_text_1.get()
    d = view_mode_text_1.get()
    with open('Cfg.yaml', 'r', encoding='utf-8') as f:
        old_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    if a is not None:
        old_data['base']['platform'] = a
    if b is not None:
        old_data['inference']['pose_net'] = b
    if c is not None:
        old_data['inference']['source'] = d
    if d is not None:
        old_data['base']['view_mode'] = e
    with open('Cfg.yaml', "w", encoding="utf-8") as f:
        yaml.dump(old_data, f)
    lixian.destroy()
    login_offline()
def press_online():
    a = pose_text.get()
    b = ip_text.get()
    c = port_text.get()
    d = source_text.get()
    e = view_mode_text.get()
    with open('Cfg.yaml', 'r', encoding='utf-8') as f:
        old_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    if a is not None:
        old_data['inference']['pose_net'] = a
    if b is not None:
        old_data['inference']['IP'] = b
    if c is not None:
        old_data['inference']['port'] = c
    if d is not None:
        old_data['inference']['source'] = d
    if e is not None:
        old_data['base']['view_mode'] = e
    with open('Cfg.yaml', "w", encoding="utf-8") as f:
        yaml.dump(old_data, f)
    shuru.destroy()
    login_online()
def login_online():  # 在线模式
    try:
        result = subprocess.run(["python", r"E:\github\TL_SLR\clint.py"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")
    finally:
        messagebox.showinfo("登录成功", f"在线模式已启动：{output}")


def login_offline():  # 离线模式
    root.update()  # 更新窗口，确保进度条显示
    print('2')
    try:
        result = subprocess.run(["python", r"E:\github\TL_SLR\camera.py"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)
        messagebox.showinfo("登录成功", f"离线模式已启动：{output}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")



def examine():
    # 创建主窗口
    # exa = tk.Tk()
    # exa.title("选择计算设备")
    transform = torch.cuda.is_available()
    message = "The GPU is available: {}\n".format(transform)
    messagebox.showinfo("Device Availability", message)
    # radio_texts = ["GPU", "CPU", "None"]  # 单选框文本
    # radio_var = IntVar()  # 单选框变量
    # for index, text in enumerate(radio_texts):
    #     Radiobutton(exa, text=text, variable=radio_var, value=index).pack()
    # radio_var.set(0)  # 设置默认选中项
    # Button(exa, text='确定', cursor="hand2", command=lambda: [event_1() if radio_var.get() == 0 else None,
    #                                                           event_2() if radio_var.get() == 1 else None,
    #                                                           event_3() if radio_var.get() == 2 else None]).pack()
    # exa.mainloop()


def login_xuexi():  # 学习模式
    try:
        result = subprocess.run(["python", "train.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)
        messagebox.showinfo("登录成功", f"学习模式已启动：{output}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")




def create_learn_mode_window():
    learn_window = ttk.Toplevel(root)
    learn_window.title("学习模式窗口")

    # 设置学习模式窗口的大小和位置与主窗口一致
    window_make(learn_window)
    # 添加学习模式窗口的内容和功能，保持与主窗口一致
    background_label = tk.Label(learn_window, image=bg)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    button_width = 12
    button_height = 2
    quit_button = ttk.Button(learn_window, text="返回", command=learn_window.destroy,
                            width=button_width, bootstyle="info")
    quit_button.place(relx=0.5, rely=0.8, anchor='center')
    login_xuexi()
    learn_window.mainloop()


def create_translate_mode_window():
    translate_window = tk.Toplevel(root)
    translate_window.title("翻译模式窗口")
    window_make(translate_window)
    # 添加翻译模式窗口的内容和功能，与主窗口一致
    background_label = tk.Label(translate_window, image=bg)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    # ImageLabel = tk.Label(translate_window, image=tk_image)
    # ImageLabel.place(width=90, height=90, relx=0.425, rely=0.03)

    button_width = 19
    button_height = 1

    # 在线模式按钮，红色
    online_Button = ttk.Button(translate_window, text='在线模式',
                              command=online_input, width=button_width,bootstyle="default-outline-toolbutton")
    online_Button.place(relx=0.5, rely=0.32, anchor='center')

    # 离线模式按钮，绿色
    offline_Button = ttk.Button(translate_window, text='离线模式',
                               command=offline_input, width=button_width,bootstyle="success-outline-toolbutton")
    offline_Button.place(relx=0.5, rely=0.425, anchor='center')
    mixline_Button = ttk.Button(translate_window, text='综合模式',
                               command=login_online, width=button_width,bootstyle="dark-outline-toolbutton")
    mixline_Button.place(relx=0.5, rely=0.53, anchor='center')
    # 添加退出按钮
    quit_button = ttk.Button(translate_window, text="返回", command=translate_window.destroy,
                            width=10, bootstyle="info")
    quit_button.place(relx=0.5, rely=0.86, anchor='center')

    # 添加进度条
    progress_bar = ttk.Progressbar(translate_window, orient="horizontal", length=300, mode="indeterminate")
    progress_bar.place(relx=0.5, rely=0.9, anchor='center')
    Checkbutton=ttk.Button(translate_window,text="GPU",command=quit_app,bootstyle="default - round - toggle")#我不知道启动GPU用什么函数
    Checkbutton.place(relx=0.7,rely=0.425,anchor='center')
    translate_window.mainloop()


def quit_app():
    root.destroy()


def window_make(options):
    # 设置窗口大小和位置
    window_width = 960
    window_height = 540
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    options.geometry(f"{window_width}x{window_height}+{x}+{y}")


# 创建主窗口
root = ttk.Window()
root.title("手语识别系统登录")
canvas = Canvas(root, highlightthickness=0)  # 创建Canvas控件，并设置边框厚度为0
canvas.place(width=960, height=540)  # 设置Canvas控件大小及位置
bg = PhotoImage(file=r"assets\logo1.png")  # 【这里记得要改成对应的路径】

# image2 = Image.open(r"assets\photo_1.png")
# background_image = ImageTk.PhotoImage(image2)
w = bg.width()
h = bg.height()
window_make(root)
# style = Style(theme='sandstone')
# # 添加背景图片
# background_label = tk.Label(root, image=background_image)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)
canvas.create_image(480, 240, image=bg)  # 添加背景图片


# # 添加用户名和密码的输入框

button_width = 18
button_height = 2

# # 翻译模式按钮，红色
online_Button = ttk.Button(root, text='翻译',  command=create_translate_mode_window, width=button_width,bootstyle="default-outline-toolbutton")
online_Button.place(relx=0.5, rely=0.30, anchor='center')
# # 学习模式按钮，绿色
offline_Button = ttk.Button(root, text='学习',
                         command=create_learn_mode_window, width=button_width,bootstyle="info-outline-toolbutton" )
offline_Button.place(relx=0.5, rely=0.45, anchor='center')

# # 添加退出按钮
quit_button = ttk.Button(root, text="退出", width=10,
                          bootstyle="info")
quit_button.place(relx=0.5, rely=0.85, anchor='center')
#

examine()
# 启动主循环
root.mainloop()
