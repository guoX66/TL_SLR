import os
import tkinter as tk
import platform
from tkinter import messagebox
from tkinter import *
import subprocess
from ttkbootstrap import Style
from PIL import ImageTk, Image
from tkinter import ttk
from _utils.configs import ba_Cfg

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


def ini_yaml():
    trt = False  # tensorrt 是否可用
    MYD = False  # 神经棒NCS2 是否可用
    try:  # pytorch 是否可用
        import torch
        pyt = torch.cuda.is_available()
    except:
        pyt = False
    if pyt:  # pytorch可用时 tensorrt才可用
        try:  # tensorrt 是否可用
            import tensorrt
            trt = True
        except:
            pass
    try:  # openvino 是否可用
        from openvino.runtime import Core
        ie = Core()
        OV = True
        OV_devices = Core().available_devices
        if 'MYRIAD' in OV_devices:  # NCS2 是否可用
            MYD = True
    except:
        OV = False

    plat = []
    if pyt or trt:
        plat.append(['pc'])
    if MYD:
        plat.append(['rp'])
    if OV:
        plat.append(['rp', 'pc'])
    plat_list = list(set(plat[0]) & set(plat[1]) & set(plat[2]))  # 求交集
    return pyt, trt, OV, MYD, plat_list


def login_online():  # 在线模式
    root.update()  # 更新窗口，确保进度条显示
    print('1')
    try:
        command = [python_executable, f"{current_dir}/clint.py"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")
    finally:
        messagebox.showinfo("登录成功", f"在线模式已启动：{output}")


def login_offline():  # 离线模式
    progress_bar.start()  # 启动进度条
    root.update()  # 更新窗口，确保进度条显示
    print('2')
    try:
        command = [python_executable, f"{current_dir}/camera.py"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")
    finally:
        messagebox.showinfo("登录成功", f"离线模式已启动：{output}")
        progress_bar.stop()  # 停止进度条


def login_xuexi():  # 学习模式
    progress_bar.start()  # 启动进度条
    root.update()  # 更新窗口，确保进度条显示
    print('2')
    try:
        result = subprocess.run(["python", "zhiwu.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        error_output = result.stderr.decode("utf-8")
        print(output)
        print(error_output)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("错误", f"登录失败：{str(e)}")
    finally:
        messagebox.showinfo("登录成功", f"学习模式已启动：{output}")
        progress_bar.stop()  # 停止进度条


def create_learn_mode_window():
    learn_window = tk.Toplevel(root)
    learn_window.title("学习模式窗口")

    # 设置学习模式窗口的大小和位置与主窗口一致
    window_width = 600
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    learn_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 添加学习模式窗口的内容和功能，保持与主窗口一致
    background_label = tk.Label(learn_window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # 其他内容和功能...


def create_translate_mode_window():
    translate_window = tk.Toplevel(root)
    translate_window.title("翻译模式窗口")

    # 设置翻译模式窗口的大小和位置与主窗口一致
    window_width = 600
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    translate_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # 添加翻译模式窗口的内容和功能，与主窗口一致
    background_label = tk.Label(translate_window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    ImageLabel = tk.Label(translate_window, image=tk_image)
    ImageLabel.place(width=90, height=90, relx=0.425, rely=0.03)

    button_width = 12
    button_height = 2

    # 在线模式按钮，红色
    online_Button = tk.Button(translate_window, text='在线模式', font=("黑体", 15), fg='red', relief=tk.GROOVE,
                              command=login_online, width=button_width, height=button_height)
    online_Button.place(relx=0.5, rely=0.35, anchor='center')

    # 离线模式按钮，绿色
    offline_Button = tk.Button(translate_window, text='离线模式', font=("黑体", 15), fg='green', relief=tk.GROOVE,
                               command=login_offline, width=button_width, height=button_height)
    offline_Button.place(relx=0.5, rely=0.5, anchor='center')
    offline_Button = tk.Button(translate_window, text='综合模式', font=("黑体", 15), fg='red', relief=tk.GROOVE,
                               command=login_online, width=button_width, height=button_height)
    offline_Button.place(relx=0.5, rely=0.65, anchor='center')
    # 添加退出按钮
    quit_button = tk.Button(translate_window, text="返回", font=("黑体", 10), command=translate_window.destroy,
                            width=button_width, height=button_height)
    quit_button.place(relx=0.5, rely=0.8, anchor='center')

    # 添加进度条
    progress_bar = ttk.Progressbar(translate_window, orient="horizontal", length=300, mode="indeterminate")
    progress_bar.place(relx=0.5, rely=0.9, anchor='center')


def quit_app():
    root.destroy()


# 创建主窗口
root = tk.Tk()
root.title("手语识别系统登录")

image2 = Image.open(r"assets\logo1.jpg")
background_image = ImageTk.PhotoImage(image2)
w = background_image.width()
h = background_image.height()
image = Image.open(r"assets\logo2.jpg")  # 替换为你自己的图像文件路径
tk_image = ImageTk.PhotoImage(image)

# 设置窗口大小和位置
window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

style = Style(theme='sandstone')

# 添加背景图片
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
ImageLabel = tk.Label(root, image=tk_image)
ImageLabel.place(width=90, height=90, relx=0.425, rely=0.03)

# 添加用户名和密码的输入框
button_width = 12
button_height = 2

# 在线模式按钮，红色
online_Button = tk.Button(root, text='翻译', font=("黑体", 15), fg='red', relief=tk.GROOVE,
                          command=create_translate_mode_window, width=button_width, height=button_height)
online_Button.place(relx=0.5, rely=0.35, anchor='center')

# 离线模式按钮，绿色
offline_Button = tk.Button(root, text='学习', font=("黑体", 15), fg='green', relief=tk.GROOVE,
                           command=create_learn_mode_window, width=button_width, height=button_height)
offline_Button.place(relx=0.5, rely=0.5, anchor='center')

# 添加退出按钮
quit_button = tk.Button(root, text="退出", font=("黑体", 10), command=quit_app, width=button_width,
                        height=button_height)
quit_button.place(relx=0.5, rely=0.8, anchor='center')

# 添加进度条
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.place(relx=0.5, rely=0.9, anchor='center')

# 启动主循环
root.mainloop()
