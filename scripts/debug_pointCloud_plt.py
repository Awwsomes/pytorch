# import matplotlib.pyplot as plt
# import numpy as np
#
# # 1. 开启交互模式
# plt.ion()
#
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121)  # 2D 图
# ax2 = fig.add_subplot(122, projection='3d')  # 3D 图
#
# # 初始化空的线条
# line2d, = ax1.plot([], [], 'r-')
# line3d, = ax2.plot([], [], [], 'b-')
#
# # 设置坐标轴范围（实时绘图需要预设范围，或者动态调整）
# ax1.set_xlim(0, 100)
# ax1.set_ylim(-1.5, 1.5)
# ax2.set_xlim(-1, 1);
# ax2.set_ylim(-1, 1);
# ax2.set_zlim(0, 10)
#
# t_data, x_data, y_data, z_data = [], [], [], []
#
# # 模拟数据流
# for i in range(100):
#     # 模拟接收到一个新数据点
#     new_t = i
#     new_x = np.sin(i * 0.1)
#     new_y = np.cos(i * 0.1)
#     new_z = i * 0.1
#
#     t_data.append(new_t)
#     x_data.append(new_x)
#     y_data.append(new_y)
#     z_data.append(new_z)
#
#     # 2. 更新数据，而不是重新创建 plot
#     line2d.set_data(t_data, x_data)
#     line3d.set_data(x_data, y_data)
#     line3d.set_3d_properties(z_data)  # 3D 必须单独更新 Z 轴
#
#     # 3. 核心：刷新画布
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#
#     plt.pause(0.01)  # 控制刷新频率，给 CPU 喘息时间
#
# plt.ioff()  # 结束交互模式
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import proj3d
#
# # 开启交互模式
# plt.ion()
#
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122, projection='3d')
#
# # --- 初始化图形对象 ---
# line2d, = ax1.plot([], [], 'r-', alpha=0.6)
# cursor2d, = ax1.plot([], [], 'ko', markersize=6)
# ann2d = ax1.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
#                      bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))
#
# line3d, = ax2.plot([], [], [], 'b-', alpha=0.6)
# cursor3d, = ax2.plot([], [], [], 'ko', markersize=6)
# # 3D 标注通常用 text 对象模拟
# text3d = ax2.text(0, 0, 0, "", bbox=dict(boxstyle="round", fc="cyan", alpha=0.7))
#
# ann2d.set_visible(False)
# text3d.set_visible(False)
#
# t_data, x_data, y_data, z_data = [], [], [], []
#
#
# def on_click(event):
#     if not t_data: return
#
#     # --- 处理 2D 图点击 ---
#     if event.inaxes == ax1:
#         dists = np.abs(np.array(t_data) - event.xdata)
#         idx = np.argmin(dists)
#
#         px, py = t_data[idx], x_data[idx]
#         cursor2d.set_data([px], [py])
#         ann2d.xy = (px, py)
#         ann2d.set_text(f"t:{px}\nx:{py:.2f}")
#         ann2d.set_visible(True)
#
#     # --- 处理 3D 图点击 (核心难点) ---
#     elif event.inaxes == ax2:
#         # 获取当前 3D 投影下的 2D 屏幕坐标
#         xs, ys, _ = proj3d.proj_transform(x_data, y_data, z_data, ax2.get_proj())
#
#         # 计算鼠标点击位置 (event.x, event.y) 与所有投影点之间的像素距离
#         # 注意：这里用的是 display 坐标（像素），而非数据坐标
#         dists = np.hypot(xs - event.x, ys - event.y)
#         idx = np.argmin(dists)
#
#         px, py, pz = x_data[idx], y_data[idx], z_data[idx]
#         cursor3d.set_data([px], [py])
#         cursor3d.set_3d_properties([pz])
#
#         text3d.set_position_3d((px, py, pz))
#         text3d.set_text(f"x:{px:.1f}\ny:{py:.1f}\nz:{pz:.1f}")
#         text3d.set_visible(True)
#
#     fig.canvas.draw_idle()
#
#
# # 绑定事件
# fig.canvas.mpl_connect('button_press_event', on_click)
#
# # --- 模拟数据流 ---
# ax1.set_xlim(0, 100);
# ax1.set_ylim(-1.5, 1.5)
# ax2.set_xlim(-1, 1);
# ax2.set_ylim(-1, 1);
# ax2.set_zlim(0, 10)
#
# for i in range(100):
#     t_data.append(i)
#     x_data.append(np.sin(i * 0.1))
#     y_data.append(np.cos(i * 0.1))
#     z_data.append(i * 0.1)
#
#     line2d.set_data(t_data, x_data)
#     line3d.set_data(x_data, y_data)
#     line3d.set_3d_properties(z_data)
#
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.pause(0.01)
#
# plt.ioff()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d import proj3d

# 开启交互模式
plt.ion()

fig = plt.figure(figsize=(14, 7))
plt.subplots_adjust(left=0.15, bottom=0.1)  # 为按钮留出左侧空间

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# --- 数据容器 ---
t_data, x_data, y_data, z_data = [], [], [], []
current_view = 'x-t'  # 默认视角

# --- 初始化图形对象 ---
line2d, = ax1.plot([], [], 'r-', alpha=0.6, label='2D View')
cursor2d, = ax1.plot([], [], 'ko', markersize=6, zorder=5)
ann2d = ax1.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))

line3d, = ax2.plot([], [], [], 'b-', alpha=0.6)
cursor3d, = ax2.plot([], [], [], 'ko', markersize=6)
text3d = ax2.text(0, 0, 0, "", bbox=dict(boxstyle="round", fc="cyan", alpha=0.7))


# --- 切换视图的逻辑 ---
def update_2d_axes(label):
    global current_view
    current_view = label
    # 根据选择切换坐标轴标签
    if label == 'x-t':
        ax1.set_ylabel('X Value')
        ax1.set_ylim(-1.5, 1.5)
    elif label == 'y-t':
        ax1.set_ylabel('Y Value')
        ax1.set_ylim(-1.5, 1.5)
    elif label == 'z-t':
        ax1.set_ylabel('Z Value')
        ax1.set_ylim(0, 10)

    ax1.set_xlabel('Time (t)')
    ann2d.set_visible(False)
    cursor2d.set_data([], [])
    fig.canvas.draw_idle()


# 添加单选按钮
rax = plt.axes([0.02, 0.4, 0.08, 0.2], facecolor='#f0f0f0')
radio = RadioButtons(rax, ('x-t', 'y-t', 'z-t'))
radio.on_clicked(update_2d_axes)


# --- 点击事件处理 ---
def on_click(event):
    if not t_data: return

    if event.inaxes == ax1:
        # 寻找最近的 t
        dists = np.abs(np.array(t_data) - event.xdata)
        idx = np.argmin(dists)

        px = t_data[idx]
        # 根据当前视图获取对应的 y 轴数据
        py = x_data[idx] if current_view == 'x-t' else (y_data[idx] if current_view == 'y-t' else z_data[idx])

        cursor2d.set_data([px], [py])
        ann2d.xy = (px, py)
        ann2d.set_text(f"t:{px}\nval:{py:.2f}")
        ann2d.set_visible(True)

    elif event.inaxes == ax2:
        # 3D 投影拾取
        xs, ys, _ = proj3d.proj_transform(x_data, y_data, z_data, ax2.get_proj())
        dists = np.hypot(xs - event.x, ys - event.y)
        idx = np.argmin(dists)

        px, py, pz = x_data[idx], y_data[idx], z_data[idx]
        cursor3d.set_data([px], [py])
        cursor3d.set_3d_properties([pz])
        text3d.set_position_3d((px, py, pz))
        text3d.set_text(f"x:{px:.1f}\ny:{py:.1f}\nz:{pz:.1f}")
        text3d.set_visible(True)

    fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_click)

# --- 主循环 ---
ax1.set_xlim(0, 100)
update_2d_axes('x-t')  # 初始化设置

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(0, 10)

for i in range(100):
    t_data.append(i)
    x_data.append(np.sin(i * 0.1))
    y_data.append(np.cos(i * 0.1))
    z_data.append(i * 0.1)

    # 更新 2D 线条（根据当前视图动态选择数据）
    if current_view == 'x-t':
        line2d.set_data(t_data, x_data)
    elif current_view == 'y-t':
        line2d.set_data(t_data, y_data)
    elif current_view == 'z-t':
        line2d.set_data(t_data, z_data)

    # 更新 3D 线条 (x-y-z)
    line3d.set_data(x_data, y_data)
    line3d.set_3d_properties(z_data)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

plt.ioff()
plt.show()