# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# import numpy as np
# import sys
# from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
#
# class RealTimePlotter:
#     def __init__(self):
#         self.app = QApplication(sys.argv)
#
#         # 1. 创建一个标准的 Qt 窗口作为主容器
#         self.main_window = QWidget()
#         self.main_window.resize(1000, 800)
#         self.layout = QVBoxLayout()  # 创建垂直布局
#         self.main_window.setLayout(self.layout)
#
#         # 2. 创建 2D 绘图控件 (PlotWidget 而不是 GraphicsLayoutWidget)
#         self.pw2d = pg.PlotWidget(title="Real-time 2D (X-T)")
#         self.curve2d = self.pw2d.plot(pen='r')
#         self.layout.addWidget(self.pw2d)  # 将 2D 控件加入布局
#
#         # 3. 创建 3D 绘图控件
#         self.view3d = gl.GLViewWidget()
#         self.layout.addWidget(self.view3d)  # 将 3D 控件加入布局
#
#         # 3D 辅助设施
#         self.view3d.addItem(gl.GLGridItem())
#         self.curve3d = gl.GLLinePlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 1, 1), width=2)
#         self.view3d.addItem(self.curve3d)
#
#         # 数据容器和定时器保持不变...
#         self.t_data, self.x_data, self.pos_3d = [], [], []
#         self.timer = pg.QtCore.QTimer()
#         self.timer.timeout.connect(self.update)
#         self.timer.start(20)
#
#         self.main_window.show()
#
#     def update(self):
#         # 模拟产生新数据
#         idx = len(self.t_data)
#         new_t = idx * 0.05
#         new_x = np.sin(new_t)
#         new_y = np.cos(new_t)
#         new_z = new_t * 0.1
#
#         self.t_data.append(new_t)
#         self.x_data.append(new_x)
#         self.pos_3d.append([new_x, new_y, new_z])
#
#         # 更新 2D 图 (只需传递 numpy 数组)
#         self.curve2d.setData(self.t_data, self.x_data)
#
#         # 更新 3D 图
#         self.curve3d.setData(pos=np.array(self.pos_3d))
#
#     def run(self):
#         sys.exit(self.app.exec())
#
# if __name__ == '__main__':
#     plotter = RealTimePlotter()
#     plotter.run()

# from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# import numpy as np
# import sys
#
#
# class RealTimePlotter:
#     def __init__(self):
#         self.app = QApplication(sys.argv)
#         self.main_window = QWidget()
#         self.main_window.setWindowTitle("Advanced 3D/2D Plotter (Coordinate Tracking)")
#         self.layout = QVBoxLayout()
#         self.main_window.setLayout(self.layout)
#
#         # 增加一个状态栏显示坐标
#         self.label = QLabel("Mouse Coordinate: ( - , - )")
#         self.layout.addWidget(self.label)
#
#         # --- 2D 部分 ---
#         self.pw2d = pg.PlotWidget()
#         self.curve2d = self.pw2d.plot(pen='r')
#         self.layout.addWidget(self.pw2d)
#
#         # 添加 2D 十字光标
#         self.vLine = pg.InfiniteLine(angle=90, movable=False)
#         self.hLine = pg.InfiniteLine(angle=0, movable=False)
#         self.pw2d.addItem(self.vLine, ignoreBounds=True)
#         self.pw2d.addItem(self.hLine, ignoreBounds=True)
#
#         # 绑定 2D 鼠标移动事件
#         self.proxy = pg.SignalProxy(self.pw2d.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
#
#         # --- 3D 部分 ---
#         self.view3d = gl.GLViewWidget()
#         self.layout.addWidget(self.view3d)
#         self.view3d.addItem(gl.GLGridItem())
#
#         # 3D 轨迹
#         self.curve3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(0, 1, 1, 1), width=2)
#         self.view3d.addItem(self.curve3d)
#
#         # 3D 高亮选点（用于显示选中的位置）
#         self.selection_point = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(1, 0, 0, 1), size=10)
#         self.view3d.addItem(self.selection_point)
#
#         # 数据和定时器
#         self.pos_3d = []
#         self.timer = pg.QtCore.QTimer()
#         self.timer.timeout.connect(self.update_data)
#         self.timer.start(50)
#
#         self.main_window.show()
#
#     def mouseMoved(self, evt):
#         pos = evt[0]  # 获取鼠标位置
#         if self.pw2d.sceneBoundingRect().contains(pos):
#             mousePoint = self.pw2d.plotItem.vb.mapSceneToView(pos)
#             self.label.setText(f"Point: ({mousePoint.x():.2f}, {mousePoint.y():.2f})")
#             # 更新十字准星
#             self.vLine.setPos(mousePoint.x())
#             self.hLine.setPos(mousePoint.y())
#
#     def update_data(self):
#         # 模拟生成轨迹数据
#         idx = len(self.pos_3d)
#         t = idx * 0.1
#         x, y, z = np.sin(t), np.cos(t), t * 0.1
#         self.pos_3d.append([x, y, z])
#
#         points = np.array(self.pos_3d)
#         self.curve3d.setData(pos=points)
#         self.curve2d.setData(points[:, 0])  # 仅画 X 轴
#
#     def run(self):
#         sys.exit(self.app.exec())
#
#
# if __name__ == '__main__':
#     plotter = RealTimePlotter()
#     plotter.run()

# from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
# import numpy as np
# import sys
#
#
# class RealTimePlotter:
#     def __init__(self):
#         self.app = QApplication(sys.argv)
#         self.main_window = QWidget()
#         self.main_window.setWindowTitle("Advanced Plotter: XT/XY Switcher")
#         self.layout = QVBoxLayout()
#         self.main_window.setLayout(self.layout)
#
#         # --- 1. 顶部控制栏 (按钮) ---
#         self.btn_layout = QHBoxLayout()
#         self.btn_xt = QPushButton("切换到 X-T 视图")
#         self.btn_xy = QPushButton("切换到 X-Y 视图")
#
#         # 绑定点击事件
#         self.btn_xt.clicked.connect(lambda: self.switch_mode("XT"))
#         self.btn_xy.clicked.connect(lambda: self.switch_mode("XY"))
#
#         self.btn_layout.addWidget(self.btn_xt)
#         self.btn_layout.addWidget(self.btn_xy)
#         self.layout.addLayout(self.btn_layout)
#
#         # 状态显示
#         self.label = QLabel("当前模式: X-T (时间序列)")
#         self.layout.addWidget(self.label)
#
#         # --- 2. 2D 部分 ---
#         self.pw2d = pg.PlotWidget()
#         self.curve2d = self.pw2d.plot(pen=pg.mkPen('r', width=2))
#         self.pw2d.showGrid(x=True, y=True)
#         self.layout.addWidget(self.pw2d)
#
#         self.mode = "XT"  # 默认模式
#         self.set_axes_labels()
#
#         # --- 3. 3D 部分 ---
#         self.view3d = gl.GLViewWidget()
#         self.layout.addWidget(self.view3d)
#         self.view3d.addItem(gl.GLGridItem())
#         self.curve3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(0, 1, 1, 1), width=2)
#         self.view3d.addItem(self.curve3d)
#
#         # 数据容器
#         self.t_data, self.x_data, self.y_data, self.pos_3d = [], [], [], []
#
#         # 定时器
#         self.timer = pg.QtCore.QTimer()
#         self.timer.timeout.connect(self.update_data)
#         self.timer.start(50)
#
#         self.main_window.show()
#
#     def switch_mode(self, mode):
#         self.mode = mode
#         self.set_axes_labels()
#         if mode == "XT":
#             self.label.setText("当前模式: X-T (横轴: 时间, 纵轴: X坐标)")
#         else:
#             self.label.setText("当前模式: X-Y (横轴: X坐标, 纵轴: Y坐标)")
#
#     def set_axes_labels(self):
#         if self.mode == "XT":
#             self.pw2d.setLabel('bottom', 'Time', units='s')
#             self.pw2d.setLabel('left', 'X Coordinate', units='m')
#         else:
#             self.pw2d.setLabel('bottom', 'X Coordinate', units='m')
#             self.pw2d.setLabel('left', 'Y Coordinate', units='m')
#
#     def update_data(self):
#         # 模拟生成机器人圆周运动轨迹
#         idx = len(self.t_data)
#         t = idx * 0.1
#         x, y, z = np.sin(t), np.cos(t), t * 0.1
#
#         self.t_data.append(t)
#         self.x_data.append(x)
#         self.y_data.append(y)
#         self.pos_3d.append([x, y, z])
#
#         # --- 2D 数据更新逻辑切换 ---
#         if self.mode == "XT":
#             # X轴为时间，Y轴为X值
#             self.curve2d.setData(self.t_data, self.x_data)
#         else:
#             # X轴为X值，Y轴为Y值 (这就是轨迹平面图)
#             self.curve2d.setData(self.x_data, self.y_data)
#
#         # 3D 始终绘制完整轨迹
#         self.curve3d.setData(pos=np.array(self.pos_3d))
#
#     def run(self):
#         sys.exit(self.app.exec())
#
#
# if __name__ == '__main__':
#     plotter = RealTimePlotter()
#     plotter.run()

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtWidgets import QStackedWidget # 记得导入
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys


class RealTimePlotter:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.main_window = QWidget()
        self.main_window.setWindowTitle("RoboViz: Multi-View Switcher")
        self.main_window.resize(1000, 800)
        self.layout = QVBoxLayout()
        self.main_window.setLayout(self.layout)

        # --- 1. 顶部控制栏 ---
        self.btn_layout = QHBoxLayout()

        # 定义三个模式按钮
        self.btn_xt = QPushButton("X-T 视图 (2D)")
        self.btn_xy = QPushButton("X-Y 轨迹 (2D)")
        self.btn_xyz = QPushButton("X-Y-Z 空间 (3D)")

        self.btn_xt.clicked.connect(lambda: self.switch_mode("XT"))
        self.btn_xy.clicked.connect(lambda: self.switch_mode("XY"))
        self.btn_xyz.clicked.connect(lambda: self.switch_mode("XYZ"))

        for btn in [self.btn_xt, self.btn_xy, self.btn_xyz]:
            self.btn_layout.addWidget(btn)
        self.layout.addLayout(self.btn_layout)

        self.label = QLabel("当前模式: X-T 视图")
        self.layout.addWidget(self.label)

        # --- 2. 绘图区域 (2D 和 3D 放在同一个垂直布局中) ---

        # 2D 控件
        self.pw2d = pg.PlotWidget()
        self.curve2d = self.pw2d.plot(pen=pg.mkPen('r', width=2))
        self.pw2d.showGrid(x=True, y=True)
        self.layout.addWidget(self.pw2d)

        # 3D 控件
        self.view3d = gl.GLViewWidget()
        self.view3d.addItem(gl.GLGridItem())
        self.curve3d = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=(0, 1, 1, 1), width=2, antialias=True)
        self.view3d.addItem(self.curve3d)
        self.layout.addWidget(self.view3d)

        # 创建坐标轴对象
        self.axes = gl.GLAxisItem()

        # 设置坐标轴的大小（x, y, z 方向的长度）
        self.axes.setSize(x=10, y=10, z=10)

        # 2. 调整网格位置，让它处于 Z=0 的平面，方便观察高度
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)  # 放大网格
        self.view3d.addItem(self.grid)

        # 将坐标轴添加到 3D 视图中
        self.view3d.addItem(self.axes)

        # 注意：这需要较新版本的 pyqtgraph
        t_x = gl.GLTextItem(pos=(10, 0, 0), text='X', color=(255, 0, 0, 255))
        t_y = gl.GLTextItem(pos=(0, 10, 0), text='Y', color=(0, 255, 0, 255))
        t_z = gl.GLTextItem(pos=(0, 0, 10), text='Z', color=(0, 0, 255, 255))

        self.view3d.addItem(t_x)
        self.view3d.addItem(t_y)
        self.view3d.addItem(t_z)

        # 初始状态：显示 2D，隐藏 3D
        self.view3d.hide()
        self.mode = "XT"

        # --- 3. 数据处理 ---
        self.t_data, self.x_data, self.y_data, self.z_data, self.pos_3d = [], [], [], [], []

        # # 设置拉伸系数：这两个控件在布局中如果可见，都会尝试占满全部空间
        # self.layout.setStretch(1, 1)  # 1号位置（2D）系数为 1
        # self.layout.setStretch(2, 1)  # 2号位置（3D）系数为 1

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(50)

        self.main_window.show()

    def switch_mode(self, mode):
        self.mode = mode

        # 处理控件显隐
        if mode == "XYZ":
            self.pw2d.hide()
            self.view3d.show()
            self.label.setText("当前模式: 3D 空间轨迹 (鼠标可拖拽旋转)")
        else:
            self.view3d.hide()
            self.pw2d.show()
            if mode == "XT":
                self.pw2d.setLabel('bottom', 'Time', units='s')
                self.pw2d.setLabel('left', 'X', units='m')
                self.label.setText("当前模式: X-T 时间序列")
            else:
                self.pw2d.setLabel('bottom', 'X', units='m')
                self.pw2d.setLabel('left', 'Y', units='m')
                self.pw2d.setAspectLocked(True)  # X-Y 视图强制比例一致，轨迹不失真
                self.label.setText("当前模式: X-Y 平面运动轨迹")

    def update_data(self):
        # 模拟生成 3D 螺旋上升轨迹
        idx = len(self.t_data)
        t = idx * 0.1
        x, y, z = np.sin(t), np.cos(t), t * 0.05

        self.t_data.append(t)
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
        self.pos_3d.append([x, y, z])

        # 更新数据（无论是否显示，后台都在更新，保证切换时数据连续）
        if self.mode == "XT":
            self.curve2d.setData(self.t_data, self.x_data)
        elif self.mode == "XY":
            self.curve2d.setData(self.x_data, self.y_data)

        # 3D 始终更新数据
        self.curve3d.setData(pos=np.array(self.pos_3d))

    def run(self):
        sys.exit(self.app.exec())


if __name__ == '__main__':
    plotter = RealTimePlotter()
    plotter.run()