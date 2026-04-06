import sys
import os
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QGridLayout, QLabel, QScrollArea,
                             QFileDialog, QInputDialog, QMessageBox, QSizePolicy, QStatusBar)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QSize, QObject, QEvent


class ImageLabel(QLabel):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pixmap_original = QPixmap(image_path)
        self.update_display()

    def update_display(self, size=200):
        if not self.pixmap_original.isNull():
            scaled_pixmap = self.pixmap_original.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.setFixedSize(scaled_pixmap.size() + QSize(20, 20))

    def enterEvent(self, event):
        self.setStyleSheet("border: 3px solid #0078d7; background-color: #e0f0ff;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        super().leaveEvent(event)


class GlobalFilter(QObject):
    """全局过滤器：管你是谁，只要是滚轮事件我都要先看一眼"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def eventFilter(self, obj, event):
        # 只关心滚轮事件
        if event.type() == QEvent.Wheel:
            # 检查 Ctrl
            if QApplication.keyboardModifiers() == Qt.ControlModifier:
                # 执行缩放
                self.main_window.handle_zoom(event.angleDelta().y())
                # 彻底拦截，谁也别想拿到这个事件
                return True
        # 其他事件正常放行
        return super().eventFilter(obj, event)


class DatasetSorter(QMainWindow):
    def __init__(self, root_dir, class_names):
        super().__init__()
        self.root_dir = root_dir
        self.class_names = class_names

        self.prefix_key = None
        self.grid_columns = 4
        self.thumb_size = 200
        self.tabs_data = {}

        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.setWindowTitle("分类数据集整理工具")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.default_info = "提示：鼠标悬停图片 -> 按快捷键移动 0-9/10-19(键盘字母第一行+0-9)/20-29(键盘字母第二行+0-9)/30-39(键盘字母第三行+0-9)"
        self.info_label = QLabel(self.default_info)
        main_layout.addWidget(self.info_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tab_widget)

        self.setStatusBar(QStatusBar())
        self.status_bar = self.statusBar()

        # 【最小修改】在右侧添加永久提示标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.status_label)

    def load_data(self):
        for name in self.class_names:
            dir_path = os.path.join(self.root_dir, name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        for idx, name in enumerate(self.class_names):
            tab = QWidget()
            layout = QVBoxLayout(tab)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)

            grid_container = QWidget()
            grid_layout = QGridLayout(grid_container)
            grid_layout.setSpacing(10)

            scroll.setWidget(grid_container)
            layout.addWidget(scroll)

            self.tabs_data[idx] = {
                'grid_layout': grid_layout
            }

            self.tab_widget.addTab(tab, f"{idx+1}. {name}")

        if self.class_names:
            self.refresh_tab(0)

    def on_tab_changed(self, index):
        self.refresh_tab(index)

    def refresh_tab(self, index):
        if index not in self.tabs_data:
            return

        grid_layout = self.tabs_data[index]['grid_layout']
        current_class = self.class_names[index]

        # 清空
        for i in reversed(range(grid_layout.count())):
            w = grid_layout.itemAt(i).widget()
            if w: w.setParent(None)

        folder_path = os.path.join(self.root_dir, current_class)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        files.sort()

        for i, filename in enumerate(files):
            full_path = os.path.join(folder_path, filename)
            try:
                # 1. 创建图片控件
                img_widget = ImageLabel(full_path)
                img_widget.update_display(self.thumb_size)

                # 2. 【最小修改】创建一个垂直容器，包含图片和名字
                item_container = QWidget()
                item_layout = QVBoxLayout(item_container)
                item_layout.setContentsMargins(0, 0, 0, 0)  # 去掉边距
                item_layout.setSpacing(2)
                # 让布局内容靠上居中，防止分离
                item_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

                # 添加图片
                item_layout.addWidget(img_widget, 0, Qt.AlignCenter)

                # 添加文件名标签
                name_label = QLabel(filename[:25] + "..." if len(filename) > 25 else filename)  # 文件名过长截断
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setStyleSheet("font-size: 15px; color: #000000;")
                item_layout.addWidget(name_label, 0, Qt.AlignCenter)

                # 3. 将容器加入网格
                row = i // self.grid_columns
                col = i % self.grid_columns
                grid_layout.addWidget(item_container, row, col)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    def handle_zoom(self, delta):
        """缩放逻辑"""
        old_cols = self.grid_columns
        if delta > 0:
            self.grid_columns = min(10, self.grid_columns + 1)
        else:
            self.grid_columns = max(1, self.grid_columns - 1)

        if old_cols != self.grid_columns:
            self.refresh_tab(self.tab_widget.currentIndex())
            self.statusBar().showMessage(f"列数: {self.grid_columns}", 1000)

    def keyPressEvent(self, event):
        key = event.key()

        row1_key = [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T, Qt.Key_Y, Qt.Key_U, Qt.Key_I, Qt.Key_O, Qt.Key_P]
        row2_key = [Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_F, Qt.Key_G, Qt.Key_H, Qt.Key_J, Qt.Key_K, Qt.Key_L]
        row3_key = [Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V, Qt.Key_B, Qt.Key_N, Qt.Key_M]

        if key in row1_key:
            self.prefix_key = 10
            self.info_label.setText("(对应类别 10-19) - 请按数字键 0-9")
            return
        if key in row2_key:
            self.prefix_key = 20
            self.info_label.setText("(对应类别 20-29) - 请按数字键 0-9")
            return
        if key in row3_key:
            self.prefix_key = 30
            self.info_label.setText("(对应类别 30-39) - 请按数字键 0-9")
            return

        if Qt.Key_0 <= key <= Qt.Key_9:
            num = key - Qt.Key_0

            if self.prefix_key is None:
                target_idx = num
            else:
                target_idx = self.prefix_key + num

            # target_idx -= 1

            self.process_move(target_idx)
            self.prefix_key = None
            # 恢复初始提示
            self.info_label.setText(self.default_info)
            return

        if key == Qt.Key_Escape:
            self.prefix_key = None
            # 恢复初始提示
            self.info_label.setText(self.default_info)
            return

        super().keyPressEvent(event)

    def process_move(self, target_idx):
        # print(target_idx)
        if target_idx <= 0 or target_idx > len(self.class_names):
            self.status_label.setText(f"❌ 无效类别序号{target_idx}")
            self.status_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        current_idx = self.tab_widget.currentIndex()
        grid_layout = self.tabs_data[current_idx]['grid_layout']
        # print(current_idx)

        if current_idx + 1 == target_idx:
            self.status_label.setText(f"❗ 已在目标类别{target_idx}，无需移动")
            self.status_label.setStyleSheet("color: #FFA500; padding: 0 10px;")
            return

        active_widget = None
        for i in range(grid_layout.count()):
            container_widget = grid_layout.itemAt(i).widget()
            # 【最小修改】如果是容器，进去找里面的 ImageLabel
            if container_widget:
                # 尝试在容器的布局里找 ImageLabel
                # 或者直接找它的 children
                img_label = container_widget.findChild(ImageLabel)
                if img_label and img_label.underMouse():
                    active_widget = img_label
                    break

        if not active_widget:
            self.status_label.setText("❗ 请先将鼠标悬停在图片上")
            self.status_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        src_path = active_widget.image_path
        if not os.path.exists(src_path):
            self.status_label.setText(f"❌ 文件不存在:{src_path}")
            self.status_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        target_class = self.class_names[target_idx - 1]
        target_dir = os.path.join(self.root_dir, target_class)

        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, filename)
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
            counter += 1

        if dst_path == src_path:
            self.status_label.setText(f"❗ 目标路径与源路径一致，无需移动操作")
            self.status_label.setStyleSheet("color: #FFA500; padding: 0 10px;")
            return
        shutil.move(src_path, dst_path)
        self.refresh_tab(current_idx)

        #  右侧显示成功提示
        self.status_label.setText(f"✅ {filename} 已移动至 [{target_idx}] {target_class}")
        self.status_label.setStyleSheet("color: #388e3c; padding: 0 10px;")

def main():
    app = QApplication(sys.argv)

    root_dir = QFileDialog.getExistingDirectory(None, "请选择数据集根目录")
    if not root_dir:
        return

    subfolders = [f.name for f in os.scandir(root_dir) if f.is_dir()]
    try:
        subfolders.sort(key=int)
    except:
        subfolders.sort()
    default_classes = ", ".join(subfolders) if subfolders else "class_0, class_1"

    text, ok = QInputDialog.getText(None, "设置类别", "请输入类别名称：", text=default_classes)
    if not ok or not text:
        return

    class_names = [x.strip() for x in text.split(',')]

    # class_names = ["1", "2", "3", "4", "5"]

    window = DatasetSorter(root_dir, class_names)

    # 【关键一步】给整个 App 安装过滤器
    global_filter = GlobalFilter(window)
    app.installEventFilter(global_filter)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()