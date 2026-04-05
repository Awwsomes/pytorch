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
        self.setWindowTitle("AI 数据集整理工具")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.info_label = QLabel("提示：鼠标悬停图片 -> 按快捷键移动 (0-9/Q+0-9/W+0-9/E+0-9)。 Ctrl+滚轮：调整列数")
        main_layout.addWidget(self.info_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tab_widget)

        self.setStatusBar(QStatusBar())

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
                img_widget = ImageLabel(full_path)
                img_widget.update_display(self.thumb_size)
                row = i // self.grid_columns
                col = i % self.grid_columns
                grid_layout.addWidget(img_widget, row, col)
            except:
                pass

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

        if key == Qt.Key_Q:
            self.prefix_key = 'Q'
            self.info_label.setText("<b>当前模式：Q</b> (对应类别 10-19) - 请按数字键 0-9")
            return
        if key == Qt.Key_W:
            self.prefix_key = 'W'
            self.info_label.setText("<b>当前模式：W</b> (对应类别 20-29) - 请按数字键 0-9")
            return
        if key == Qt.Key_E:
            self.prefix_key = 'E'
            self.info_label.setText("<b>当前模式：E</b> (对应类别 30-39) - 请按数字键 0-9")
            return

        if Qt.Key_0 <= key <= Qt.Key_9:
            num = key - Qt.Key_0
            target_idx = -1

            if self.prefix_key is None:
                target_idx = num
            elif self.prefix_key == 'Q':
                target_idx = 10 + num
            elif self.prefix_key == 'W':
                target_idx = 20 + num
            elif self.prefix_key == 'E':
                target_idx = 30 + num

            target_idx -= 1

            self.process_move(target_idx)
            self.prefix_key = None
            self.info_label.setText("提示：鼠标悬停图片 -> 按快捷键移动 (0-9/Q+0-9/W+0-9/E+0-9)。 Ctrl+滚轮：调整列数")
            return

        if key == Qt.Key_Escape:
            self.prefix_key = None
            # 恢复初始提示
            self.info_label.setText("提示：鼠标悬停图片 -> 按快捷键移动 (0-9/Q+0-9/W+0-9/E+0-9)。 Ctrl+滚轮：调整列数")
            return

        super().keyPressEvent(event)

    def process_move(self, target_idx):
        if target_idx < 0 or target_idx >= len(self.class_names):
            return

        current_idx = self.tab_widget.currentIndex()
        grid_layout = self.tabs_data[current_idx]['grid_layout']

        active_widget = None
        for i in range(grid_layout.count()):
            widget = grid_layout.itemAt(i).widget()
            if isinstance(widget, ImageLabel) and widget.underMouse():
                active_widget = widget
                break

        if not active_widget:
            return

        src_path = active_widget.image_path
        target_class = self.class_names[target_idx]
        target_dir = os.path.join(self.root_dir, target_class)

        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, filename)
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
            counter += 1

        shutil.move(src_path, dst_path)
        self.refresh_tab(current_idx)


def main():
    app = QApplication(sys.argv)

    root_dir = QFileDialog.getExistingDirectory(None, "请选择数据集根目录")
    if not root_dir:
        return

    subfolders = [f.name for f in os.scandir(root_dir) if f.is_dir()]
    subfolders.sort(key=int)
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