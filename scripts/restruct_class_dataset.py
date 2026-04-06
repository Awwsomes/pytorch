import sys
import os
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QGridLayout, QLabel, QScrollArea,
                             QFileDialog, QInputDialog, QMessageBox, QSizePolicy, QStatusBar, QRubberBand)
from PyQt5.QtGui import QPixmap, QFont, QPalette
from PyQt5.QtCore import Qt, QSize, QObject, QEvent, QRect, QPoint


class ImageLabel(QLabel):
    """自定义图片标签，支持选中状态"""

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.is_selected = False
        self.setAlignment(Qt.AlignCenter)
        self.update_style()
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

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("border: 3px solid #ff9800; background-color: #fff3e0;")
        else:
            self.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")

    def enterEvent(self, event):
        if not self.is_selected:
            self.setStyleSheet("border: 3px solid #0078d7; background-color: #e0f0ff;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update_style()
        super().leaveEvent(event)


class GridContainer(QWidget):
    """专门的网格容器，处理拖拽框选逻辑"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None  # 会在外部设置
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.selection_start = QPoint()

    def set_main_window(self, mw):
        self.main_window = mw

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_start = event.pos()
            self.rubber_band.setGeometry(QRect(self.selection_start, QSize()))
            self.rubber_band.show()

            # 如果没按Ctrl，清空之前的选择
            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                if self.main_window:
                    self.main_window.clear_selection()

    def mouseMoveEvent(self, event):
        if not self.rubber_band.isHidden():
            self.rubber_band.setGeometry(QRect(self.selection_start, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.rubber_band.isHidden():
            self.rubber_band.hide()
            selection_rect = self.rubber_band.geometry()

            # 遍历所有子控件，看谁在框选范围内
            if self.main_window:
                for container in self.findChildren(QWidget):  # 找图片的容器
                    img_label = container.findChild(ImageLabel)
                    if img_label:
                        # 把容器的坐标映射到 grid 坐标系
                        container_geom = container.geometry()
                        if selection_rect.intersects(container_geom):
                            img_label.is_selected = True
                            img_label.update_style()
                            self.main_window.selected_images.add(img_label)


class GlobalFilter(QObject):
    """全局过滤器：处理 Ctrl+滚轮"""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if QApplication.keyboardModifiers() == Qt.ControlModifier:
                self.main_window.handle_zoom(event.angleDelta().y())
                return True
        return False


class DatasetSorter(QMainWindow):
    def __init__(self, root_dir, class_names):
        super().__init__()
        self.root_dir = root_dir
        self.class_names = class_names

        self.prefix_key = None
        self.grid_columns = 4
        self.thumb_size = 200
        self.tabs_data = {}
        self.selected_images = set()

        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.setWindowTitle("AI 数据集整理工具 - 框选版")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.info_label = QLabel("提示：拖拽框选图片 -> 按快捷键批量移动 (0-9/一行+0-9/...) | Ctrl+滚轮：调整列数")
        main_layout.addWidget(self.info_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        main_layout.addWidget(self.tab_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
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

            # 使用自定义的 GridContainer
            grid_container = GridContainer()
            grid_container.set_main_window(self)  # 绑定主窗口引用

            grid_layout = QGridLayout(grid_container)
            grid_layout.setSpacing(10)

            scroll.setWidget(grid_container)
            layout.addWidget(scroll)

            self.tabs_data[idx] = {
                'grid_widget': grid_container,
                'grid_layout': grid_layout
            }

            self.tab_widget.addTab(tab, f"{idx + 1}. {name}")

        if self.class_names:
            self.refresh_tab(0)

    def on_tab_changed(self, index):
        self.refresh_tab(index)

    def clear_selection(self):
        for img in self.selected_images:
            img.is_selected = False
            img.update_style()
        self.selected_images.clear()

    def refresh_tab(self, index):
        if index not in self.tabs_data:
            return

        data = self.tabs_data[index]
        grid_layout = data['grid_layout']
        current_class = self.class_names[index]

        self.clear_selection()

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

                item_container = QWidget()
                item_layout = QVBoxLayout(item_container)
                item_layout.setContentsMargins(2, 2, 2, 2)
                item_layout.setSpacing(4)
                item_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
                item_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

                item_layout.addWidget(img_widget, 0, Qt.AlignCenter)

                name_label = QLabel(filename[:20] + "..." if len(filename) > 20 else filename)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setStyleSheet("font-size: 11px; color: #333;")
                name_label.setWordWrap(True)
                item_layout.addWidget(name_label, 0, Qt.AlignCenter)

                row = i // self.grid_columns
                col = i % self.grid_columns
                grid_layout.addWidget(item_container, row, col, Qt.AlignCenter)
            except Exception as e:
                pass

    def handle_zoom(self, delta):
        old_cols = self.grid_columns
        if delta > 0:
            self.grid_columns = min(10, self.grid_columns + 1)
        else:
            self.grid_columns = max(1, self.grid_columns - 1)
        if old_cols != self.grid_columns:
            self.refresh_tab(self.tab_widget.currentIndex())
            self.status_label.setText(f"列数: {self.grid_columns}")
            self.status_label.setStyleSheet("color: #666; padding: 0 10px;")

    def keyPressEvent(self, event):
        key = event.key()

        row1_keys = {Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T, Qt.Key_Y, Qt.Key_U, Qt.Key_I, Qt.Key_O, Qt.Key_P}
        row2_keys = {Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_F, Qt.Key_G, Qt.Key_H, Qt.Key_J, Qt.Key_K, Qt.Key_L}
        row3_keys = {Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V, Qt.Key_B, Qt.Key_N, Qt.Key_M}

        if key in row1_keys:
            self.prefix_key = 10
            self.info_label.setText("<b>当前模式：第一行字母</b> (对应类别 10-19) - 请按数字键 0-9")
            return
        if key in row2_keys:
            self.prefix_key = 20
            self.info_label.setText("<b>当前模式：第二行字母</b> (对应类别 20-29) - 请按数字键 0-9")
            return
        if key in row3_keys:
            self.prefix_key = 30
            self.info_label.setText("<b>当前模式：第三行字母</b> (对应类别 30-39) - 请按数字键 0-9")
            return

        if Qt.Key_0 <= key <= Qt.Key_9:
            num = key - Qt.Key_0
            target_idx = -1

            if self.prefix_key is None:
                target_idx = num
            else:
                target_idx = self.prefix_key + num

            self.process_move(target_idx)
            self.prefix_key = None
            self.info_label.setText("提示：拖拽框选图片 -> 按快捷键批量移动 (0-9/一行+0-9/...) | Ctrl+滚轮：调整列数")
            return

        if key == Qt.Key_Escape:
            self.prefix_key = None
            self.clear_selection()
            self.info_label.setText("提示：拖拽框选图片 -> 按快捷键批量移动 (0-9/一行+0-9/...) | Ctrl+滚轮：调整列数")
            return

        super().keyPressEvent(event)

    def process_move(self, target_idx):
        if target_idx <= 0 or target_idx > len(self.class_names):
            self.status_label.setText(f"❌ 无效类别序号 {target_idx}")
            self.status_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        current_idx = self.tab_widget.currentIndex()

        images_to_move = []
        if self.selected_images:
            images_to_move = list(self.selected_images)
        else:
            grid_layout = self.tabs_data[current_idx]['grid_layout']
            active_widget = None
            for i in range(grid_layout.count()):
                container_widget = grid_layout.itemAt(i).widget()
                if container_widget:
                    img_label = container_widget.findChild(ImageLabel)
                    if img_label and img_label.underMouse():
                        active_widget = img_label
                        break
            if active_widget:
                images_to_move = [active_widget]

        if not images_to_move:
            self.status_label.setText("❗ 请先框选图片或将鼠标悬停在图片上")
            self.status_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        target_class = self.class_names[target_idx - 1]
        target_dir = os.path.join(self.root_dir, target_class)
        success_count = 0

        for img_widget in images_to_move:
            src_path = img_widget.image_path
            if not os.path.exists(src_path):
                continue

            filename = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)
            success_count += 1

        self.refresh_tab(current_idx)

        if success_count > 0:
            self.status_label.setText(f"✅ 已批量移动 {success_count} 张至 [{target_idx}] {target_class}")
            self.status_label.setStyleSheet("color: #388e3c; padding: 0 10px;")


def main():
    app = QApplication(sys.argv)

    root_dir = QFileDialog.getExistingDirectory(None, "请选择数据集根目录")
    if not root_dir:
        return

    subfolders = [f.name for f in os.scandir(root_dir) if f.is_dir()]
    subfolders.sort()
    default_classes = ", ".join(subfolders) if subfolders else "class_1, class_2, class_3"

    text, ok = QInputDialog.getText(None, "设置类别", "请输入类别名称（按顺序，逗号分隔）：", text=default_classes)
    if not ok or not text:
        return

    class_names = [x.strip() for x in text.split(',')]

    window = DatasetSorter(root_dir, class_names)

    global_filter = GlobalFilter(window)
    app.installEventFilter(global_filter)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()