import sys
import os
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QGridLayout, QLabel, QScrollArea,
                             QFileDialog, QInputDialog, QMessageBox, QSizePolicy, QStatusBar, QRubberBand)
from PyQt5.QtGui import QPixmap, QFont, QPalette
from PyQt5.QtCore import Qt, QSize, QObject, QEvent, QRect, QPoint

class ImageLabel(QLabel):
    """自定义图片标签，"""

    def __init__(self, image_path, txt_dir=None):
        super().__init__()
        self.image_path = image_path
        self.txt_dir = txt_dir
        self.is_selected = False
        self.setAlignment(Qt.AlignCenter)
        self.update_style()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pixmap_original = QPixmap(image_path)
        self.update_display()

    def update_display(self, size=200):
        # 清空现有布局（如果是第二次调用，防止重叠）
        if self.layout():
            QWidget().setLayout(self.layout())

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        if not self.pixmap_original.isNull():
            scaled_pixmap = self.pixmap_original.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            img_label = QLabel()
            img_label.setPixmap(scaled_pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(img_label, 0, Qt.AlignCenter)

            # 【新增】读取并显示 Top2 置信度
            if self.txt_dir:
                conf_text = self._load_confidence()
                if conf_text:
                    conf_label = QLabel(conf_text)
                    conf_label.setAlignment(Qt.AlignCenter)
                    conf_label.setStyleSheet(
                        "font-size: 10px; color: #555; background-color: rgba(255,255,255,180); padding: 1px;")
                    conf_label.setWordWrap(True)
                    main_layout.addWidget(conf_label, 0, Qt.AlignCenter)

            self.setFixedSize(QSize(size, size) + QSize(20, 60))  # 增加高度放文字

    # 【新增】解析 TXT 的辅助函数
    def _load_confidence(self):
        if not self.txt_dir: return None
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        txt_path = os.path.join(self.txt_dir, base_name + ".txt")

        if not os.path.exists(txt_path):
            return None

        try:
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            top2 = []
            # 取前2行解析
            for i in range(min(2, len(lines))):
                parts = lines[i].split()
                if len(parts) >= 2:
                    conf = float(parts[0])
                    cls_name = int(parts[1])
                    top2.append(f"{cls_name}: {conf:.2f}")

            return "\n".join(top2)
        except Exception:
            return None

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

    def mouseReleaseEvent(self, event):
        # print("imagelabel mouse release")
        if event.button() == Qt.LeftButton:
            # 这里的 main_window 需要在 DatasetSorter 创建它时传入
            # 或者通过 self.window() 获取顶层窗口
            main_win = self.window()

            # 检查是否按住 Ctrl
            modifiers = QApplication.keyboardModifiers()
            if not (modifiers & Qt.ControlModifier):
                # 没按 Ctrl，清空之前的
                main_win.clear_selection()
                self.is_selected = True
            else:
                # 按了 Ctrl，反转选中状态
                self.is_selected = not self.is_selected

            self.update_style()

            # 同步主窗口的集合
            if self.is_selected:
                main_win.selected_images.add(self)
            else:
                main_win.selected_images.discard(self)

        super().mousePressEvent(event)

class GridContainer(QWidget):
    """专门的网格容器，处理拖拽框选逻辑"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None  # 会在外部设置
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.selection_start = QPoint()
        self.mouse_left_press = False

    def set_main_window(self, mw):
        self.main_window = mw

    def mousePressEvent(self, event):
        self.mouse_left_press = True
        # print("GridContainer mousePressEvent")
        # if event.button() == Qt.LeftButton:
        #     self.selection_start = event.pos()
        #     self.rubber_band.setGeometry(QRect(self.selection_start, QSize()))
        #     self.rubber_band.show()
        #
        # 如果没按Ctrl，清空之前的选择
        if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
            if self.main_window:
                self.main_window.clear_selection()

    def mouseMoveEvent(self, event):
        # print("GridContainer mouseMoveEvent")
        # print(self.rubber_band.isHidden())
        # print(event.button() == Qt.LeftButton)
        if self.mouse_left_press and self.rubber_band.isHidden():
            self.selection_start = event.pos()
            self.rubber_band.setGeometry(QRect(self.selection_start, QSize()))
            self.rubber_band.show()
        if not self.rubber_band.isHidden():
            self.rubber_band.setGeometry(QRect(self.selection_start, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        # print("GridContainer mouseReleaseEvent")
        self.mouse_left_press = False
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
                            img_label.is_selected = not img_label.is_selected
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
    def __init__(self, root_dir:str, class_names:list, txt_dir:str):
        super().__init__()
        self.root_dir = root_dir
        class_names.append("垃圾桶")
        self.class_names = class_names
        self.txt_dir = txt_dir  # 【新增】保存路径

        self.prefix_key = None
        self.grid_columns = 4
        self.thumb_size = 200
        self.tabs_data = {}
        self.selected_images = set()

        # 【新增】撤回功能相关
        self.operation_history = []
        self.max_history = 20  # 最多保存20次操作
        # 【新增】重做历史
        self.redo_history = []

        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.setWindowTitle("分类数据集整理工具")
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
        self.operation_label = QLabel("就绪")
        self.operation_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.operation_label)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #666; padding: 0 10px;")
        self.status_bar.addWidget(self.info_label)

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

        # 状态栏打印类别图片总数
        self.info_label.setText(f"总数：{len(files)}")

        for i, filename in enumerate(files):
            full_path = os.path.join(folder_path, filename)
            try:
                # 【修改】传入 txt_dir 和 class_names
                img_widget = ImageLabel(full_path, self.txt_dir)
                img_widget.update_display(self.thumb_size)

                item_container = QWidget()
                item_layout = QVBoxLayout(item_container)
                item_layout.setContentsMargins(2, 2, 2, 2)
                item_layout.setSpacing(4)
                item_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
                item_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

                item_layout.addWidget(img_widget, 0, Qt.AlignCenter)

                name_label = QLabel(filename[:25] + "..." if len(filename) > 25 else filename)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setStyleSheet("font-size: 11px; color: #333;")
                name_label.setWordWrap(True)
                item_layout.addWidget(name_label, 0, Qt.AlignCenter)

                row = i // self.grid_columns
                col = i % self.grid_columns
                grid_layout.addWidget(item_container, row, col, Qt.AlignCenter)
            except Exception as e:
                pass

    def remove_images_from_ui(self, images_to_remove):
        # 1. 立即停止界面刷新和事件处理（防止闪烁和冲突）
        self.setUpdatesEnabled(False)

        current_idx = self.tab_widget.currentIndex()
        grid_layout = self.tabs_data[current_idx]['grid_layout']

        try:
            # 2. 预先找出所有受影响的容器，并断开它们与 ImageLabel 的联系
            containers_to_kill = []
            for img in images_to_remove:
                # 这里的检查非常重要，防止对已销毁对象的二次操作
                try:
                    p = img.parentWidget()
                    if p:
                        containers_to_kill.append(p)
                        # 关键：手动清除引用，防止后续事件触发
                        img.setParent(None)
                except RuntimeError:
                    continue

            # 3. 提取保留的组件
            remaining_containers = []
            for i in range(grid_layout.count()):
                w = grid_layout.itemAt(i).widget()
                if w and w not in containers_to_kill:
                    remaining_containers.append(w)

            # 4. 清空布局（只解除关联，不销毁）
            while grid_layout.count() > 0:
                grid_layout.takeAt(0)

            # 5. 彻底销毁被删除的容器
            for container in containers_to_kill:
                container.deleteLater()

            # 6. 重新填装保留的组件
            for i, container in enumerate(remaining_containers):
                row = i // self.grid_columns
                col = i % self.grid_columns
                grid_layout.addWidget(container, row, col, Qt.AlignCenter)

            self.info_label.setText(f"总数：{len(remaining_containers)}")

        finally:
            # 7. 无论如何都要恢复界面刷新，否则界面会卡死
            self.setUpdatesEnabled(True)
            self.update()  # 强制重绘一次


    def handle_zoom(self, delta):
        old_cols = self.grid_columns
        if delta > 0:
            self.grid_columns = min(10, self.grid_columns + 1)
        else:
            self.grid_columns = max(1, self.grid_columns - 1)
        if old_cols != self.grid_columns:
            self.refresh_tab(self.tab_widget.currentIndex())
            self.operation_label.setText(f"列数: {self.grid_columns}")
            self.operation_label.setStyleSheet("color: #666; padding: 0 10px;")

    def undo_operation(self):
        if not self.operation_history:
            self.operation_label.setText("⚠️ 没有可撤回的操作")
            self.operation_label.setStyleSheet("color: #FFA500; padding: 0 10px;")
            return

        # 取出最后一次操作
        last_operation = self.operation_history.pop()
        # 【新增】存入重做历史
        self.redo_history.append(last_operation)

        undo_count = 0

        # 逆向移动：从目标路径移回源路径
        for src_path, dst_path in reversed(last_operation):
            if os.path.exists(dst_path):
                try:
                    # 确保源目录存在（防止文件夹被删）
                    src_dir = os.path.dirname(src_path)
                    if not os.path.exists(src_dir):
                        os.makedirs(src_dir)

                    shutil.move(dst_path, src_path)
                    undo_count += 1
                except Exception as e:
                    print(f"撤回失败: {e}")

        # 刷新当前页面
        current_idx = self.tab_widget.currentIndex()
        self.refresh_tab(current_idx)

        self.operation_label.setText(f"↩️ 已撤回 {undo_count} 张图片")
        self.operation_label.setStyleSheet("color: #1976d2; padding: 0 10px;")

    def redo_operation(self):
        if not self.redo_history:
            self.operation_label.setText("⚠️ 没有可恢复的操作")
            self.operation_label.setStyleSheet("color: #FFA500; padding: 0 10px;")
            return

        # 取出最后一次撤回的操作
        last_operation = self.redo_history.pop()
        # 放回撤回历史
        self.operation_history.append(last_operation)

        redo_count = 0
        # 重新执行移动：从源路径移到目标路径
        for src_path, dst_path in last_operation:
            if os.path.exists(src_path):
                try:
                    dst_dir = os.path.dirname(dst_path)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)

                    shutil.move(src_path, dst_path)
                    redo_count += 1
                except Exception as e:
                    print(f"恢复失败: {e}")

        # 刷新当前页面
        current_idx = self.tab_widget.currentIndex()
        self.refresh_tab(current_idx)

        self.operation_label.setText(f"↪️ 已恢复 {redo_count} 张图片")
        self.operation_label.setStyleSheet("color: #1976d2; padding: 0 10px;")

    def keyPressEvent(self, event):
        key = event.key()

        # 【新增】Ctrl+Z 撤回
        if key == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            self.undo_operation()
            return

        # 【新增】Ctrl+Y 恢复（重做）
        if key == Qt.Key_Y and event.modifiers() == Qt.ControlModifier:
            self.redo_operation()
            return

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

        if key == Qt.Key_Delete:
            self.process_move(len(self.class_names))

        super().keyPressEvent(event)

    def process_move(self, target_idx):
        if target_idx <= 0 or target_idx > len(self.class_names):
            self.operation_label.setText(f"❌ 无效类别序号 {target_idx}")
            self.operation_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
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
            self.operation_label.setText("❗ 请先框选图片或将鼠标悬停在图片上")
            self.operation_label.setStyleSheet("color: #d32f2f; padding: 0 10px;")
            return

        # 新增：删除操作
        if target_idx == len(self.class_names):
            target_dir = os.path.join(self.root_dir, "垃圾桶")
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_class = self.class_names[target_idx - 1]
            target_dir = os.path.join(self.root_dir, target_class)
        success_count = 0

        # 【新增】记录本次操作的所有文件路径
        current_operation = []

        for img_widget in images_to_move:
            src_path = img_widget.image_path
            if not os.path.exists(src_path):
                continue

            filename:str = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)
            success_count += 1
            # 【新增】记录源路径和最终的目标路径
            current_operation.append((src_path, dst_path))

        # 【新增】将本次成功的操作存入历史，并清空重做栈
        if current_operation:
            self.operation_history.append(current_operation)
            self.redo_history.clear()  # 新增：有新操作就不能重做之前的了
            # 限制历史长度
            if len(self.operation_history) > self.max_history:
                self.operation_history.pop(0)

        # self.refresh_tab(current_idx)
        self.remove_images_from_ui(images_to_move)

        if success_count > 0:
            if target_idx == len(self.class_names):
                self.operation_label.setText(f"✅ 已批量删除 {success_count} 张至 [{target_idx}]")
            else:
                self.operation_label.setText(f"✅ 已批量移动 {success_count} 张至 [{target_idx}] {target_class}")
                self.operation_label.setStyleSheet("color: #388e3c; padding: 0 10px;")

def main():
    app = QApplication(sys.argv)

    root_dir = QFileDialog.getExistingDirectory(None, "请选择数据集根目录")
    if not root_dir:
        return

    # 【新增】选择 TXT 标签目录（可跳过）
    raw_model_predict_path_txt = os.path.join(root_dir, "raw_model_predict_path.txt")
    if os.path.exists(raw_model_predict_path_txt):
        with open(raw_model_predict_path_txt, 'r') as txt_path_file:
            txt_dir = txt_path_file.readline().strip()
            txt_dir = os.path.join(txt_dir, "labels")
            # print(txt_dir)
        if not os.path.exists(txt_dir):
            txt_dir = None
    else:
        txt_dir = None

    if txt_dir is None:
        txt_dir = QFileDialog.getExistingDirectory(None, "请选择 TXT 标签目录（点击取消则不展示置信度）")
    if not txt_dir:
        txt_dir = None

    subfolders = [f.name for f in os.scandir(root_dir) if f.is_dir()]  # scandir和listdir有啥区别
    if "垃圾桶" in subfolders:
        subfolders.remove("垃圾桶")
    try:
        subfolders.sort(key=int)
    except ValueError:
        subfolders.sort()
    default_classes = ", ".join(subfolders) if subfolders else "class_1, class_2, class_3"

    text, ok = QInputDialog.getText(None, "设置类别", "请输入类别名称（按顺序，逗号分隔）：", text=default_classes)
    if not ok or not text:
        return

    class_names = [x.strip() for x in text.split(',')]

    window = DatasetSorter(root_dir, class_names, txt_dir)

    global_filter = GlobalFilter(window)
    app.installEventFilter(global_filter)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()