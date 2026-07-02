import os
import random
import ast
import tkinter as tk
from PIL import Image, ImageTk

# ========== 配置参数 ==========
GRID_SIZE = 3          # 3x3 九宫格
CELL_SIZE = 400        # 每个格子像素大小
IMG_MARGIN = 15        # 图片距离格子边框的边距
EMPTY_COUNT = 3        # 空位数量
BLUE_COUNT = 3         # 蓝色数量
RED_COUNT = 3          # 红色数量
MAX_CATEGORY = 31      # 类别 1~31
SEQ_FILE = "./config/sudoku.txt"
RED_FOLDER = r"/home/awwsome/RC/juanZhou/Red"
BLUE_FOLDER = r"/home/awwsome/RC/juanZhou/Blue"
IMG_EXT = ".png"       # 图片后缀
DEBUG_MODE = False     # 调试模式：True 会在每个图片位置画红框，方便确认坐标

# 类别范围约束
GROUP_A_CATEGORY = 1        # 位置1-3固定只能用类别1
GROUP_B_MIN_CAT = 2         # 位置4-9最小类别
GROUP_B_MAX_CAT = 16        # 位置4-9最大类别

# 列表索引 0~8 对应 位置编号 1~9
GRID_INDEX_MAP = [
    [6, 7, 8],   # 第0行（最上面）：位置7 8 9
    [3, 4, 5],   # 第1行：位置4 5 6
    [0, 1, 2]    # 第2行（最下面）：位置1 2 3
]

GROUP_A_INDICES = [0, 1, 2]   # 位置1/2/3（A组）
GROUP_B_INDICES = [3, 4, 5, 6, 7, 8]  # 位置4~9（B组）

BATCH_SIZE = 4  # 每多少组换一次类别

# 兼容 Pillow 新旧版本
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


# ========== 工具函数 ==========
def find_image_path(cat, color):
    folder = RED_FOLDER if color == "red" else BLUE_FOLDER
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
        path = os.path.join(folder, f"{cat}{ext}")
        if os.path.isfile(path):
            return path
    return None


def load_history():
    """加载历史序列集合"""
    history = set()
    if os.path.exists(SEQ_FILE):
        with open(SEQ_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seq = ast.literal_eval(line)
                        history.add(tuple(seq))
                    except:
                        continue
    return history


def save_sequence(seq_list):
    """追加保存序列到文件"""
    with open(SEQ_FILE, "a", encoding="utf-8") as f:
        f.write(str(seq_list) + "\n")


# ========== 主窗口类 ==========
class GridWindow:
    def __init__(self, root):
        self.root = root
        self.history = load_history()
        self.current_seq = None
        self.images = []
        self.gen_count = len(self.history)

        # 批次控制
        self.batch_remaining = 0
        self.a_pool = []  # A组元素池（类别1 + 空位）
        self.b_pool = []  # B组元素池（类别2-16 + 空位）

        total_size = CELL_SIZE * GRID_SIZE
        self.canvas = tk.Canvas(root, width=total_size, height=total_size,
                                bg="white", highlightthickness=0)
        self.canvas.pack(padx=20, pady=20)

        # 底部提示文字
        self.tip_label = tk.Label(root, text="按 Enter 键生成下一组",
                                  font=("微软雅黑", 10), fg="gray")
        self.tip_label.pack(pady=(0, 15))

        # 绑定回车键
        root.bind("<Return>", self.next_sequence)

        # 初始生成第一组
        self.next_sequence()

    def _new_batch(self):
        """开启新批次：固定类别集合，接下来BATCH_SIZE组只打乱位置"""
        # 1. 随机分配A组内 蓝/红/空 的数量
        max_blue_a = min(len(GROUP_A_INDICES), BLUE_COUNT)
        a_blue = random.randint(0, max_blue_a)
        max_red_a = min(len(GROUP_A_INDICES) - a_blue, RED_COUNT)
        a_red = random.randint(0, max_red_a)
        a_empty = len(GROUP_A_INDICES) - a_blue - a_red

        # 2. 推算B组内 蓝/红/空 的数量
        b_blue = BLUE_COUNT - a_blue
        b_red = RED_COUNT - a_red
        b_empty = EMPTY_COUNT - a_empty

        # 3. 抽取B组类别（2~16 不重复）
        blue_b_cats = random.sample(range(GROUP_B_MIN_CAT, GROUP_B_MAX_CAT + 1), b_blue)
        red_b_cats = random.sample(range(GROUP_B_MIN_CAT, GROUP_B_MAX_CAT + 1), b_red)

        # 4. 构造元素池
        self.a_pool = ['1_blue'] * a_blue + ['1_red'] * a_red + [0] * a_empty
        self.b_pool = [f"{cat}_blue" for cat in blue_b_cats] + \
                      [f"{cat}_red" for cat in red_b_cats] + \
                      [0] * b_empty

        self.batch_remaining = BATCH_SIZE
        print(f"\n=== 新批次 ===")
        print(f"A组(位置1-3)：蓝{a_blue} 红{a_red} 空{a_empty}")
        print(f"B组(位置4-9)：蓝{blue_b_cats} 红{red_b_cats} 空{b_empty}")

    def _clear_canvas(self):
        """清空画布与图片引用"""
        self.canvas.delete("all")
        self.images.clear()

    def _draw_grid(self):
        """绘制黑色九宫格边框"""
        total = CELL_SIZE * GRID_SIZE
        self.canvas.create_rectangle(0, 0, total, total,
                                     outline="black", width=3)
        for i in range(1, GRID_SIZE):
            x = i * CELL_SIZE
            self.canvas.create_line(x, 0, x, total, fill="black", width=2)
            y = i * CELL_SIZE
            self.canvas.create_line(0, y, total, y, fill="black", width=2)

    def _draw_images(self):
        """根据当前序列绘制图片"""
        img_size = CELL_SIZE - IMG_MARGIN * 2

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                idx = GRID_INDEX_MAP[row][col]
                item = self.current_seq[idx]
                if item == 0:
                    continue

                cat_str, color = item.split("_")
                cat = int(cat_str)

                cx = col * CELL_SIZE + CELL_SIZE // 2
                cy = row * CELL_SIZE + CELL_SIZE // 2

                if DEBUG_MODE:
                    x0 = col * CELL_SIZE + IMG_MARGIN
                    y0 = row * CELL_SIZE + IMG_MARGIN
                    x1 = x0 + img_size
                    y1 = y0 + img_size
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="red")

                path = find_image_path(cat, color)
                if not path:
                    self.canvas.create_text(cx, cy, text=f"{cat}\n({color})",
                                            font=("Arial", 14), fill="red")
                    continue

                try:
                    pil_img = Image.open(path).convert("RGBA")
                    pil_img = pil_img.resize((img_size, img_size), RESAMPLE)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.images.append(tk_img)
                    self.canvas.create_image(cx, cy, image=tk_img)
                except Exception as e:
                    self.canvas.create_text(cx, cy, text="加载失败", fill="red")
                    print(f"[错误] {path}: {e}")

    def _refresh(self):
        """刷新整个画布"""
        self._clear_canvas()
        self._draw_grid()
        self._draw_images()
        self.canvas.update()

    def next_sequence(self, event=None):
        """生成下一组不重复序列并刷新显示"""
        for _ in range(10000):
            # 批次用完则自动开启新批次
            if self.batch_remaining <= 0:
                self._new_batch()

            # 打乱两组元素池，生成新序列（类别不变，位置变）
            a_shuffled = self.a_pool.copy()
            random.shuffle(a_shuffled)
            b_shuffled = self.b_pool.copy()
            random.shuffle(b_shuffled)
            seq = a_shuffled + b_shuffled  # 索引0-2为A组，3-8为B组

            seq_tuple = tuple(seq)
            if seq_tuple not in self.history:
                self.current_seq = seq
                self.history.add(seq_tuple)
                save_sequence(seq)
                self.gen_count += 1
                self.batch_remaining -= 1

                self.tip_label.config(
                    text=f"第 {self.gen_count} 组 | 本批次剩余 {self.batch_remaining} 组 | 按 Enter 生成下一组"
                )
                print(f"第{self.gen_count}组: {seq}")
                self._refresh()
                return

        # 生成失败提示
        self.tip_label.config(text="已无法生成更多不重复序列", fg="red")
        print("警告：尝试次数过多，无法生成新序列")


# ========== 主入口 ==========
def main():
    root = tk.Tk()
    root.title("九宫格序列生成器")
    root.resizable(False, False)
    GridWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()