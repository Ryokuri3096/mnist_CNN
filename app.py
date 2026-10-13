import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


# ======================
# 模型结构（必须与训练时完全一致）
# ======================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ======================
# 主应用类
# ======================
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别器")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")  # 深色窗口背景
        self.last_cell = None   # 记录上一个格子坐标，用于拖拽时插值

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载整个模型
        try:
            self.model = Net().to(self.device)
            self.model.load_state_dict(torch.load("CNN_model.pth", map_location=self.device))
            self.model.eval()
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}\n请确认模型文件路径和结构。")
            self.root.destroy()
            return

        # 预处理转换（与训练时一致）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self._build_ui()

    def _build_ui(self):
        # 白色画布
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white",
                                cursor="cross", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # 按钮区域
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="清空", command=self._clear_canvas, width=10,
                font=("Cascadia Code", 12), bg="#333333", fg="white",
                activebackground="#444444", activeforeground="white",
                relief=tk.FLAT).pack(side=tk.LEFT, padx=10)

        # 结果显示标签
        self.result_label = tk.Label(self.root, text="请绘制一个数字 (0-9)",
                                     font=("Cascadia Code", 14), bg="#1e1e1e", fg="white")
        self.result_label.pack(pady=5)

        # 概率条形图区域（深色背景）
        self.prob_canvas = tk.Canvas(self.root, width=400, height=240,
                                     bg="#1e1e1e", highlightthickness=0)
        self.prob_canvas.pack(pady=5)

        # 初始化 PIL 图像：白色背景
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def _paint(self, event):
        """按 10x10 格子绘制，模拟 28x28 分辨率"""
        gx = event.x // 10
        gy = event.y // 10

        if self.last_cell is not None:
            lx, ly = self.last_cell
            # 在上一格和当前格之间插值，避免快速拖动时出现断点
            steps = max(abs(gx - lx), abs(gy - ly))
            if steps == 0:
                self._draw_block(gx, gy)
            else:
                for i in range(steps + 1):
                    t = i / steps
                    ix = round(lx + (gx - lx) * t)
                    iy = round(ly + (gy - ly) * t)
                    self._draw_block(ix, iy)
        else:
            self._draw_block(gx, gy)

        self.last_cell = (gx, gy)

    def _draw_block(self, gx, gy):
        """绘制一个 10x10 的方块，对应 28x28 图像中的一个像素"""
        if not (0 <= gx < 28 and 0 <= gy < 28):
            return
        x1, y1 = gx * 10, gy * 10
        x2, y2 = x1 + 10, y1 + 10
        # Canvas 上画 10x10 黑块
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="#1c1c1c")
        # PIL 图像上同步画黑色方块
        self.draw.rectangle([x1, y1, x2, y2], fill=0)

    def _on_release(self, event):
        """松开鼠标时立即识别（延迟 50ms 让最后一笔绘制完成）"""
        self.last_cell = None
        self.root.after(50, self._predict)

    def _clear_canvas(self):
        """清空画布和 PIL 图像"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_cell = None
        self.result_label.config(text="请绘制一个数字 (0-9)")
        self.prob_canvas.delete("all")

    def _preprocess_image(self):
        """将白底黑字处理成 MNIST 风格的黑底白字，并按质心居中到 28x28"""
        bbox = self.image.getbbox()
        if not bbox:
            return None, None

        digit = self.image.crop(bbox)
        # 反转为黑底白字，匹配 MNIST
        digit = ImageOps.invert(digit)

        # 1. 缩放到最长边为 20 像素，保持比例
        w, h = digit.size
        if w > h:
            new_w = 20
            new_h = max(1, int(round(h * 20 / w)))
        else:
            new_h = 20
            new_w = max(1, int(round(w * 20 / h)))
        digit = digit.resize((new_w, new_h), Image.LANCZOS)

        # 2. 计算质心（灰度加权中心）
        arr = np.array(digit, dtype=np.float32)
        total = arr.sum()
        if total > 0:
            ys, xs = np.indices(arr.shape)
            cy = (ys * arr).sum() / total
            cx = (xs * arr).sum() / total
        else:
            cy, cx = arr.shape[0] / 2, arr.shape[1] / 2

        # 3. 放到 28x28 黑底画布上，使质心位于 (14, 14)
        new_img = Image.new("L", (28, 28), color=0)
        paste_x = int(round(14 - cx))
        paste_y = int(round(14 - cy))
        new_img.paste(digit, (paste_x, paste_y))

        tensor = self.transform(new_img).unsqueeze(0)
        return tensor, new_img
    
    def _predict(self):
        """执行识别并显示概率"""
        tensor, processed_img = self._preprocess_image()
        if tensor is None:
            # messagebox.showwarning("提示", "请先在画布上绘制一个数字。")
            return

        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)[0]
            predicted = torch.argmax(probabilities).item()
            confidence = probabilities[predicted].item() * 100

        self.result_label.config(text=f"识别结果: {predicted}  (置信度: {confidence:.2f}%)")
        self._draw_probability_bars(probabilities.cpu().numpy())

    def _draw_probability_bars(self, probs):
        """绘制 0-9 的概率条形图"""
        self.prob_canvas.delete("all")
        bar_width = 30
        gap = 5
        start_x = 20
        base_y = 200
        max_height = 180
        predicted_idx = probs.argmax()

        for i, p in enumerate(probs):
            x1 = start_x + i * (bar_width + gap)
            y1 = base_y - p * max_height
            x2 = x1 + bar_width
            y2 = base_y

            # 最高概率用绿色，其余用蓝色
            color = "#4CAF50" if i == predicted_idx else "#00A8E8"
            self.prob_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            self.prob_canvas.create_text(x1 + bar_width / 2, base_y + 15,
                                         text=str(i), font=("Cascadia Code", 10), fill="white")
            if p > 0.05:
                self.prob_canvas.create_text(x1 + bar_width / 2, y1 - 8,
                                             text=f"{p:.2f}", font=("Cascadia Code", 8), fill="white")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()