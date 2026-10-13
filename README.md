# 手写数字识别器（MNIST + PyTorch + Tkinter）

一个基于 PyTorch 和 MNIST 数据集的手写数字识别程序，提供图形界面，支持鼠标手写输入、实时识别，并显示 0~9 每个数字的预测概率。

## 功能特性

- 🖌️ **手写输入**：在白色画布上用鼠标书写数字，画布以 10×10 像素块模拟 28×28 分辨率。
- 📊 **概率展示**：以条形图显示 0~9 每个类别的预测概率。
- 🧠 **CNN 模型**：使用卷积神经网络，在 MNIST 测试集上准确率可达 99% 以上。
- 🔄 **自动预处理**：裁剪、反转、缩放、质心居中，对齐 MNIST 数据分布。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- numpy
- Tkinter（Python 标准库，Linux 下可能需要单独安装）

### Linux 安装 Tkinter

```bash
sudo apt-get install python3-tk