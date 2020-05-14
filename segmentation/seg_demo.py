import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image

path_img= 'imgs/demo_img3.png'

preprocess=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

input_image=Image.open(path_img).convert('RGB')
# 加载 pytorch/vision 库里的 deeplabv3_resnet101 模型， pretrained=True 表示加载在 Pascal VOC 数据集上预训练好的权重
model=torch.hub.load('pytorch/vision','deeplabv3_resnet101', pretrained=True)
# 设置模型为非训练状态
model.eval()

# 对图片进行预处理
input_tensor=preprocess(input_image)
# unsqueeze 的作用是增加维度，第一个维度为 Batch，形成一个 Batch再输入模型：[3, H, W] 变为 [1, 3, H, W]
input_bchw=input_tensor.unsqueeze(0)

with torch.no_grad():
    # 模型输出是一个 dict, 我们需要获取 out 对应的 4d 张量：output_4d.shape: [1, 21, H, W]
    output_4d=model(input_bchw)['out']
    # 取出 Batch 的第一个数据， output.shape: [21, H, W]
    output=output_4d[0]
    # 取出每个像素概率最大的类别对应的index, output_predictions.shape: [H, W]
    output_predictions=output.argmax(0)
    # 由于 Pascal VOC 数据集有 21 个类别，因此这里需要生成 21 个类别的不同颜色。
    # 这是RGB颜色映射矩阵，至于为什么是这些数就不清楚了，这里就是自动的生成21个颜色，并且颜色要有一定区分，
    # 也可以手动设置21个RGB颜色
    # palette: tensor([33554431,    32767,  2097151])
    palette=torch.tensor([2**25-1,2**15-1,2**21-1])
    # colors.size: [21, 3]
    colors=torch.as_tensor([i for i in range(21)])[:,None]*palette
    # 转化为 0-255 之间
    colors=(colors%255).numpy().astype('uint8')
    r=Image.fromarray(output_predictions.byte().cpu().numpy())
    # 把类别映射到颜色
    r.putpalette(colors)
    plt.subplot(121).imshow(r)
    plt.subplot(122).imshow(input_image)
    # 需要先调用 plt.savefig()，再调用 plt.show()
    # 在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片。
    # 原因参考 https://blog.csdn.net/u010099080/article/details/52912439
    plt.savefig("segmentation.png")
    plt.show()

