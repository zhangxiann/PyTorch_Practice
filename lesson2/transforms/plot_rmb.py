# 画图查看四套人民币和四五套人民币
import os
from enviroments import project_dir,rmb_split_dir
from PIL import Image
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 第四套1元人民币
path_img1_4=os.path.join(rmb_split_dir, "train", "1","0B89KOA3.jpg")
path_img1_4 = Image.open(path_img1_4).convert('RGB')
# 第四套100元人民币
path_img100_4=os.path.join(rmb_split_dir, "train", "100","0A4DSPGE.jpg")
path_img100_4 = Image.open(path_img100_4).convert('RGB')
# 第五套100元人民币
path_img100_5=os.path.join(project_dir, "data", "rmb_test_data", "100","1001.jpg")
path_img100_5 = Image.open(path_img100_5).convert('RGB')

plt.subplot(3, 1, 1)
plt.title("第四套1元人民币")
plt.imshow(path_img1_4)

plt.subplot(3, 1, 2)
plt.title("第四套100元人民币")
plt.imshow(path_img100_4)

plt.subplot(3, 1, 3)
plt.title("第五套100元人民币")
plt.imshow(path_img100_5)
plt.show()

plt.pause(0.5)
plt.close()