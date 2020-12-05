

# PyTorch 学习笔记

<p align='center'>
<a href="https://github.com/zhangxiann/PyTorch_Practice" target="_blank"><img alt="GitHub" src="https://img.shields.io/github/stars/zhangxiann/PyTorch_Practice?label=Stars&style=flat-square&logo=GitHub"></a>
<a href="https://pytorch.zhangxiann.com/" target="_blank"><img alt="Website" src="https://img.shields.io/website?label=%E5%9C%A8%E7%BA%BF%E7%94%B5%E5%AD%90%E4%B9%A6&style=flat-square&down_color=blue&down_message=%E7%82%B9%E8%BF%99%E9%87%8C&up_color=blue&up_message=%E7%82%B9%E8%BF%99%E9%87%8C&url=https://pytorch.zhangxiann.com/&logo=Gitea"></a>
</p>

<p align='center'>
<a href="https://www.github.com/zhangxiann" target="_blank"><img src="https://img.shields.io/badge/作者-@zhangxiann-000000.svg?style=flat-square&logo=GitHub"></a>
<a href="https://www.zhihu.com/people/zhangxiann/posts" target="_blank"><img src="https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-@张贤同学-000000.svg?style=flat-square&logo=Zhihu"></a>
<a href="https://image.zhangxiann.com/QRcode_8cm.jpg" target="_blank"><img src="https://img.shields.io/badge/公众号-@张贤同学-000000.svg?style=flat-square&logo=WeChat"></a>
</p>


这个仓库是我学习 PyTorch 过程中所记录的学习笔记汇总，包括 **25** 篇文章，是我学习 **PyTorch** 期间所记录的内容，点击查看在线电子书：[https://pytorch.zhangxiann.com/](https://pytorch.zhangxiann.com/)。

学习笔记的结构遵循课程的顺序，共分为 8 周，循序渐进，**力求通俗易懂**。



## 代码

配套代码：[https://github.com/zhangxiann/PyTorch_Practice](https://github.com/zhangxiann/PyTorch_Practice)

所有代码均在 PyCharm 中通过测试，建议通过 git 克隆到本地运行。

<!--more-->



## 数据

由于代码中会用到一些第三方的数据集，这里给出百度云的下载地址（如果有其他更好的数据托管方式，欢迎告诉我）。

数据下载地址：
链接：https://pan.baidu.com/s/1f9wQM7gvkMVx2x5z6xC9KQ 
提取码：w7xt



## 面向读者

本教程假定读你有一定的机器学习和深度学习基础。

如果你没有学习过机器学习或者深度学习，建议先观看 Andrew ng 的深度学习（Deep Learning）课程，课程地址： [https://mooc.study.163.com/university/deeplearning_ai#/c](https://mooc.study.163.com/university/deeplearning_ai#/c)。

然后再学习本教程，效果会更佳。



## 学习计划

这个学习笔记共 25 章，分为 8 周进行的，每周大概 3 章（当然你可以根据自己的进度调整），每章花费的时间约 30 分钟到 2 个小时之间。

目录大纲如下：

- **Week 1（基本概念）**
  - [1.1 PyTorch 简介与安装](https://zhuanlan.zhihu.com/p/185037101)
  - [1.2 Tensor(张量)介绍](https://zhuanlan.zhihu.com/p/187564399)
  - [1.3 张量操作与线性回归](https://zhuanlan.zhihu.com/p/189952916)
  - [1.4 计算图与动态图机制](https://zhuanlan.zhihu.com/p/191648279)
  - [1.5 autograd 与逻辑回归](https://zhuanlan.zhihu.com/p/191652343)
- **Week 2（图片处理与数据加载）**
  - [2.1 DataLoader 与 DataSet](https://zhuanlan.zhihu.com/p/197888612)
  - [2.2 图片预处理 transforms 模块机制](https://zhuanlan.zhihu.com/p/200866666)
  - [2.3 二十二种 transforms 图片数据预处理方法](https://zhuanlan.zhihu.com/p/200876072)
- **Week 3（模型构建）**
  - [3.1 模型创建步骤与 nn.Module](https://zhuanlan.zhihu.com/p/203405689)
  - [3.2 卷积层](https://zhuanlan.zhihu.com/p/206427963)
  - [3.3 池化层、线性层和激活函数层](https://zhuanlan.zhihu.com/p/208259650)
- **Week 4（模型训练）**
  - [4.1 权值初始化](https://zhuanlan.zhihu.com/p/210137182)
  - [4.2 损失函数](https://zhuanlan.zhihu.com/p/212691653)
  - [4.3 优化器](https://zhuanlan.zhihu.com/p/213824542)
- **Week 5（可视化与 Hook）**
  - [5.1 TensorBoard 介绍](https://zhuanlan.zhihu.com/p/217415374)
  - [5.2 Hook 函数与 CAM 算法](https://zhuanlan.zhihu.com/p/222496848)
- **Week 6（正则化）**
  - [6.1 weight decay 和 dropout](https://zhuanlan.zhihu.com/p/225606205)
  - [6.2 Normalization](https://zhuanlan.zhihu.com/p/232487440)
- **Week 7（模型其他操作）**
  - [7.1 模型保存与加载](https://zhuanlan.zhihu.com/p/245645490)
  - [7.2 模型 Finetune](https://zhuanlan.zhihu.com/p/245652282)
  - [7.3 使用 GPU 训练模型](https://zhuanlan.zhihu.com/p/254738836)
- **Week 8（实际应用）**
  - [8.1 图像分类简述与 ResNet 源码分析](https://zhuanlan.zhihu.com/p/254761587)
  - [8.2 目标检测简介](https://zhuanlan.zhihu.com/p/259494709)
  - [8.3 GAN（生成对抗网络）简介](https://zhuanlan.zhihu.com/p/258321589)
  - [8.4 手动实现 RNN](https://zhuanlan.zhihu.com/p/263531494)
