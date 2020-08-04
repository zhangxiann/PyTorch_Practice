; # PyTorch_Practice
; This repository is for my practice of PyTorch

; ## Semantic Segmentation

; - `seg_demo.py` : use  `torch.hub` to load deeplab-v3 to inference segmentation on some images in the `imgs` folder.
; - `unet_portrait_matting.py` : use a subset of [PortraitDataset](http://xiaoyongshen.me/webpage_portrait/index.html) to train Unet from scratch, you can ; download the sub dataset from [待补充]()
; - `portrait_inference.py` : load the train weights(`checkpoint_399_epoch.pkl`) to the Unet model, and inference some images from [PortraitDataset];;(http://xiaoyongshen.me/webpage_portrait/index.html)
; - `checkpoint_399_epoch.pkl` : save the weights after training Unet 400 epochs.

这是用于记录学习 [深度之眼](https://ai.deepshare.net/) **PyTorch** 框架版课程期间的用到的代码。
完整笔记请查看 [https://blog.zhangxiann.com/2020/06/01/PyTorch/PyTorch%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/](https://blog.zhangxiann.com/2020/06/01/PyTorch/PyTorch%20%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/)
