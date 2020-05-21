# PyTorch_Practice
This repository is for my practice of PyTorch

## Semantic Segmentation

- `seg_demo.py` : use  `torch.hub` to load deeplab-v3 to inference segmentation on some images in the `imgs` folder.
- `unet_portrait_matting.py` : use a subset of [PortraitDataset](http://xiaoyongshen.me/webpage_portrait/index.html) to train Unet from scratch, you can download the sub dataset from [待补充]()
- `portrait_inference.py` : load the train weights(`checkpoint_399_epoch.pkl`) to the Unet model, and inference some images from [PortraitDataset](http://xiaoyongshen.me/webpage_portrait/index.html)
- `checkpoint_399_epoch.pkl` : save the weights after training Unet 400 epochs.

