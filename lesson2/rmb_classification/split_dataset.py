# -*- coding: utf-8 -*-

import os
import random
import shutil

# 创建文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)
    # 'data\\RMB_data'
    dataset_dir = os.path.join("data", "RMB_data")
    # 'data\\rmb_split'
    split_dir = os.path.join("data", "rmb_split")
    # 'data\\rmb_split\\train'
    train_dir = os.path.join(split_dir, "train")
    # 'data\\rmb_split\\valid'
    valid_dir = os.path.join(split_dir, "valid")
    # 'data\\rmb_split\\test'
    test_dir = os.path.join(split_dir, "test")

    # 训练集
    train_pct = 0.8
    # 验证集
    valid_pct = 0.1
    # 测试集
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):
        # dirs: ['1', '100']
        for sub_dir in dirs:
            # 文件列表
            imgs = os.listdir(os.path.join(root, sub_dir))
            # 取出 jpg 结尾的文件
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            # 计算图片数量
            img_count = len(imgs)
            # 计算训练集索引的结束位置
            train_point = int(img_count * train_pct)
            # 计算验证集索引的结束位置
            valid_point = int(img_count * (train_pct + valid_pct))
            # 把数据划分到训练集、验证集、测试集的文件夹
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)
                # 创建文件夹
                makedir(out_dir)
                # 构造目标文件名
                target_path = os.path.join(out_dir, imgs[i])
                # 构造源文件名
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])
                # 复制
                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
