# -*- coding: utf-8 -*-
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import imageio
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from common_tools import set_seed
from torch.utils.data import DataLoader
from my_dataset import CelebADataset
from dcgan import Discriminator, Generator
import enviroments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(1)  # 设置随机种子

# confg

# data_dir = os.path.join(BASE_DIR, "..", "..", "data", "img_align_celeba_2k")
data_dir = enviroments.img_align_celeba
out_dir = os.path.join(BASE_DIR, "..", "log_gan")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ngpu = 0    # Number of GPUs available. Use 0 for CPU mode.
IS_PARALLEL = True if ngpu > 1 else False
checkpoint_interval = 10

image_size = 64
nc = 3
nz = 100
ngf = 128  # 64
ndf = 128   # 64
num_epochs = 20
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_idx = 1    # 0.9
fake_idx = 0    # 0.1

lr = 0.0002
batch_size = 64
beta1 = 0.5

d_transforms = transforms.Compose([transforms.Resize(image_size),
                   transforms.CenterCrop(image_size),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -1 ,1
               ])
if __name__ == '__main__':
    # step 1: data

    train_set = CelebADataset(data_dir=data_dir, transforms=d_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)

    # show train img
    flag = 0
    # flag = 1
    if flag:
        img_bchw = next(iter(train_loader))
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(img_bchw.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        plt.close()

    # step 2: model
    net_g = Generator(nz=nz, ngf=ngf, nc=nc)
    net_g.initialize_weights()

    net_d = Discriminator(nc=nc, ndf=ndf)
    net_d.initialize_weights()

    net_g.to(device)
    net_d.to(device)

    if IS_PARALLEL and torch.cuda.device_count() > 1:
        net_g = nn.DataParallel(net_g)
        net_d = nn.DataParallel(net_d)

    # step 3: loss
    criterion = nn.BCELoss()

    # step 4: optimizer
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=8, gamma=0.1)
    lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=8, gamma=0.1)

    # step 5: iteration
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):

            ############################
            # (1) Update D network
            ###########################

            net_d.zero_grad()

            # create training data
            real_img = data.to(device)
            b_size = real_img.size(0)
            # 根据 (b_size,) 构造矩阵，使用 real_idx填充
            real_label = torch.full((b_size,), real_idx, device=device)

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_img = net_g(noise)
            fake_label = torch.full((b_size,), fake_idx, device=device)

            # train D with real img
            out_d_real = net_d(real_img)
            loss_d_real = criterion(out_d_real.view(-1), real_label)

            # train D with fake img
            out_d_fake = net_d(fake_img.detach())
            loss_d_fake = criterion(out_d_fake.view(-1), fake_label)

            # backward
            loss_d_real.backward()
            loss_d_fake.backward()
            # 损失函数使用两者之和
            loss_d = loss_d_real + loss_d_fake

            # Update D
            optimizerD.step()

            # record probability
            d_x = out_d_real.mean().item()      # D(x)
            d_g_z1 = out_d_fake.mean().item()   # D(G(z1))

            ############################
            # (2) Update G network
            ###########################
            net_g.zero_grad()

            label_for_train_g = real_label  # 1
            out_d_fake_2 = net_d(fake_img)

            loss_g = criterion(out_d_fake_2.view(-1), label_for_train_g)
            loss_g.backward()
            optimizerG.step()

            # record probability
            d_g_z2 = out_d_fake_2.mean().item()  # D(G(z2))

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         loss_d.item(), loss_g.item(), d_x, d_g_z1, d_g_z2))

            # Save Losses for plotting later
            G_losses.append(loss_g.item())
            D_losses.append(loss_d.item())

        lr_scheduler_d.step()
        lr_scheduler_g.step()

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = net_g(fixed_noise).detach().cpu()
        img_grid = vutils.make_grid(fake, padding=2, normalize=True).numpy()
        img_grid = np.transpose(img_grid, (1, 2, 0))
        plt.imshow(img_grid)
        plt.title("Epoch:{}".format(epoch))
        # plt.show()
        plt.savefig(os.path.join(out_dir, "{}_epoch.png".format(epoch)))

        # checkpoint
        if (epoch+1) % checkpoint_interval == 0:

            checkpoint = {"g_model_state_dict": net_g.state_dict(),
                          "d_model_state_dict": net_d.state_dict(),
                          "epoch": epoch}
            path_checkpoint = os.path.join(out_dir, "checkpoint_{}_epoch.pkl".format(epoch))
            torch.save(checkpoint, path_checkpoint)

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(out_dir, "loss.png"))

    # save gif
    imgs_epoch = [int(name.split("_")[0]) for name in list(filter(lambda x: x.endswith("epoch.png"), os.listdir(out_dir)))]
    imgs_epoch = sorted(imgs_epoch)

    imgs = list()
    for i in range(len(imgs_epoch)):
        img_name = os.path.join(out_dir, "{}_epoch.png".format(imgs_epoch[i]))
        imgs.append(imageio.imread(img_name))

    imageio.mimsave(os.path.join(out_dir, "generation_animation.gif"), imgs, fps=2)

    print("done")

