# coding=utf-8
import torch.autograd
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

# 创建文件夹
from dataset_gan.utils import read_split_data

if not os.path.exists('../img_CGAN'):
    os.mkdir('../img_CGAN')
# GPU

1111111111
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 2, 128, 128)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        channel1 = img[0:1, :, :]  # 提取通道1，形状为 (batch_size, 1, height, width)
        channel2 = img[1:3, :, :]  # 提取通道2，形状为 (batch_size, 1, height, width)
        return channel1,channel2, torch.tensor([1, 2])

    # @staticmethod
    # def collate_fn(batch):
    #     # 官方实现的default_collate可以参考
    #     # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    #     images, labels = tuple(zip(*batch))
    #
    #     images = torch.stack(images, dim=0)
    #     labels = torch.as_tensor(labels)
    #     return images, labels


root = "D:\\YQS\\code\\flower_photos"  # 数据集所在根目录

train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(128),
                                   transforms.CenterCrop(128),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

11
batch_size = 8
   # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

   # print('Using {} dataloader workers'.format(nw))
train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               #num_workers=nw,
                                               )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_epoch = 25


# data loader 数据载入
# dataloader = torch.utils.data.DataLoader(
#     dataset=mnist, batch_size=batch_size, shuffle=True
# )

####### 定义生成器 Generator #####
#是否为反卷积

def blockNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, stride = 2,pad=1, dropout=0.):
    block = nn.Sequential()
    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_deconv' % name,
                         nn.ConvTranspose2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block


class G_Encoder(nn.Module):
    def __init__(self):
        super(G_Encoder,self).__init__()
        self.layer1 = blockNet(in_c=1,out_c=64,name='conv1',size=4,stride=2)
        self.layer2 = blockNet(in_c=64,out_c=128,name='conv2',size=4,stride=2)
        self.layer3 = blockNet(in_c=128,out_c=128,name='conv3',size=4,stride=2)
        self.layer4 = blockNet(in_c=128,out_c=256,name='conv4',size=4,stride=2)
        self.layer5 = blockNet(in_c=256,out_c=256,name='conv5',size=4,stride=2)
        self.layer6 = blockNet(in_c=256,out_c=512,name='conv6',size=4,stride=2)
        self.layer7 = blockNet(in_c=512,out_c=512,name='conv7',size=2,stride=2,pad=0)
        self.fclayer = nn.Linear(512,64)

    def forward(self,x):
        out1 = self.layer1(x)
        print("out1.shape")
        print(out1.shape)
        out2 = self.layer2(out1)
        print("out2.shape")
        print(out2.shape)
        out3 = self.layer3(out2)
        print("out3.shape")
        print(out3.shape)
        out4 = self.layer4(out3)
        print("out4.shape")
        print(out4.shape)
        out5 = self.layer5(out4)
        print("out5.shape")
        print(out5.shape)
        out6 = self.layer6(out5)
        print("out6.shape")
        print(out6.shape)
        out7 = self.layer7(out6)
        print("out7.shape")
        print(out7.shape)
        temp = out7.view(-1,512)
        mark_en = self.fclayer(temp)
        return mark_en,out7,

class G_Decoder(nn.Module):
    def __init__(self):
        super(G_Decoder,self).__init__()
        self.layer1 = blockNet(in_c=1,out_c=64,name='conv1',size=4,stride=2)
        self.layer2 = blockNet(in_c=64,out_c=128,name='conv2',size=4,stride=2)
        self.layer3 = blockNet(in_c=128,out_c=128,name='conv3',size=4,stride=2)
        self.layer4 = blockNet(in_c=128,out_c=256,name='conv4',size=4,stride=2)
        self.layer5 = blockNet(in_c=256,out_c=256,name='conv5',size=4,stride=2)
        self.layer6 = blockNet(in_c=256,out_c=512,name='conv6',size=4,stride=2)
        self.layer7 = blockNet(in_c=512,out_c=512,name='conv7',size=2,stride=2,pad=0)
        self.fclayer = nn.Linear(512, 64)
        self.dlayer1 = blockNet(in_c=1024,out_c=512,transposed=True,name='deconv1',size=2,stride=2,pad=0)
        self.dlayer2 = blockNet(in_c=1024,out_c=256,transposed=True,name='deconv2',size=4,stride=2)
        self.dlayer3 = blockNet(in_c=512,out_c=256,transposed=True,name='deconv3',size=4,stride=2)
        self.dlayer4 = blockNet(in_c=512,out_c=128,transposed=True,name='deconv4',size=4,stride=2)
        self.dlayer5 = blockNet(in_c=256,out_c=128,transposed=True,name='deconv5',size=4,stride=2)
        self.dlayer6 = blockNet(in_c=256,out_c=64,transposed=True,name='deconv6',size=4,stride=2)
        self.dlayer7 = blockNet(in_c=128,out_c=64,transposed=True,name='deconv7',size=2,stride=2,pad=0)
        self.layer8 = blockNet(in_c=64, out_c=64,  name='conv8', size=3, stride=1)
        self.layer9 = blockNet(in_c=64, out_c=64,  name='conv9', size=3, stride=1)
        self.layer10 = blockNet(in_c=64, out_c=2,  name='conv10', size=3, stride=1)

    def forward(self,x,mark_de):
        out1 = self.layer1(x)

        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        print("out7.shape")
        print(out7.shape)
        temp = out7.view(-1, 512)
        mark_en = self.fclayer(temp)
        d_out1 = self.dlayer1(torch.cat((mark_de,out7),1))
        print("d_out1.shape")
        print(d_out1.shape)
        d_out2 = self.dlayer2(torch.cat((d_out1,out6),1))
        print("d_out2.shape")
        print(d_out2.shape)
        d_out3 = self.dlayer3(torch.cat((d_out2,out5),1))
        print("d_out3.shape")
        print(d_out3.shape)
        d_out4 = self.dlayer4(torch.cat((d_out3,out4),1))
        print("d_out4.shape")
        print(d_out4.shape)
        d_out5 = self.dlayer5(torch.cat((d_out4,out3),1))
        print("d_out5.shape")
        print(d_out5.shape)
        d_out6 = self.dlayer6(torch.cat((d_out5,out2),1))
        print("d_out6.shape")
        print(d_out6.shape)
        d_out7 = self.dlayer7(torch.cat((d_out6,out1),1))
        print("d_out7.shape")
        print(d_out7.shape)
        out8 = self.layer8(d_out7)
        print("out8.shape")
        print(out8.shape)
        out9 = self.layer9(out8)
        print("out9.shape")
        print(out9.shape)
        out10 = self.layer10(out9)
        print("out10.shape")
        print(out10.shape)
        return out10
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.G_Encoder   = G_Encoder()
        self.G_Decoder   = G_Decoder()

        self.layer = nn.Linear(66,512)
        self.gen = nn.Sequential(
            nn.Linear(66, 1024),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(1024, 1024),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(1024, 1024),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(1024, 512),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )
    def forward(self,yixing,leigong):
        mark_en,output = self.G_Encoder(yixing)
        print("laigongshape")
        print(leigong.shape)
        print("mark_enshape")
        print(mark_en.shape)
       # leigong = leigong.view(-1,)
        mark = torch.cat((mark_en,leigong),1)
        mark_de = self.gen(mark)
        mark_de = mark_de.view(-1,512,1,1)
        print("mark_De shape")
        print(mark_de.shape)
        output = self.G_Decoder(yixing,mark_de)
        return nn.Tanh()(output),nn.Tanh()(mark)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer0 = nn.Linear(66,128*128)
        self.Relu = nn.LeakyReLU(0.2)
        self.layer1 = blockNet(in_c=4, out_c=16, name='conv1', size=4, stride=2)
        self.layer2 = blockNet(in_c=16, out_c=16, name='conv2', size=4, stride=2)
        self.layer3 = blockNet(in_c=16, out_c=32, name='conv3', size=4, stride=2)
        self.layer4 = blockNet(in_c=32, out_c=32, name='conv4', size=4, stride=2)
        self.layer5 = blockNet(in_c=32, out_c=64, name='conv5', size=4, stride=2)
        self.layer6 = blockNet(in_c=64, out_c=64, name='conv6', size=4, stride=2)
        self.layer7 = blockNet(in_c=64, out_c=128, name='conv7', size=4, stride=2)
        self.fclayer1 = blockNet(in_c=128, out_c=16, name='fconv1', size=3, stride=1)
        self.fclayer2 = blockNet(in_c=16, out_c=16, name='fconv2', size=3, stride=1)
        self.fclayer3 = blockNet(in_c=16, out_c=1, name='fconv3', size=3, stride=1)


    def forward(self,yixing,mark,liuchang):
        mark = self.Relu(self.layer0(mark))
        mark = mark.view(-1,1,128,128)
        print("mark.shape")
        print(mark.shape)
        print("yixing.shape")
        print(yixing.shape)
        out = torch.cat((mark,yixing),1)
        out = torch.cat((out,liuchang),1)
        out1 = self.layer1(out)
        print("out1.shape")
        print(out1.shape)
        out2 = self.layer2(out1)
        print("out2.shape")
        print(out2.shape)
        out3 = self.layer3(out2)
        print("out3.shape")
        print(out3.shape)
        out4 = self.layer4(out3)
        print("out4.shape")
        print(out4.shape)
        out5 = self.layer5(out4)
        print("out5.shape")
        print(out5.shape)
        out6 = self.layer6(out5)
        print("out6.shape")
        print(out6.shape)
        out7 = self.layer7(out6)
        print("out7.shape")
        print(out7.shape)

        out8 = self.fclayer1(out7)
        out9 = self.fclayer2(out8)
        out10 = self.fclayer3(out9)

        return nn.Sigmoid()(out10)






# 创建对象
D = Discriminator()
G = Generator()
D = D.to(device)
G = G.to(device)

# 载入模型
# G.load_state_dict(torch.load('./generator_CGAN_z100.pth'))
# D.load_state_dict(torch.load('./discriminator_CGAN_z100.pth'))
#########判别器训练train#####################
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (yixing, liuchang,leigong) in enumerate(train_loader):

     for a in range(3):
        num_img = yixing.size(0)
        # label_onehot = torch.zeros((num_img,10)).to(device)
        # label_onehot[torch.arange(num_img),label]=1
        # view()函数作用把img变成[batch_size,channel_size,784]
        #img = img.view(num_img,  -1)  # 将图片展开为28*28=784
        yixing = yixing.view(num_img, 1, 128, 128).to(device)  # 将图片展开为28*28=784
        liuchang = liuchang.view(num_img, 2, 128, 128).to(device)  # 将图片展开为28*28=784

        real_label = torch.ones(num_img).to(device)  # 定义真实的图片label为1
        fake_label = torch.zeros(num_img).to(device)  # 定义假的图片的label为0
        # 开启异常检测模式
        torch.autograd.set_detect_anomaly(True)
        fake_img,mark = G(yixing,leigong)
        # 计算真实图片的损失
        fake_img_clone = fake_img.clone()
        fake_out = D(yixing,mark,fake_img_clone).view(-1)  # 判别器判断假的图片

        print()
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好

       # fake_img_clone = fake_img.clone()
        # fake_out = D(yixing, mark, fake_img_clone).view(-1)

        real_out = D(yixing,mark,liuchang).view(-1)
        # print(real_out.shape)
        # print(fake_label.shape)
        #real_out= real_out.view(-1,8)# 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # print(fake_label.shape)
        # # 计算假的图片的损失
  #      z = torch.randn(num_img, z_dimension).to(device)  # 随机生成一些噪声
  #       fake_img, mark = G(yixing, leigong)
  #       fake_out = D(yixing,mark,fake_img).view(-1)  # 判别器判断假的图片


        # print()
        # d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        # fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好



        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

    fake_img, mark = G(yixing, leigong)
    output = D(yixing,mark,fake_img).squeeze(1)  # 经过判别器得到的结果
    g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss

        # bp and optimize
    g_optimizer.zero_grad()  # 梯度归0
    g_loss.backward()  # 进行反向传播
    g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        # 打印中间的损失
        # try:
    if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.item(), g_loss.item(),
                torch.mean(real_scores).item(), torch.mean(fake_scores).item()  # 打印的是真实图片的损失均值
            ))
        # except BaseException as e:
        #     pass

    if epoch == 0:
            real_images = to_img(liuchang.cpu().data)
            save_image(real_images, './img_DCGAN/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img_DCGAN/fake_images-{}.png'.format(epoch + 1))
    # 保存模型
    torch.save(G.state_dict(), './generator_DCGAN.pth')
    torch.save(D.state_dict(), './discriminator_DCGAN.pth')

