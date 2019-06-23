import torch
print(torch.__version__)
print(torch.cuda.is_available())

%matplotlib inline
%reload_ext autoreload
%autoreload 2

from abc import ABC, abstractmethod
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.nn.utils.spectral_norm import spectral_norm
import torchvision.models as models
from torch import nn
from torch.nn import functional as F
import numpy as np
from skimage import io, transform, color
from matplotlib import pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import time

parser = {
  'loadSize': 176,
  'fineSize': 128,
  'seed': 123,
  'batchSize': 100,
  'testBatchSize': 70,
  'nEpochs': 170,
  'threads': 4,
  'lr': 0.000005
}

def rgb2gray(rgb_imgs):
    return color.rgb2gray(rgb_imgs)

def cut_model(m, arr):
    cut, lr_cut = arr
    return list(m.children())[:cut] if cut else [m]

def imshow_img(img):
    img = np.rollaxis(img, 0, 3)
    plt.imshow(img)
    plt.show()

def imshow(bw_img, out_img, in_img):
    out_img = np.rollaxis(out_img, 0, 3)
    in_img = np.rollaxis(in_img, 0, 3)
    bw_img = np.rollaxis(bw_img, 0, 3)
    fig = plt.figure()
    
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(bw_img)
    plt.axis('off')
    a.set_title('Gray')
    
    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(out_img)
    plt.axis('off')
    a.set_title('Ground Truth')
    
    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(in_img)
    plt.axis('off')
    a.set_title('Colorized')
    plt.show()

def preprocessing(data):
    data = data.permute(0, 2, 3, 1)
    
    N, H, W, C = data.shape
    gray_imgs = np.zeros((N, 3, H, W))
    gray_ch = rgb2gray(data.cpu().numpy())
    gray_imgs[:, 0, :, :] = gray_ch
    gray_imgs[:, 1, :, :] = gray_ch
    gray_imgs[:, 2, :, :] = gray_ch
    return gray_imgs

def get_samplers(dataset_size):
    indices = list(range(dataset_size))
    validation_split = .0
    shuffle_dataset = True
    random_seed = 42
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def load_data(opt):
    datapath = 'ResizedData/'

    dataset = torchvision.datasets.ImageFolder(datapath,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice(
                                                       [transforms.Resize(opt['loadSize'], interpolation=1),
                                                        transforms.Resize(opt['loadSize'], interpolation=2),
                                                        transforms.Resize(opt['loadSize'], interpolation=3),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']), interpolation=1),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']), interpolation=2),
                                                        transforms.Resize((opt['loadSize'], opt['loadSize']), interpolation=3)]),
                                                   transforms.RandomChoice(
                                                       [transforms.RandomResizedCrop(opt['fineSize'], interpolation=1),
                                                        transforms.RandomResizedCrop(opt['fineSize'], interpolation=2),
                                                        transforms.RandomResizedCrop(opt['fineSize'], interpolation=3)]),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor()]))
    return dataset

class ConvBlock(nn.Module):
    def __init__(self, ni:int, no:int, ks:int=3, stride:int=1, pad:int=None, bn:bool=True, bias:bool=True):
        super().__init__()   
        if pad is None: pad = ks//2//stride

        # he always uses spectral norm, and doesn't use actn after conv
        layers = [spectral_norm(nn.Conv2d(ni, no, ks, stride, padding=pad, bias=bias))]
        
        if bn:
            layers.append(nn.BatchNorm2d(no))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class UpSampleBlock(nn.Module):
    # bn=True

    def __init__(self, ni:int, nf:int, ks:int=3):
        super().__init__()
        layers = []
        # ConvBlock here's always called with these params
        #    -> Pad=1 and kernel=3, so it results same size of image with different size of channels (nf*4)
        #       bn=True, applying batch norm to ConvBlock
        layers += [ConvBlock(ni, nf*4, ks=ks, bn=True), 
                   nn.PixelShuffle(2), 
                   nn.BatchNorm2d(nf)]
            
        self.sequence = nn.Sequential(*layers)
        self._icnr_init() # I don't know what it does!!!!!!!
        
    def _icnr_init(self):
        conv_shuffle = self.sequence[0].seq[0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel) ## supply the conv with initial weights

    @staticmethod
    def _icnr(x:torch.Tensor, scale:int=2):
        init = nn.init.kaiming_normal_
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel) # elly fhemto anha method to initialize weights of conv w enha metgaba mn training on imageNet data (from pytorch doc)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel
    
    def forward(self, x):
        return self.sequence(x)

class UnetBlock(nn.Module):
    def __init__(self, up_in:int, x_in:int, n_out:int, self_attention:bool=False):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = ConvBlock(x_in,  x_out,  ks=1, bn=False)
        self.tr_conv = UpSampleBlock(up_in, up_out)
        self.relu = nn.ReLU(inplace=True)
        
        out_layers = []
        out_layers.append(nn.BatchNorm2d(n_out))
        
        if self_attention:
            out_layers.append(SelfAttention(n_out))
            
        self.out = nn.Sequential(*out_layers)
        
        
    def forward(self, up_p:int, x_p:int):
        up_p = self.tr_conv(up_p) # doubles the size of image, changes the size of channels
        x_p = self.x_conv(x_p) # same size, changes the size of channels
        x = torch.cat([up_p, x_p], dim=1)
        x = self.relu(x)
        return self.out(x)

class SaveFeatures():
    features=None
    def __init__(self, m:nn.Module): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def remove(self): 
        self.hook.remove()

# Don't understand it!!!!!!!!!!!!!!!!
class SelfAttention(nn.Module):
    def __init__(self, in_channel:int, gain:int=1):
        super().__init__()
        self.query = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.key = self._spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),gain=gain)
        self.value = self._spectral_init(nn.Conv1d(in_channel, in_channel, 1), gain=gain)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def _spectral_init(self, module:nn.Module, gain:int=1):
        nn.init.kaiming_uniform_(module.weight, gain)
        if module.bias is not None:
            module.bias.data.zero_()

        return spectral_norm(module)

    def forward(self, input:torch.Tensor):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input
        return out

class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20,70,10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        self.m_feat.requires_grad = False
        
        vgg_arr = list(self.m_feat.children())
        self.part1 = vgg_arr[:22]
        self.part1 = nn.Sequential(*self.part1)
        self.relu1 = vgg_arr[22]
        
        self.part2 = vgg_arr[23:32]
        self.part2 = nn.Sequential(*self.part2)
        self.relu2 = vgg_arr[32]
        
        self.part3 = vgg_arr[33:42]
        self.part3 = nn.Sequential(*self.part3)
        self.relu3 = vgg_arr[42]
        
        self.part4 = vgg_arr[43:]
        self.part4 = nn.Sequential(*self.part4)
        
        
        self.wgts = layer_wgts
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        
        x = self.part1(x)
        relu1_out = self.relu1(x)
        x = self.part2(relu1_out)
        relu2_out = self.relu2(x)
        x = self.part3(relu2_out)
        relu3_out = self.relu3(x)
        x = self.part4(relu3_out)
        
        # hook is a way for watching the activation layers and here he's watching layers 22, 32, 42 (Relu)
        if clone:
            return [relu1_out.clone(), relu2_out.clone(), relu3_out.clone()]
        else:
            return [relu1_out, relu2_out, relu3_out]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        # gets the loss of the output of the three layers and
        # gives each of them different weight
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)] 
        
        return sum(self.feat_losses)
    

class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_trainable(self, trainable:bool):
        set_trainable(self, trainable)

    @abstractmethod
    def forward(self, x_in:torch.Tensor, max_render_sz:int=400):
        pass

    def get_device(self):
        return next(self.parameters()).device


class Unet34(GeneratorModule): 
    def __init__(self, nf_factor:int=1):
        super().__init__()
        
        self.encoder = torchvision.models.resnet34(pretrained=True)
        
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)
        self.conv2 = nn.Sequential(self.encoder.maxpool, 
                                   self.encoder.layer1)
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        
        self.relu = nn.ReLU()
        self.up1 = UnetBlock(512,           256, 512*nf_factor)
        self.up2 = UnetBlock(512*nf_factor, 128, 512*nf_factor)
        self.up3 = UnetBlock(512*nf_factor,  64, 512*nf_factor, self_attention=True)
        self.up4 = UnetBlock(512*nf_factor,  64, 256*nf_factor)
        
        self.up5 = UpSampleBlock(256*nf_factor, 32*nf_factor)
        self.out= nn.Sequential(ConvBlock(32*nf_factor, 3, ks=3, bn=False), nn.Tanh())

    def forward(self, x: torch.Tensor):
        # Encoding Part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        # Decoding Part
        x = self.relu(conv5)
        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)
        x = self.up5(x)
        x = self.out(x)
        return x

class GenResult():
    def __init__(self, gcost, gaddlloss):
        self.gcost=gcost
        self.gaddlloss=gaddlloss

class DCCritic(nn.Module):
    def __init__(self,nf:int=256):
        super(DCCritic, self).__init__()
        use_bias = True
        #   Convolution Block #1
        self.initial = nn.Sequential(
             nn.Sequential(
                spectral_norm(nn.Conv2d(3, nf, kernel_size=4, padding=1,stride=2, bias=use_bias)),
                nn.LeakyReLU(0.2, inplace=True)),
            nn.Dropout2d(0.2),
            #   Convolution Block #2
             nn.Sequential(
                spectral_norm(nn.Conv2d(nf, nf, kernel_size=3, padding=1, stride=1, bias=use_bias)),
                 nn.BatchNorm2d(nf),

                nn.LeakyReLU(0.2, inplace=True)
            )   
        )
        self.mid = nn.Sequential(
             nn.Dropout2d(.5),

        #   Convolution Block #3
            nn.Sequential(
                spectral_norm(nn.Conv2d(nf, nf*2, kernel_size=4, padding=1,stride=2, bias=use_bias)),
                nn.BatchNorm2d(nf*2),

                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(nf*2,1)),
            nn.Dropout2d(.5),
            #   Convolution Block #4
             nn.Sequential(
               
                spectral_norm(nn.Conv2d(nf*2, nf*4, kernel_size=4, padding=1,stride=2, bias=use_bias)),
                nn.BatchNorm2d(nf*4),

                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(nf*4,1)

            ),

            nn.Dropout2d(.5),
            #   Convolution Block #5
             nn.Sequential(
                spectral_norm(nn.Conv2d(nf*4, nf*8, kernel_size=4, padding=1,stride=2, bias=use_bias)),
                             nn.BatchNorm2d(nf*8),

                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(nf*8,1)

            )
        )
        self.out = nn.Sequential(
        #   Convolution Block #6
           nn.Sequential(
              spectral_norm(nn.Conv2d(nf*8, 1, kernel_size=4, padding=0,stride=1, bias=False))
          )
        )



    def get_layer_groups(self)->[]:
        return children(self)

    def forward(self, input):
        x = self.initial(input)
        x = self.mid(x)
        return self.out(x), x



class CriticResult():
    def __init__(self, hingeloss, dreal, dfake, dcost):
        self.hingeloss=hingeloss
        self.dreal=dreal
        self.dfake=dfake
        self.dcost=dcost



opt = parser
generator = Unet34(nf_factor=2).cuda()
discriminator = DCCritic(nf=256).cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt['lr'],betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt['lr'],betas=(0.5, 0.999))
epoch_begin = 0

lol = torch.load("Models/model_epoch_disc_128_13.pth")
discriminator.load_state_dict(lol['model_state_dict'])
optimizer_D.load_state_dict(lol['optimizer_state_dict'])

lol = torch.load("Models/model_epoch_gen_128_13.pth")
generator.load_state_dict(lol['model_state_dict'])
optimizer_G.load_state_dict(lol['optimizer_state_dict'])
epoch_begin = lol['epoch']

for param_group in optimizer_D.param_groups:
        param_group['lr'] = param_group['lr']/10

for param_group in optimizer_G.param_groups:
        param_group['lr'] = param_group['lr']/20

generator.train()
discriminator.train()

fl = FeatureLoss()

def train_discriminator(orig_image:torch.Tensor, real_image:torch.Tensor) -> CriticResult:                     
    optimizer_D.zero_grad() 
    fake_image = generator(orig_image)
    
    dfake_raw,_ = discriminator(fake_image)
    dfake = torch.nn.ReLU()(1.0 + dfake_raw).mean()
    
    dreal_raw,_ = discriminator(real_image)
    dreal = torch.nn.ReLU()(1.0 - dreal_raw).mean()
    
    discriminator.zero_grad()     
    hingeloss = dfake + dreal
    hingeloss.backward()
    optimizer_D.step()
    
    return hingeloss.item()

def train_generator(orig_image:torch.Tensor, real_image:torch.Tensor) -> GenResult:
    optimizer_G.zero_grad()         
    fake_image = generator(orig_image)
    
    dres, img = discriminator(fake_image)
    mn = torch.mean(dres)
    d_loss = -1 * mn
    added_loss = fl(fake_image, real_image)
    total_loss = d_loss + added_loss

    total_loss.backward()
    optimizer_G.step()

    return total_loss.item()


# FeatureLoss VGG16
fl = FeatureLoss()

def train(training_data_loader):
    gen_loss = 0
    disc_loss = 0
    sec_batch = time.time()
    seconds = time.time()
    batches_lens = len(training_data_loader)
    for iteration, data in enumerate(training_data_loader, 1):
        if iteration % 100 == 0:
            seconds_beg = time.time()
            print ("Batch Data Load Time", seconds_beg - seconds)
        real_image = data[0].cuda()
        gray_scale = preprocessing(data[0])
        orig_image = torch.from_numpy(gray_scale).float().cuda()

        if iteration % 100 == 0:
            seconds_preprocessing = time.time()
            print ("Preprocessing Time:", seconds_preprocessing - seconds_beg)
        disc_loss += train_discriminator(orig_image, real_image)
        
        if iteration % 100 == 0:
            seconds_disc = time.time()
            print ("Discriminator Time:", seconds_disc - seconds_preprocessing)
        
        gen_loss += train_generator(orig_image, real_image)
        
        if iteration % 100 == 0:
            seconds_gen = time.time()
            print ("Generator Time:", seconds_gen - seconds_disc)
        
        print (f'[{iteration}/{batches_lens}] ==>', "Total Time:", time.time() - seconds)
        print ("---------------------------------------")
        seconds = time.time()
    print ("Batch Time:", time.time() - sec_batch)
    gen_total_loss = gen_loss / batches_lens
    disc_total_loss = disc_loss / batches_lens
    print("===> Avg. genloss: {:.4f} dB".format(gen_total_loss))
    print("===> Avg. discloss: {:.4f} dB".format(disc_total_loss))
    return gen_total_loss, disc_total_loss

from math import log10
def test(testing_data_loader):
    avg_psnr = 0
    criterion2 = nn.MSELoss()
    with torch.no_grad():
        i = 0
        for iteration, data in enumerate(testing_data_loader, 1):
            real_image = data[0].cuda()
            gray_scale = preprocessing(data[0])
            ch = torch.from_numpy(gray_scale)
            orig_image = torch.cat((ch, ch, ch), 1).float().cuda()
            
            optimizer_G.zero_grad()
            output = generator(orig_image)
            mse = criterion2(output.cuda(), real_image)
            if i%10 == 0:
                imshow(orig_image[0].data.cpu().numpy(), real_image[0].data.cpu().numpy(), output[0].data.cpu().numpy())
            
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
        i += 1
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def checkpoint(epoch, gen_loss, disc_loss):

    model_out_gen = "Models/model_epoch_gen_128_{}.pth".format(epoch)
    model_out_disc = "Models/model_epoch_disc_128_{}.pth".format(epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'loss': gen_loss
    }, model_out_gen)
    torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            'loss': disc_loss
    }, model_out_disc)

    print("Checkpoint saved to {}".format(model_out_disc))


data = load_data(opt)

dataset_size = len(data)
print(dataset_size)

print('===> Loading datasets')
train_sampler, valid_sampler = get_samplers(dataset_size)

training_data_loader = DataLoader(dataset=data, batch_size=opt['batchSize'], sampler=train_sampler, num_workers=9)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=13, gamma=0.1)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=13, gamma=0.1)

sec_epoch = time.time()
for epoch in range(epoch_begin+1, opt['nEpochs']):
    print(f'Epoch: {epoch}\n')
    gen_loss, disc_loss = train(training_data_loader)
    scheduler_G.step()
    scheduler_D.step()
    checkpoint(epoch, gen_loss, disc_loss)
    print ("Epoch Time:", time.time() - sec_epoch)
