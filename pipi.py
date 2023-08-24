import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
# 继承Data.Dataset并对__init__和__getitem__重载
# __getitem__采用sitk读取nii图像转为array
class Dataset(Data.Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        # 读取图像的尺寸大小为：(1, 160, 192, 160)
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        return img_arr

    # 读入fixed图像
    f_img = sitk.ReadImage(args.atlas_file)  # 读取图像，图像尺寸为[160, 192, 160] 对应为(x,y,z)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]  # 插入两个新维度(1, 1, 160, 192, 160)
    # 由于sitk的原因，图像X轴与Z轴发生了对调，输出形状为：(160, 192, 160)对应为(z,y,x)
    vol_size = input_fixed.shape[2:]  # 获取图像的尺寸
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)  # 重复图像的数组，重复次数为batchsize
    input_fixed = torch.from_numpy(input_fixed).to(
        device).float()  # 将图像数组转为tensor并加载到GPU中    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))

    datas = Dataset(files=train_files)  # 定义dataset
    DataLoad = Data.DataLoader(datas, batch_size=args.batch_size, shuffle=True, num_workers=4,
                               drop_last=True)  # 定义dataloader
    for i in range(1, args.n_iter + 1):
        # 迭代获取图像，每次获取batchsize张图像
        input_moving = iter(DataLoad).next()
        # 载入的尺寸为   [batchisize, 1, 160, 192, 160]
        input_moving = input_moving.to(device).float()


enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 16, 16]
class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn                                               # bn标准化
        self.dim = dim                                             # 特征图
        self.enc_nf = enc_nf                                       # encoding
        self.vm2 = len(dec_nf) == 7
        # Encoder部分
        # input的尺寸    （Batchsize，in_channel,z,y,x）(batchsize, 1, 160, 192, 160)
        # output的尺寸   （Batchsize，out_channel,z_o,y_o,x_o）(batchsize, 16, z_o,y_o,x_o)
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            if i == 0:
            # encoding部分每一层会进行下采样，图像尺寸会变为上一层的一半2
            # 第一层将moving和fixed拼接，1通道变2通道  Conv3d(2, 16, kernel_size=(3, 3, 3)) 尺寸：160, 192, 160
                prev_nf = 2
                self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 3, 1, batchnorm=bn))
            else:
            # 第二层 Conv3d(16, 32, kernel_size=(3, 3, 3))   尺寸：80, 96, 80
            # 第三层 Conv3d(32, 32, kernel_size=(3, 3, 3))   尺寸：40, 48, 40
            # 第四层 Conv3d(32, 32, kernel_size=(3, 3, 3))   尺寸：20, 24, 20
            # 第五层 Conv3d(32, 32, kernel_size=(3, 3, 3))   尺寸：10, 12, 10
                prev_nf = enc_nf[i - 1]
                self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 3, 2, batchnorm=bn))
        self.enc.append(self.conv_block(dim, enc_nf[-1], enc_nf[-1], 3, 2, batchnorm=bn))
        # Decoder函数
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))   # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn)) # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn)) # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] * 2, dec_nf[3], batchnorm=bn))# 4
        self.dec.append(self.conv_block(dim, dec_nf[3] + enc_nf[0], dec_nf[4], batchnorm=bn)) # 5
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[4], dec_nf[5], batchnorm=bn) # 6
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


 def forward(self, src, tgt): #input_moving, input_fixed
        x = torch.cat([src, tgt], dim=1) # 合并input_moving, input_fixed (1,2,160,192,160)
        # 编码部分
        x_enc = [x]
        for i, l in enumerate(self.enc):  #enc = [16, 32, 32, 32]
            x = l(x_enc[-1])
            print(i,x.shape)
            x_enc.append(x)  # 经过4层的encoding
        y = x_enc[-1]
        # 解码部分，每一层会进行拼接和上采样
        for i in range(3):
            # 卷积
            y = self.dec[i](y)
            # 上采样
            y = self.upsample(y)
            # 拼接
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        y = self.dec[3](y)
        y = self.upsample(y)
        y = torch.cat([y, x_enc[1]], dim=1)
        y = self.dec[4](y)
        if self.vm2:
            y = self.vm2_conv(y)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        vectors = [torch.arange(0, s) for s in size] #(160, 192, 160)
        # 创建一个三维的张量
        # vectors = [[0,1,2,...,160]
        #            [0,1,2,...,192]
        #            [0,1,2,...,160]]

        grids = torch.meshgrid(vectors) # 生成网格(坐标)，一个元组，包含三个张量，
        # 以便获取图像对应位置的坐标，这三个张量分别是三个坐标轴，刚好对应我们熟悉的x，y，z轴的坐标定位。
        grid = torch.stack(grids)  #  [C D H W] (3, 160, 192, 160) 扩张成一个三维张量
        grid = torch.unsqueeze(grid, 0)  # 增加batchsize维度 (1, 3, 160, 192, 160)

        grid = grid.type(torch.FloatTensor) # 将网格数据变换为浮点型
        self.register_buffer('grid', grid)  # 将grid写入内存且不随优化器优化而改变

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow # 网格加上一个位移（网格加上形变场）
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            # 标准化为[-1,1]

        if len(shape) == 2:
            # 二维情况
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            # 三维情况
            #[B C D H W] (1, 3, 160, 192, 160)
            new_locs = new_locs.permute(0, 2, 3, 4, 1) # 维度换位
            #[B D H W C] (1, 160, 192, 160, 3)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
        # grid_sample是一个采样函数，提供一个input的Tensor以及一个对应的网格grid，然后根据grid中每个位置
        # 提供的坐标信息(这里指input中像素的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出


# 平滑损失
def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0
# 均方误差损失
def mse_loss(x, y):
    return torch.mean((x - y) ** 2)
# ncc损失
def ncc_loss(I, J, win=None):
    # 图像维度
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    # Cov(X,Y) = E[(X-E(X))(Y-E(Y))]
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    # 方差部分
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross

def train():
    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec =[32, 32, 32, 32, 16, 16]
    # 定义Unet对象
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    Resume = False
    # Resume = False 重新训练；True为加载以前训练过的模型
    if Resume:
        path_checkpoint = 'Checkpoint/7300.pth'
        checkpoint = torch.load(path_checkpoint)
        UNet.load_state_dict(checkpoint)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()
    # 设置优化器和损失函数
    opt = Adam(UNet.parameters(), lr=args.lr)
    # 相似损失函数
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    for i in range(1, args.n_iter + 1):
        # fixed图像和训练图像放进UNet中，返回形变场
        flow_m2f = UNet(input_moving, input_fixed)
        # 训练图像和形变场放进STN中，返回配准后图像
        m2f = STN(input_moving, flow_m2f) #（待配准图像，形变场） retuen 配准后图像
        # 计算损失
        sim_loss = sim_loss_fn(m2f, input_fixed)   #（配准后图像，标准图像）
        grad_loss = grad_loss_fn(flow_m2f)  #计算形变场的平滑损失
        loss = sim_loss + args.alpha * grad_loss
        # 反向传播
        opt.zero_grad()
        loss.backward()

# 计算dice系数
def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        # 判断等于标签值的像素位置为True，否为Fasle，得到两个矩阵内元素为True或Fasle
        # gt = [True,True,Fasle,..., True
        #       Fasle,False,True,...,Fasle
        #       Fasle,True,True,...,Fasle]
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def test():
    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # 读取fixed图像
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    # 加载模型
    UNet.load_state_dict(torch.load(args.checkpoint_path))
    # 用于生成配准图像的STN
    STN_img = SpatialTransformer(vol_size).to(device)
    # 用于生成标签图像的STN，最近插值
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()
    DSC = []
    # 读取fixed图像对应的label
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))
    for file in test_file_lst:
        name = os.path.split(file)[1]
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        # 读入moving图像对应的label
        label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
        input_label = torch.from_numpy(input_label).to(device).float()
        # 获得配准后的图像和label
        pred_flow = UNet(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(input_label, pred_flow)
        # 计算DSC
        dice = compute_label_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
