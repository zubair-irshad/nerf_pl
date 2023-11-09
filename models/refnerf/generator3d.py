import torch
import torch.nn.functional as F

class Generator3D(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max, train_opacity_rgb):
        super(Generator3D, self).__init__()
        self.z_size = 128       # options.z_size
        self.bias = False       # options.bias
        self.voxel_size = 64  # options.voxel_size
        padd = (1, 1, 1)
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.train_opacity_rgb = train_opacity_rgb
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.z_size, self.voxel_size * 16, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.BatchNorm3d(self.voxel_size * 16),
            torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 16, self.voxel_size *8, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 8),
            torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 8, self.voxel_size * 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 4),
            torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 4, self.voxel_size* 2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size * 2),
            torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size * 2, self.voxel_size, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.voxel_size),
            torch.nn.ReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.voxel_size, int(self.voxel_size/2), kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(int(self.voxel_size/2)),
            torch.nn.ReLU())
        # self.layer7 = torch.nn.Sequential(
        #     torch.nn.ConvTranspose3d(int(self.voxel_size/2), 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
        #     torch.nn.Sigmoid())
        if self.train_opacity_rgb:
            self.layer7 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(int(self.voxel_size/2), 4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)))
        else:    
            self.layer7 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(int(self.voxel_size/2), 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)))

    def forward(self, x, z):
        out = z.view(-1, self.z_size, 1, 1, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        # extract density from voxels using trilinear interpolation
        x = x.reshape(1, -1, 1, 1, 3)
        self.xyz_min = self.xyz_min.to(x.device)
        self.xyz_max = self.xyz_max.to(x.device)
        ind_norm = (
            (x - self.xyz_min)
            / (self.xyz_max - self.xyz_min)
        ).flip((-1,)) * 2 - 1
        out = F.grid_sample(out, ind_norm.float(),
                                padding_mode="border", align_corners=False)
        return out