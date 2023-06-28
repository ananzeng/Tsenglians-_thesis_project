import torch
from torch import nn

import torch.nn.functional as F
from utils.imutils import Normalize

class PSA_Spatial_only_branch(nn.Module):
    """
    https://arxiv.org/pdf/2107.00782.pdf
    https://mp.weixin.qq.com/s/KwU3rj1kF4UemmtIXsrcng
    """
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        return spatial_out
    
class PSA_Channel_only_branch(nn.Module):
    """
    https://arxiv.org/pdf/2107.00782.pdf
    https://mp.weixin.qq.com/s/KwU3rj1kF4UemmtIXsrcng
    """
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out
    
class GAM_Attention(nn.Module):  
    """
    https://mp.weixin.qq.com/s/vPr-pD5B2s5lgNOsUbySqQ
    https://github.com/northBeggar/Plug-and-Play/blob/main/GAM%20attention.py
    https://arxiv.org/pdf/2112.05561v1.pdf
    """
    
    def __init__(self, in_channels, out_channels, rate=4):  
        super(GAM_Attention, self).__init__()  

        self.channel_attention = nn.Sequential(  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )  
      
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(out_channels)  
        )  
      
    def forward(self, x):  
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  
      
        x = x * x_channel_att  
      
        x_spatial_att = self.spatial_attention(x).sigmoid()  
        out = x * x_spatial_att  
      
        return out 
    
class nam_module(nn.Module):
    """
    https://mp.weixin.qq.com/s/06v0XeDd6VTVtK_roc3KWg
    https://github.com/Christian-lyc/NAM
    https://arxiv.org/abs/2111.12419
    """
    
    def __init__(self, channels, t=16):
        super(nam_module, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual #

        return x
    
class simam_module(nn.Module):
    """
    https://github.com/ZjjConan/SimAM
    https://mp.weixin.qq.com/s/8kn3zAygVKdfnaOSm3AMzg
    https://proceedings.mlr.press/v139/yang21o.html
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
class CA_Block(nn.Module):
    """
    https://github.com/houqb/CoordAttention
    https://arxiv.org/abs/2103.02907
    """
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out
    
class ECA_layer(nn.Module):
    """Constructs a ECA module.
    https://arxiv.org/abs/1910.03151
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
class PAM_Module(nn.Module):
    
    """ Position attention module
    https://arxiv.org/abs/1809.02983
    """

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
    
class CAM_Module(nn.Module):
    """ Channel attention module
    https://arxiv.org/abs/1809.02983
    """

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
    
class SALayer(nn.Module):
    def __init__(self, channels):
        super(SALayer,self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w  = x.size()
        att = self.conv(x)
        att = self.sigmoid(att)
        att = att.view(b, 1, h, w)
        return x * att.expand_as(x)    

    
class SELayer(nn.Module):
    """
    https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        #self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        out = x * y.expand_as(x) #defult output
        #out = self.gamma * out + x
        return out
    
class DRS(nn.Module):
    """ 
    DRS non-learnable setting
    hyperparameter O , additional training paramters X
    """
    def __init__(self, delta):
        super(DRS, self).__init__()
        #self.relu = nn.ReLU()
        self.delta = delta
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        #x = self.relu(x) #容軒有註解?
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.delta
        
        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
        
        return x
    
class DRS_learnable(nn.Module):
    """ 
    DRS learnable setting
    hyperparameter X , additional training paramters O 
    """
    def __init__(self, channel):
        super(DRS_learnable, self).__init__()
        # self.relu = nn.ReLU()
        
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()

        # x = self.relu(x)
        
        """ 1: max extractor """
        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        """ 2: suppression controller"""
        control = self.global_avg_pool(x).view(b, c)
        control = self.fc(control).view(b, c, 1, 1)
        control = control.expand_as(x)

        """ 3: suppressor"""
        x = torch.min(x, x_max * control)
            
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()
        
        self.se_layer = SELayer(in_channels*2, 16)
        #self.sa_layer = SALayer(in_channels*2)
        #self.cam_layer = CAM_Module(in_channels*2)
        #self.pam_layer = PAM_Module(in_channels*2)
        #self.eca_layer = ECA_layer(in_channels*2)
        #self.CA_Block = CA_Block(1024, h=56, w=56)
        #self.simam_module = simam_module(in_channels*2)
        #self.nam_module = nam_module(in_channels*2)
        #self.GAM_Attention = GAM_Attention(in_channels*2, in_channels*2)
        #self.PSA_Channel_only_branch = PSA_Channel_only_branch(in_channels*2)
        #self.PSA_Spatial_only_branch = PSA_Spatial_only_branch(in_channels*2)
        
        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)
        
        #修改區
        branch2 = self.se_layer(branch2) #b6, b7都有加_70.384%(0.35)
        #branch2 = self.sa_layer(branch2) #b6, b7都有加_68.374%(0.35)
        #branch2 = self.cam_layer(branch2) #b6, b7都有加_69.926%(0.35)
        #branch2 = self.pam_layer(branch2) #b6, b7都有加_69.952%(0.35)
        #branch2 = self.eca_layer(branch2) #b6, b7都有加_70.185%(0.35)
        #branch2 = self.CA_Block(branch2) #NOT WORKING
        #branch2 = self.simam_module(branch2) #b6, b7都有加_69.607%(0.35)
        #branch2 = self.nam_module(branch2) #b6, b7都有加_70.083%(0.35)
        #branch2 = self.GAM_Attention(branch2) #PSA_Channel_only_branch
        #branch2 = self.PSA_Channel_only_branch(branch2)
        #branch2 = self.PSA_Spatial_only_branch(branch2)
        
        #branch2_se_layer = self.se_layer(branch2) 
        #branch2_sa_layer = self.sa_layer(branch2) 
        #branch2_cam_layer = self.cam_layer(branch2) 
        #branch2_pam_layer = self.pam_layer(branch2) 
        #branch2_PSA_Channel_only_branch = self.PSA_Channel_only_branch(branch2)
        #branch2_PSA_Spatial_only_branch = self.PSA_Spatial_only_branch(branch2)
        #branch2 = branch2_se_layer + branch2_sa_layer #b6, b7都有加_69.913%(0.35)
        #branch2 = branch2_cam_layer + branch2_pam_layer #b6, b7都有加_69.149%(0.35)
        #branch2 = branch2_PSA_Channel_only_branch + branch2_PSA_Spatial_only_branch #
        
        x = branch1 + branch2
        #x = self.se_layer(x)
        
        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.DRS = DRS(0.55)
        self.DRS_learnable_layer6 = DRS_learnable(2048)
        self.DRS_learnable_layer7 = DRS_learnable(4096)
        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        
        #x = self.DRS(x) 2
        
        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
         
        #x = self.DRS(x) 3
        
        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        
        #x = self.DRS(x) 4
        
        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)
        
        #x = self.DRS(x)# 5
        
        x, conv5 = self.b6(x, get_x_bn_relu=True)
        x = self.DRS_learnable_layer6(x) #6
        #x = self.DRS(x) #6
        
        x = self.b7(x)
        x = self.DRS_learnable_layer7(x) #7
        #x = self.DRS(x) #7
                
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict
