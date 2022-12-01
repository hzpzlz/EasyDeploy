import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.raft.raft as raft
import models.archs.maskflownet.MaskFlownet as MaskFlownet
import models.archs.raft_hzp.raft_hzp as raft_hzp
import models.archs.pwcirr.IRR_PWC as irrpwc
import models.archs.pwcnet.pwcnet as pwcnet
import models.archs.raft_stn.raft_stn as raft_stn
import models.archs.udvd.UDVD as udvd
import models.archs.rmof.rmof as rmof
import models.archs.rmof.rmof_v1 as rmof_v1
import models.archs.maskflownet_hzp.maskflownet as MaskFlownet_hzp
import models.archs.Unet as Unet
import models.archs.STN.STN as STN
import models.archs.STN.STN_v1 as STN_v1
import models.archs.STN.STN_v2 as STN_v2
import models.archs.STN.STN_v3 as STN_v3
import models.archs.STN.STN_Unet as STN_Unet
import models.archs.STN.STN_Unet_v1 as STN_Unet_v1
import models.archs.STN.STN_Unet_v2 as STN_Unet_v2
import models.archs.STN.STN_Unet_v3 as STN_Unet_v3
import models.archs.xaba.xaba as XABA
import models.archs.xaba.xabaresd2s as XABARESD2S
import models.archs.xaba.xabav1 as XABAV1
import models.archs.xaba.xabaraw as XABARAW
import models.archs.xaba_align_denoise.xaba_align_denoise as XABA_AD

import models.archs.pwcplus.pwcplus as pwcplus
import models.archs.pwcplus.pwcnet as pwcnethty
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'RAFT':
        netG = raft.RAFT(opt)
    elif which_model == 'RMOF':
        netG = rmof.RMOF(opt)
    elif which_model == 'RMOF_v1':
        netG = rmof_v1.RMOF(opt)
    elif which_model == 'RAFT_HZP':
        netG = raft_hzp.RAFT(opt)
    elif which_model == 'RAFT_STN':
        netG = raft_stn.RAFT(opt)
    elif which_model == 'MaskFlownet_S':
        netG = MaskFlownet.MaskFlownet_S()
    elif which_model == 'PWCNet':
        netG = pwcnet.Net(output_level=opt_net['output_level'])
    elif which_model == 'MaskFlowNet_hzp':
        netG = MaskFlownet_hzp.MaskFlowNet()
    elif which_model == 'MaskFlowNet_hzp_v2':
        netG = MaskFlownet_hzp.MaskFlowNet_v2()
    elif which_model == 'MaskFlowNet_hzp_v3':
        netG = MaskFlownet_hzp.MaskFlowNet_v3(output_level=opt_net['output_level'])
    elif which_model == 'IRRPWC':
        netG = irrpwc.PWCNet()
    elif which_model == 'UDVD':
        netG = udvd.BlindVideoNet(channels_per_frame=opt_net['channels_per_frame'], out_channels=opt_net['out_channels'], bias=opt_net['bias'], blind=opt_net['blind'], sigma_known=opt_net['sigma_known'])
    elif which_model == 'Unet':
        netG = Unet.Unet(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN':
        netG = STN.STN(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_v1':
        netG = STN_v1.STN_v1(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_v2':
        netG = STN_v2.STN_v2(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_v3':
        netG = STN_v3.STN_v3(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_Unet':
        netG = STN_Unet.STN_Unet(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_Unet_v1':
        netG = STN_Unet_v1.STN_Unet_v1(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_Unet_v2':
        netG = STN_Unet_v2.STN_Unet_v2(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'STN_Unet_v3':
        netG = STN_Unet_v3.STN_Unet_v3(cpf=opt_net['channels_per_frame'], n_frame=opt['n_frames'], out_nc=opt_net['out_channels'])
    elif which_model == 'xaba':
        netG = XABA.Net()
    elif which_model == 'xabaresd2s':
        netG = XABARESD2S.Net()
    elif which_model == 'xabav1':
        netG = XABAV1.Net()
    elif which_model == 'xaba_ad':
        netG = XABA_AD.Net()
    elif which_model == 'xabaraw':
        netG = XABARAW.Net()
    elif which_model == 'xabarawlv3':
        netG = XABARAW.NetLV3()
    elif which_model == 'pwcplus':
        netG = pwcplus.PWCPlus()
    elif which_model == 'pwcnethty':
        netG = pwcnethty.PWCNet()
    elif which_model == 'PAN':
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'MSRResNet_PA':
        netG = SRResNet_arch.MSRResNet_PA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RCAN_PA':
        netG = RCAN_arch.RCAN_PA(n_resgroups=opt_net['n_resgroups'], n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'], res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'], rgb_range=opt_net['rgb_range'], scale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    #print(netG, "11111111111111")
    return netG
