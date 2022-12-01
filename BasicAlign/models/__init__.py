import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'align':
        from .Align_model import AlignModel as M
    elif model == 'alignimg':
        from .Xaba import XabaModel as M
    elif model == 'rawalign':
        from .rawalign_model import RawAlignModel as M
    elif model == 'maskflownet':
        from .MaskFlowNet_model import MaskFlowNetModel as M
    elif model == 'udvd':
        from .UDVD_model import UDVDModel as M
    elif model == 'denoise':
        from .Denoise_model import DenoiseModel as M
    elif model == 'stn':
        from .STN_model import STNModel as M
    elif model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    #print("cccccccccccccccccccccccccc")
    return m
