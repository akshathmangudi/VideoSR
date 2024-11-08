

def define_generator(opt):
    net_G_opt = opt['model']['generator']

    if net_G_opt['name'].lower() == 'frnet':  # frame-recurrent generator
        from tecogan import FRNet
        net_G = FRNet(
            in_nc=net_G_opt['in_nc'],
            out_nc=net_G_opt['out_nc'],
            nf=net_G_opt['nf'],
            nb=net_G_opt['nb'],
            degradation=opt['dataset']['degradation']['type'],
            scale=opt['scale'])

    elif net_G_opt['name'].lower() == 'egvsr':  # efficient GAN-based generator
        from egvsr import FRNet
        net_G = FRNet(
            in_nc=net_G_opt['in_nc'],
            out_nc=net_G_opt['out_nc'],
            nf=net_G_opt['nf'],
            nb=net_G_opt['nb'],
            degradation=opt['dataset']['degradation']['type'],
            scale=opt['scale'])

    elif net_G_opt['name'].lower() == 'espnet':  # ESPCN generator
        from espcn import ESPNet
        net_G = ESPNet(scale=opt['scale'])

    elif net_G_opt['name'].lower() == 'vespnet':  # VESPCN generator
        from vespcn import VESPNet
        net_G = VESPNet(
            scale=opt['scale'], channel=net_G_opt['channel'], depth=net_G_opt['depth'])

    elif net_G_opt['name'].lower() == 'sofnet':  # SOFVSR generator
        from sofvsr import SOFNet
        net_G = SOFNet(scale=opt['scale'])
    elif net_G_opt['name'].lower() == 'custom':
        from models.custom import CustomGenerator
        net_G = CustomGenerator(in_nc=net_G_opt['in_nc'])

    else:
        raise ValueError('Unrecognized generator: {}'.format(
            net_G_opt['name']))

    return net_G


def define_discriminator(opt):
    net_D_opt = opt['model']['discriminator']

    if opt['dataset']['degradation']['type'] == 'BD':
        spatial_size = opt['dataset']['train']['crop_size']
    else:  # BI
        spatial_size = opt['dataset']['train']['gt_crop_size']

    if net_D_opt['name'].lower() == 'stnet':  # spatio-temporal discriminator
        from .tecogan import SpatioTemporalDiscriminator
        net_D = SpatioTemporalDiscriminator(
            in_nc=net_D_opt['in_nc'],
            spatial_size=spatial_size,
            tempo_range=net_D_opt['tempo_range'])

    elif net_D_opt['name'].lower() == 'snet':  # spatial discriminator
        from .tecogan import SpatialDiscriminator
        net_D = SpatialDiscriminator(
            in_nc=net_D_opt['in_nc'],
            spatial_size=spatial_size,
            use_cond=net_D_opt['use_cond'])

    else:
        raise ValueError('Unrecognized discriminator: {}'.format(
            net_D_opt['name']))

    return net_D
