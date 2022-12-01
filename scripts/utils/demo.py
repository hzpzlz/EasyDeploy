from .raw_io import load_mipi12bit, load_plainraw, load_mipi10bit
from .raw import bayer_to_offsets, rescaling, pack_raw_to_4ch
import numpy as np
import cv2

def demosaic_sample(rawim):
    """
        A sample to demosaic 4 channel raw in range [0, 1]/bayer pattern of RGBG to BGR.
        Args:
            rawim:
            bayer_pattern:

        Returns:
            rgbim:
        """
    bgr = np.zeros((rawim.shape[0], rawim.shape[1], 3))
    bgr[:, :, 0] = rawim[:, :, 0]
    bgr[:, :, 1] = (rawim[:, :, 1] + rawim[:, :, 3]) / 2
    bgr[:, :, 2] = rawim[:, :, 2]

    # bgr = bgr * 255
    # bgr = bgr.astype(np.uint8)
    return bgr

def mipi12_demo(path, png_path, bl, wl, h, w, bayerpattern, ratio = 1):
    """
        A sample to load mipiraw 12bit and demosaic it to png.
        Args:
            path: path of mipi12
            png_path: path of output png
            bl: blacklevel
            wl: whitelevel
            h: height of sensor
            w: width of sensor
            bayerpattern: bayer patter
            ratio: ratio of relight
        Returns:
        """
    mipi_12bit = load_mipi12bit(path, height=h, width=w, padding_len = 16)
    offset = bayer_to_offsets(bayerpattern)
    # bayer_patter of pack is 'rgbg' which is defined by function bayer_to_offsets
    pack = pack_raw_to_4ch(mipi_12bit, offset)
    rescale = rescaling(pack, bl, wl, clipping= 1)
    rescale = rescale * ratio
    bgr = demosaic_sample(rescale)
    cv2.imwrite(png_path, bgr)
    print('save mipi12 to ', png_path)


def ideal_demo(path, png_path, bl, wl, h, w, bayerpattern, ratio = 1):
    """
        A sample to load ideal raw and demosaic it to png.
        Args:
            path: path of ideal raw
            png_path: path of output png
            bl: blacklevel
            wl: whitelevel
            h: height of sensor
            w: width of sensor
            bayerpattern: bayer patter
            ratio: ratio of relight
        Returns:
        """
    idealraw = load_plainraw(path, height=h, width=w)
    offset = bayer_to_offsets(bayerpattern)
    # bayer_patter of pack is 'rgbg' which is defined by function bayer_to_offsets
    pack = pack_raw_to_4ch(idealraw, offset)
    rescale = rescaling(pack, bl, wl, clipping= 1)
    rescale = rescale * ratio
    bgr = demosaic_sample(rescale)
    cv2.imwrite(png_path, bgr)
    print('save ideal to ', png_path)


def mipi10_demo(path, png_path, bl, wl, h, w, bayerpattern, ratio = 1):
    """
        A sample to load mipiraw 10bit and demosaic it to png.
        Args:
            path: path of mipi12
            png_path: path of output png
            bl: blacklevel
            wl: whitelevel
            h: height of sensor
            w: width of sensor
            bayerpattern: bayer patter
            ratio: ratio of relight
        Returns:
        """
    mipi_10bit = load_mipi10bit(path, height=h, width=w, padding_len = 16)
    offset = bayer_to_offsets(bayerpattern)
    # bayer_patter of pack is 'rgbg' which is defined by function bayer_to_offsets
    pack = pack_raw_to_4ch(mipi_10bit, offset)
    rescale = rescaling(pack, bl, wl, clipping= 1)
    rescale = rescale * ratio
    bgr = demosaic_sample(rescale)
    cv2.imwrite(png_path, bgr)
    print('save mipi10 to ', png_path)


if __name__ == '__main__':
    h = 3072
    w = 4080
    bayer_pattern = 'gbrg'

    black_l_ideal = 1024
    white_l_ideal = 16383

    black_l_mipi10 = 64
    white_l_mipi10 = 1023

    black_l_mipi12 = 256
    white_l_mipi12 = 4095

    mipi_12bit = './raw_demo_data/mipi_12bit/IMG_20210125_195520-622_req[1]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotFormatConvertor.RAWMIPI12'
    ideal = './raw_demo_data/ideal/IMG_20210125_195520-667_req[1]_b[0]_BPS[0][2]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotFormatConvertor.RawPlain16LSB1'
    mipi_10bit = './raw_demo_data/mipi_10bit/IMG_20210525_173952-243_req[16]_b[0]_BPS[0][0]_w[6016]_h[4512]_s[6016]_ZSLSnapshotFormatConvertor.RAWMIPI10'


    mipi12_png = './raw_demo_data/mipi_12bit/mipi12.png'
    ideal_png = './raw_demo_data/ideal/ideal.png'
    mipi10_png = './raw_demo_data/mipi_10bit/mipi10.png'

    mipi12_demo(mipi_12bit, mipi12_png, black_l_mipi12, white_l_mipi12, h, w, bayer_pattern, ratio = 60)
    ideal_demo(ideal, ideal_png, black_l_ideal, white_l_ideal, h, w, bayer_pattern, ratio = 60)

    h = 4512
    w = 6016
    mipi10_demo(mipi_10bit, mipi10_png, black_l_mipi10, white_l_mipi10, h, w, bayer_pattern, ratio = 1)
