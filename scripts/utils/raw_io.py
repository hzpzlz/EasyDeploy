import numpy as np
import math
import rawpy
import h5py
from PIL import Image
import csv
import os

def load_raw(raw_path):
    """
    Load raw image upon rawpy.
    Args:
        raw_path:

    Returns:
        rawim:
        meta: dict with 'black_level'(RGrBGb), 'white_level' and 'bayer_pattern'
    """
    meta = {}
    with rawpy.imread(raw_path) as raw:
        color_desc = bytes.decode(raw.color_desc).lower()
        raw_pattern = np.asarray(raw.raw_pattern).flatten()

        if color_desc != 'rgbg':
            raise ValueError('WRONG COLOR DESC!')
        if len(color_desc) != raw_pattern.size:
            raise ValueError('WRONG PATTERN SIZE!')

        rawim = raw.raw_image_visible.copy()
        meta['black_level'] = raw.black_level_per_channel
        meta['white_level'] = raw.white_level
        meta['bayer_pattern'] = ''
        for i in range(raw_pattern.size):
            meta['bayer_pattern'] += color_desc[raw_pattern[i]]
    return rawim, meta


def load_mipi10bit(raw_path, height, width, padding_len=1, verbose=False):
    """
    Load MIPI raw image with bit depth 10bits which has been packed without redundancy. 4 continuous pixels
    in totally 40bits are packed into 5 uint8 numbers.
    Args:
        raw_path:
        height:
        width:
        padding_len: pad byte number of lines into multiple of padding_len. Default value is 1, no padding at all.
        verbose:

    Returns:
        im: raw image with type np.uint16 and shape (height, width)
    """
    if verbose:
        print('loading 10bitMIPI from {}'.format(raw_path))
    im = np.fromfile(raw_path, np.uint8)
    padded_width = int(math.ceil(width / 4) * 4)
    bytes_per_line = int(padded_width // 4 * 5)
    padded_bytes_per_line = math.ceil(bytes_per_line / padding_len) * padding_len

    im = np.reshape(im, (int(height), int(padded_bytes_per_line))).astype(np.int32)
    im = im[:, :int(bytes_per_line)]
    im = np.reshape(im, (height, int(bytes_per_line//5), 5))

    fifth = im[..., -1]
    im = im[..., :4]
    im = np.left_shift(im, 2)

    low_bit = np.zeros_like(im)
    low_bit[..., 0] = np.bitwise_and(fifth, 3)
    low_bit[..., 1] = np.bitwise_and(np.right_shift(fifth, 2), 3)
    low_bit[..., 2] = np.bitwise_and(np.right_shift(fifth, 4), 3)
    low_bit[..., 3] = np.bitwise_and(np.right_shift(fifth, 6), 3)

    im = im + low_bit
    im = np.reshape(im, (height, int(padded_width))).astype(np.uint16)
    im = im[:, :width]
    return im


def save_mipi10bit(raw_path, im, padding_len=1, verbose=False):
    """
    Inverse of load_mipi10bit.
    Args:
        raw_path: where to save *.RAWMIPI
        im: numpy.ndarray with dtype np.uint16
        padding_len: default value is 1.
        verbose:

    Returns:

    """
    height, width = im.shape
    padded_width = math.ceil(width / 4) * 4
    bytes_per_line = padded_width // 4 * 5
    padded_bytes_per_line = math.ceil(bytes_per_line / padding_len) * padding_len

    im = im.astype(np.int)
    im = np.pad(im, ((0, 0), (0, padded_width - width)))
    im = np.reshape(im, (height, padded_width//4, 4))

    tmp = np.zeros((height, bytes_per_line//5, 5), dtype=np.int)
    tmp[:, :, :4] = np.right_shift(im, 2)

    low_bit = np.bitwise_and(im, 3)
    low_bit[:, :, 1] = np.left_shift(low_bit[:, :, 1], 2)
    low_bit[:, :, 2] = np.left_shift(low_bit[:, :, 2], 4)
    low_bit[:, :, 3] = np.left_shift(low_bit[:, :, 3], 6)

    fifth = np.sum(low_bit, axis=-1)
    tmp[:, :, 4] = fifth

    im = tmp.astype(np.uint8)
    im = np.reshape(im, (height, bytes_per_line))
    im = np.pad(im, ((0, 0), (0, padded_bytes_per_line - bytes_per_line)))
    if verbose:
        print('saving 10bit MIPI to {}'.format(raw_path))
    im.tofile(raw_path)

def laod_uint8(rawPath, width, height):

    raw8bit = np.zeros([height, width])
    file = open(rawPath, 'rb')
    rawdata = file.read()
    print(len(rawdata), '888888888')
    file.close()

    stride = (width)

    rawdata = Image.frombytes('L', (int(stride), int(height)), rawdata)
    rawdata = np.asarray(rawdata)

    raw8bit = rawdata.reshape(height, width)

    #data_per_line = int(width)
    #first_byte = np.float32(rawdata[:, 0:data_per_line:1])
    #first_byte = np.uint8(rawdata[:, 0:data_per_line:1])
    #second_byte = np.uint8(rawdata[:, 1:data_per_line:1])

    # get 10bit data
    #first_pixel = np.bitwise_or(np.left_shift(second_byte, 8), first_byte)

    #raw8bit[:, 0:width] = first_byte


    return raw8bit

def load_plainraw(rawPath, width, height):

    raw16bit = np.zeros([height, width])
    file = open(rawPath, 'rb')
    rawdata = file.read()
    print(len(rawdata), 'rrrrrrrrrrr')
    file.close()

    stride = (width * 2)

    rawdata = Image.frombytes('L', (int(stride), int(height)), rawdata)
    rawdata = np.asarray(rawdata)

    data_per_line = int(width * 2)
    first_byte = np.uint16(rawdata[:, 0:data_per_line:2])
    second_byte = np.uint16(rawdata[:, 1:data_per_line:2])

    # get 10bit data
    first_pixel = np.bitwise_or(np.left_shift(second_byte, 8), first_byte)

    raw16bit[:, 0:width] = first_pixel

    return raw16bit

# def load_grayraw(rawPath, width, height):
#
#     raw16bit = np.zeros([height, width])
#     file = open(rawPath, 'rb')
#     rawdata = file.read()
#     file.close()
#
#     stride = (width)
#
#     rawdata = Image.frombytes('L', (int(stride), int(height)), rawdata)
#     rawdata = np.asarray(rawdata)
#
#     data_per_line = int(width)
#     first_byte = np.uint16(rawdata[:, 0:data_per_line:2])
#     second_byte = np.uint16(rawdata[:, 1:data_per_line:2])
#
#     # get 10bit data
#     first_pixel = np.bitwise_or(np.left_shift(second_byte, 8), first_byte)
#
#     raw16bit[:, 0:width] = first_pixel
#
#     return raw16bit

def Convert_Plain(raw16bit, width, height):
    raw16bit = np.uint16(raw16bit)

    second_byte = np.right_shift(raw16bit, 8)
    data_per_line = int(width * 2)
    MIPI_10BIT = np.zeros([height, data_per_line])
    MIPI_10BIT[:, 0:data_per_line:2] = np.uint8(raw16bit)
    MIPI_10BIT[:, 1:data_per_line:2] = np.uint8(second_byte)

    return MIPI_10BIT


def load_mat(raw_path, transpose=True):
    """
    Read raw image from .mat file.
    Args:
        raw_path:
        transpose:

    Returns:

    """
    with h5py.File(raw_path) as f:
        ds_list = []

        def func(name, obj):
            if isinstance(obj, h5py.Dataset):
                ds_list.append(name)

        f.visititems(func)
        if len(ds_list) > 1:
            print('Warning! More than one Dataset have been found in {}'.format(raw_path))
        im = f[ds_list[0]][()]
    if transpose:
        im = im.T
    return im.astype(np.float32)

def sidd_csv_loader(csv_path):
    """
    Load SIDD csv file. The purpose is to build mapping from phone to bayer pattern.
    Args:
        csv_path:

    Returns:

    """
    csv_file = open(csv_path, 'r')
    reader = csv.reader(csv_file)
    result = {}
    for item in reader:
        if reader.line_num == 1:
            continue
        result[item[0]] = item[1]
    csv_file.close()
    return result


def sidd_dirname_parser(instance_dir):
    sidd_keys = ['instance', 'scene', 'phone', 'iso', 'speed', 'temperature', 'brightness']
    return dict(zip(sidd_keys, os.path.basename(instance_dir).split('_')))


def load_mipi12bit(raw_path, height, width, padding_len=1, verbose=False):
    """
    Load MIPI raw image with bit depth 12bits which has been packed without redundancy. 2 continuous pixels
    in totally 24bits are packed into 3 uint8 numbers.
    Args:
        raw_path:
        height:
        width:
        padding_len: pad byte number of lines into multiple of padding_len. Default value is 1, no padding at all.
        verbose:

    Returns:
        im: raw image with type np.uint16 and shape (height, width)
    """
    if verbose:
        print('loading 10bitMIPI from {}'.format(raw_path))
    rawdata = np.fromfile(raw_path, np.uint8)
    padded_width = int(math.ceil(width / 4) * 4)
    bytes_per_line = int(padded_width // 2 * 3)
    bytes_per_line = math.ceil(float(bytes_per_line) / 16) * 16
    # rawdata = rawdata.reshape([h, bytes_per_line])
    rawdata = rawdata.reshape([height, int(bytes_per_line)])

    data_per_line = int(width * 3 / 2)
    first_byte = np.uint16(rawdata[:, 0:data_per_line:3])
    second_byte = np.uint16(rawdata[:, 1:data_per_line:3])
    third_byte = np.uint16(rawdata[:, 2:data_per_line:3])

    # get 10bit data
    first_pixel = np.bitwise_or(np.left_shift(first_byte, 4), np.bitwise_and(third_byte, 15))
    second_pixel = np.bitwise_or(np.left_shift(second_byte, 4), np.right_shift(np.bitwise_and(third_byte, 240), 4))

    raw16bit = np.zeros([height, width])
    raw16bit[:, 0:width:2] = first_pixel
    raw16bit[:, 1:width:2] = second_pixel
    return raw16bit


def Convert_Unpacked12bit(raw16bit, width, height):
    raw16bit = np.uint16(raw16bit)
    first_pixel = raw16bit[:, 0:width:2]
    second_pixel = raw16bit[:, 1:width:2]
    first_byte = np.right_shift(first_pixel, 4)
    second_byte = np.right_shift(second_pixel, 4)
    first_four_bit = np.bitwise_and(first_pixel, 15)
    second_four_bit = np.bitwise_and(second_pixel, 15)
    third_byte = first_four_bit + np.left_shift(second_four_bit, 4)
    data_per_line = int(width * 3 / 2)
    padded_width = int(math.ceil(width / 4) * 4)
    bytes_per_line = int(padded_width // 2 * 3)
    bytes_per_line = int(math.ceil(float(bytes_per_line) / 16) * 16)
    MIPI_12BIT = np.zeros([height, bytes_per_line])
    MIPI_12BIT[:, 0:data_per_line:3] = np.uint8(first_byte)
    MIPI_12BIT[:, 1:data_per_line:3] = np.uint8(second_byte)
    MIPI_12BIT[:, 2:data_per_line:3] = np.uint8(third_byte)
    return MIPI_12BIT
