import os
import time
import argparse
import glob, cv2
import numpy as np
import math, copy
import pandas as pd

from utils.raw import bayer_to_offsets, rescaling, pack_raw_to_4ch, unrescaling
from utils.raw_io import load_mipi12bit, load_plainraw, load_mipi10bit
from utils.demo import demosaic_sample
import imageio

def adjust_gamma(imgs, gamma=1.0):
    '''
    :param imgs:  range 0-255
    :param gamma:
    :return:
    '''

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    return new_imgs


def Raw2Jpg(rawBayers, bayerpattern, ratio, bl, wl, OutFolder, basename):

    offset = bayer_to_offsets(bayerpattern)

    for i, rawBayer in enumerate(rawBayers):

        pack = pack_raw_to_4ch(rawBayer, offset)
        rescale = rescaling(pack, bl, wl, clipping=1)
        rescale = rescale * ratio
        out = demosaic_sample(rescale)
        out = np.minimum(np.maximum(out, 0), 1)

        out = adjust_gamma(out * 255, 2.5)

        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.jpg')
        imageio.imwrite(outputName, out) #out * 255


def NegRaw2Jpg(rawBayers, bayerpattern, ratio, bl, wl, OutFolder, basename):

    offset = bayer_to_offsets(bayerpattern)

    evNeg_ev_info = [0, -36, -12]

    for i, rawBayer in enumerate(rawBayers):

        pack = pack_raw_to_4ch(rawBayer, offset)
        rescale = rescaling(pack, bl, wl, clipping=1)

        r = 2 ** (-evNeg_ev_info[i] / 6)
        # if r > 1:
        #     if total_gain < 16:
        #         r = r
        #     elif total_gain > 16 * r:
        #         r = 1
        #     else:
        #         r = r / (total_gain / 16)

        rescale = rescale * r * ratio
        out = demosaic_sample(rescale)
        out = np.minimum(np.maximum(out, 0), 1)

        # out = (out * 255).astype(np.uint8)
        out = adjust_gamma(out * 255, 2.5)

        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.jpg')
        imageio.imwrite(outputName, out) #out * 255


def WriteRaw(rawBayers, OutFolder, basename):
    for i, rawBayer in enumerate(rawBayers):
        mergedBayer = np.minimum(np.maximum(rawBayer, 0), 16383)
        mergedBayer = mergedBayer.astype(np.uint16)
        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.idealraw')
        mergedBayer.tofile(outputName)

def warp_raw(raws, flos, refIdx):
    out_raws = []
    height, width = raws[refIdx].shape
    tmpalign = np.zeros((height, width), dtype='uint16')

    for idx in range(len(raws)):
        if idx < refIdx:
            kdx = idx
        elif idx == refIdx:
            out_raws.append(raws[refIdx])
            continue
        else:
            kdx = idx - 1

        flow = flos[kdx]
        raw_tmp = raws[idx]
        h, w = flow.shape[:2]
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]

        for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
            frame_gray1 = raw_tmp[di::2, dj::2]
            remapped_image2 = cv2.remap(frame_gray1, flow, None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101) #cv2.INTER_NEAREST
            tmpalign[di::2, dj::2] = remapped_image2

        mergedBayer = np.minimum(np.maximum(tmpalign, 0), 16383)
        mergedBayer = mergedBayer.astype(np.uint16)

        out_raws.append(mergedBayer)

    return out_raws


def convertOptRGB(optical_flows, OutFolder, basename):
    '''

    :param optical_flows:  h*w*2
    :param OutFolder:
    :param basename:
    :return:
    '''

    for i, optical_flow in enumerate(optical_flows):

        optical_flow = optical_flow.transpose(2, 0, 1)
        blob_x = optical_flow[0]
        blob_y = optical_flow[1]

        hsv = np.zeros((blob_x.shape[0], blob_y.shape[1], 3), np.uint8)
        mag, ang = cv2.cartToPolar(blob_x, blob_y)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = np.zeros((blob_x.shape[0], blob_x.shape[1],3), np.uint8)
        result[:, :, 0] = bgr[:, :, 2]
        result[:, :, 1] = bgr[:, :, 1]
        result[:, :, 2] = bgr[:, :, 0]

        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.jpg')
        cv2.imwrite(outputName, result)


def WriteGray255(grayImages, OutFolder, basename):

    for i, grayImage in enumerate(grayImages):

        outputName = os.path.join(OutFolder, "gray_" + basename + "_" + str(i) + '.jpg')
        imageio.imwrite(outputName, grayImage*255)


###################multi frame

raw_dir = "/home/hzp/datasets/night_scan/L1_project/0316_problem/problems"

dir_path = "/home/hzp/datasets/night_scan/L1_project/0316_problem/problems"
dirs = os.listdir(dir_path)
dirs.sort()

for dir in dirs:
    print(dir)
    inputFolderFlow = os.path.join(dir_path, dir)

    rawpath = os.path.join(raw_dir, dir)

    # inputFolderFlow = "/home/songxiaohong/tmpdir/align/Data/0309_4F_factorV1_base_align_later4/IMG_20220309_205354_AnchorFrame_7_align_4567" #IMG_20220309_205008_AnchorFrame_4_align_2345"

    backflowPathList = glob.glob(os.path.join(inputFolderFlow, '*backdisflo.flo'))  # GBRG
    backflowPathList.sort()

    forflowPathList = glob.glob(os.path.join(inputFolderFlow, '*fordisflo.flo'))  # GBRG
    forflowPathList.sort()

    backflows = []
    m = 1536
    n = 2048
    for i in range(int(len(backflowPathList) / 2)):
        flowu = np.fromfile(backflowPathList[2 * i], dtype=np.float32, count=-1, sep='', offset=0)
        flowv = np.fromfile(backflowPathList[2 * i + 1], dtype=np.float32, count=-1, sep='', offset=0)
        flowu = flowu.reshape(m, n)
        flowv = flowv.reshape(m, n)
        backflows.append(np.stack((flowu, flowv), axis=2))


    forflows = []
    m = 1536
    n = 2048
    for i in range(int(len(forflowPathList) / 2)):
        flowu = np.fromfile(forflowPathList[2 * i], dtype=np.float32, count=-1, sep='', offset=0)
        flowv = np.fromfile(forflowPathList[2 * i + 1], dtype=np.float32, count=-1, sep='', offset=0)
        flowu = flowu.reshape(m, n)
        flowv = flowv.reshape(m, n)
        forflows.append(np.stack((flowu, flowv), axis=2))


    backflowssub = []
    for flow in backflows:
        m, n, c = flow.shape
        x = range(0, n)
        y = range(0, m)

        xx, yy = np.meshgrid(x, y)
        flow[:, :, 0] = flow[:, :, 0] - xx.astype(np.float32)
        flow[:, :, 1] = flow[:, :, 1] - yy.astype(np.float32)

        backflowssub.append(flow)


    forflowssub = []
    for flow in forflows:
        m, n, c = flow.shape
        x = range(0, n)
        y = range(0, m)

        xx, yy = np.meshgrid(x, y)
        flow[:, :, 0] = flow[:, :, 0] - xx.astype(np.float32)
        flow[:, :, 1] = flow[:, :, 1] - yy.astype(np.float32)

        forflowssub.append(flow)

    ################################################################
    # aaa = backflowssub[0]
    # aaa = aaa[16:1536:32, 16:2048:32, 0]
    # bbb = backflowssub[2]
    # bbb = bbb[16:1536:32, 16:2048:32, 0]
    #
    #
    # ccc = forflowssub[0]
    # ccc = ccc[16:1536:32, 16:2048:32, 0]
    # ddd = forflowssub[2]
    # ddd = ddd[16:1536:32, 16:2048:32, 0]

    convertOptRGB(backflowssub, rawpath, 'colorback')
    convertOptRGB(forflowssub, rawpath, 'colorfor')

    maskbacks = []
    maskforws = []

    masksmall = []

    smallbackflow = []
    smallforflow = []
    warpflow = []

    for flowbacksrc, flowforwsrc in zip(backflowssub, forflowssub):

        amh = 1536
        amw = 2048

        flowback = flowbacksrc[16:amh:32, 16:amw:32]
        smallbackflow.append(flowback)
        # flowforw = flowforwsrc[16:amh:32, 16:amw:32]

        # flowback = np.repeat(flowback, 16, axis=0)
        # flowback = np.repeat(flowback, 16, axis=1)
        #
        # flowforw = np.repeat(flowforw, 16, axis=0)
        # flowforw = np.repeat(flowforw, 16, axis=1)


        flowforw = flowforwsrc[32:amh:64, 32:amw:64]
        flowforw = np.repeat(flowforw, 2, axis=0)
        flowforw = np.repeat(flowforw, 2, axis=1)
        smallforflow.append(flowforw)



        flowback = cv2.GaussianBlur(flowback, (3, 3), 0)
        flowforw = cv2.GaussianBlur(flowforw, (3, 3), 0)

        mh = flowback.shape[0]
        mw = flowforw.shape[1]

        indI = np.clip(((np.repeat((np.arange(mh)).reshape(mh, 1), mw, axis=1)).reshape(mh, mw) + flowback[:, :, 1]), 0,
                       mh - 1)
        indJ = np.clip(((np.repeat((np.arange(mw)).reshape(1, mw), mh, axis=0)).reshape(mh, mw) + flowback[:, :, 0]), 0,
                       mw - 1)

        backindxi = np.rint(indI)
        backindxj = np.rint(indJ)

        backindxi = backindxi.astype(np.int)
        backindxj = backindxj.astype(np.int)

        backindxi = np.clip(backindxi, 0, mh - 1)
        backindxj = np.clip(backindxj, 0, mw - 1)


        # flomean = (abs(flowback)).mean()
        # print(flomean)


        out = flowback + flowforw[backindxi, backindxj, :]
        warpflow.append(flowforw[backindxi, backindxj, :])
        mask = np.ones((mh, mw), dtype=np.uint8)
        diff = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2)
        mask[diff > 3] = 0

        masksmall.append(mask)

        # mask = cv2.medianBlur(mask, 3)
        # kernel = np.ones((1, 1), dtype=np.uint8)
        # mask = cv2.erode(mask, kernel, 1)

        repeatmask = np.repeat(mask, 16, axis=0)
        repeatmask = np.repeat(repeatmask, 16, axis=1)

        mask = cv2.resize(mask, (amw, amh), interpolation=cv2.INTER_LINEAR)


        maskbacks.append(mask)





        # #################################
        # indI = np.clip(((np.repeat((np.arange(mh)).reshape(mh, 1), mw, axis=1)).reshape(mh, mw) + flowforw[:, :, 1]), 0,
        #                mh - 1)
        # indJ = np.clip(((np.repeat((np.arange(mw)).reshape(1, mw), mh, axis=0)).reshape(mh, mw) + flowforw[:, :, 0]), 0,
        #                mw - 1)
        #
        # backindxi = np.rint(indI)
        # backindxj = np.rint(indJ)
        #
        # backindxi = backindxi.astype(np.int)
        # backindxj = backindxj.astype(np.int)
        #
        # backindxi = np.clip(backindxi, 0, mh - 1)
        # backindxj = np.clip(backindxj, 0, mw - 1)
        #
        #
        # out = flowforw + flowback[backindxi, backindxj, :]
        # mask = np.ones((mh, mw), dtype=np.uint8)
        # diff = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2)
        # mask[diff > 1] = 0
        #
        # # repeatmask = np.repeat(mask, 32, axis=0)
        # # repeatmask = np.repeat(repeatmask, 32, axis=1)
        #
        # mask = cv2.resize(mask, (amw, amh), interpolation=cv2.INTER_LINEAR)
        # maskforws.append(mask)

    WriteGray255(masksmall, rawpath, 'maskwarpbacksmall')
    WriteGray255(maskbacks, rawpath, 'maskwarpback')
    convertOptRGB(smallbackflow, rawpath, 'smallcolorback')
    convertOptRGB(smallforflow, rawpath, 'smallcolorforw')
    convertOptRGB(warpflow, rawpath, 'warpforw')


    # WriteGray255(maskforws, inputFolderFlow, 'maskwarpforw')
#############################################point
    maskresult = []
    csvfiles = glob.glob(rawpath + '/*.csv')  # bggr
    csvfiles.sort()

    lll = []

    if(len(csvfiles) > 0):

        for cc in range(3):
            df = pd.read_csv(csvfiles[cc], header=0, encoding="gb18030")
            pt_x = df["pt_x"]
            pt_y = df["pt_y"]
            query_pts_x = np.array(pt_x)[1::2].reshape(-1, 1)
            query_pts_y = np.array(pt_y)[1::2].reshape(-1, 1)

            query_pts_x = query_pts_x / 16 + 0.5
            query_pts_y = query_pts_y / 16 + 0.5

            query_pts_x = query_pts_x.astype(np.int)
            query_pts_y = query_pts_y.astype(np.int)

            img1 = (1 - masksmall[cc]) * 255  #mask反过来了 现在白色是遮挡区域   之前黑色是遮挡 白色是非遮挡区域

            imgpoints = np.zeros_like(img1)

            hh = imgpoints.shape[0]
            ww = imgpoints.shape[1]

            print("hh : ", hh, "ww : ", ww)
            for i in range(query_pts_y.shape[0]):
                imgpoints[query_pts_y[i], query_pts_x[i]] += 1

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img1, connectivity=4)
            np.savetxt(rawpath+'/'+str(cc)+'_sxh.csv', labels,  delimiter = ',')
            xxx = labels
            xxx[xxx!=0] = 1
            lll.append(xxx) 

            areas = np.zeros((num_labels, 2), dtype=np.int)
            print(labels.max())

            for i in range(hh):
                for j in range(ww):
                    areas[labels[i, j], 0] += 1
                    if imgpoints[i, j] != 0:
                        areas[labels[i, j], 1] += imgpoints[i, j]

            # outmask = np.ones_like(img1)

            for i in range(1, num_labels):
                if (areas[i, 1] < 5):
                    labels[labels == i] = 0
                else:
                    labels[labels == i] = -1

            labels[labels == -1] = 1

            labels = 1 - labels
            labels = labels.astype(np.uint8)

            labels = np.repeat(labels, 16, axis=0)
            labels = np.repeat(labels, 16, axis=1)

            labels = cv2.resize(labels, (2048, 1536), interpolation=cv2.INTER_LINEAR)


            maskresult.append(labels)


        # maskresult = []
        # for maskback, maskforw in zip(maskbacks, maskforws):
        #
        #     mask = maskback + maskforw
        #     mask[mask < 2] = 0
        #     mask[mask == 2] = 1
        #     maskresult.append(mask)
        #
        WriteGray255(maskresult, rawpath, 'masklast')
    else:
        for cc in range(3):
            img1 = (1 - masksmall[cc]) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img1, connectivity=4)
            np.savetxt(rawpath+'/'+str(cc)+'_sxh.csv', labels,  delimiter = ',')
            #xxx = labels
            #xxx[xxx!=0] = 1
            #lll.append(xxx) 

        maskresult = maskbacks
        #WriteGray255(lll, rawpath, 'lty')

    # maskresult = maskbacks

    # rawpath = "/home/songxiaohong/tmpdir/align/0302/scratch_capture_pipeline_python-deghost/result/0225_testdeghost_motion_backfor/IMG_20220225_211052_AnchorFrame_1_align_1234"

    #mipifiles = glob.glob(rawpath + '/*.algideal')  # bggr
    #mipifiles.sort()


    #baseraw = load_plainraw(mipifiles[0], 4096, 3072)
    #baseraw = np.float32(baseraw)



    #outraws = []
    #outraws.append(baseraw)


    #for i, mipifile in enumerate(mipifiles[1:]):

    #    noiseimg = load_plainraw(mipifile, 4096, 3072)
    #    output_raw = np.float32(noiseimg)


    #    tmpalign = np.zeros((3072, 4096), dtype='uint16')

    #    for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
    #        ref = output_raw[di::2, dj::2]
    #        base = baseraw[di::2, dj::2]

    #        remapped_image = maskresult[i] * ref + (1-maskresult[i]) * base

    #        tmpalign[di::2, dj::2] = remapped_image

    #    outraws.append(tmpalign)

    #WriteRaw(outraws, rawpath, "alignlast")
    #Raw2Jpg(outraws, "bggr", 2, 1024.0, 16383.0, rawpath, "alignlast")



# ###################single frame
# inputFolderFlow = "/home/songxiaohong/tmpdir/align/Data/0225_4F_factorV1_base_align_motion_distortion/IMG_20220225_211002_AnchorFrame_1_align_1234"
#
# backflowPathList = glob.glob(os.path.join(inputFolderFlow, '*backdisflo.flo'))  # GBRG
# backflowPathList.sort()
#
# forflowPathList = glob.glob(os.path.join(inputFolderFlow, '*fordisflo.flo'))  # GBRG
# forflowPathList.sort()
# # flowPathList = flowPathList[:2]
#
# # burstName = "IMG_20220217_214216_AnchorFrame_3_align_1234"
# # refIdx = int(burstName.split("/")[-1].split("_")[-3]) - 1
#
# backflows = []
# m = 1536
# n = 2048
# for i in range(int(len(backflowPathList) / 2)):
#     flowu = np.fromfile(backflowPathList[2 * i], dtype=np.float32, count=-1, sep='', offset=0)
#     flowv = np.fromfile(backflowPathList[2 * i + 1], dtype=np.float32, count=-1, sep='', offset=0)
#     flowu = flowu.reshape(m, n)
#     flowv = flowv.reshape(m, n)
#     backflows.append(np.stack((flowu, flowv), axis=2))
#
#
# forflows = []
# m = 1536
# n = 2048
# for i in range(int(len(forflowPathList) / 2)):
#     flowu = np.fromfile(forflowPathList[2 * i], dtype=np.float32, count=-1, sep='', offset=0)
#     flowv = np.fromfile(forflowPathList[2 * i + 1], dtype=np.float32, count=-1, sep='', offset=0)
#     flowu = flowu.reshape(m, n)
#     flowv = flowv.reshape(m, n)
#     forflows.append(np.stack((flowu, flowv), axis=2))
#
#
# backflowssub = []
# for flow in backflows:
#     m, n, c = flow.shape
#     x = range(0, n)
#     y = range(0, m)
#
#     xx, yy = np.meshgrid(x, y)
#     flow[:, :, 0] = flow[:, :, 0] - xx.astype(np.float32)
#     flow[:, :, 1] = flow[:, :, 1] - yy.astype(np.float32)
#
#     backflowssub.append(flow)
#
#
# forflowssub = []
# for flow in forflows:
#     m, n, c = flow.shape
#     x = range(0, n)
#     y = range(0, m)
#
#     xx, yy = np.meshgrid(x, y)
#     flow[:, :, 0] = flow[:, :, 0] - xx.astype(np.float32)
#     flow[:, :, 1] = flow[:, :, 1] - yy.astype(np.float32)
#
#     forflowssub.append(flow)
#
#
#
# convertOptRGB(backflowssub, inputFolderFlow, 'colorback')
# convertOptRGB(forflowssub, inputFolderFlow, 'colorfor')
#
# maskbacks = []
# maskforws = []
# for flowback, flowforw in zip(backflowssub, forflowssub):
#
#     mh = 1536
#     mw = 2048
#
#     indI = np.clip(((np.repeat((np.arange(mh)).reshape(mh, 1), mw, axis=1)).reshape(mh, mw) + flowback[:, :, 1]), 0,
#                    mh - 1)
#     indJ = np.clip(((np.repeat((np.arange(mw)).reshape(1, mw), mh, axis=0)).reshape(mh, mw) + flowback[:, :, 0]), 0,
#                    mw - 1)
#
#     backindxi = np.rint(indI)
#     backindxj = np.rint(indJ)
#
#     backindxi = backindxi.astype(np.int)
#     backindxj = backindxj.astype(np.int)
#
#     backindxi = np.clip(backindxi, 0, mh - 1)
#     backindxj = np.clip(backindxj, 0, mw - 1)
#
#
#     out = flowback[backindxi, backindxj, :] + flowforw
#     mask = np.ones((mh, mw), dtype=np.uint8)
#     diff = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2)
#     mask[diff > 1] = 0
#     maskbacks.append(mask)
#
#
#     # #################################
#     # indI = np.clip(((np.repeat((np.arange(mh)).reshape(mh, 1), mw, axis=1)).reshape(mh, mw) + flowforw[:, :, 1]), 0,
#     #                mh - 1)
#     # indJ = np.clip(((np.repeat((np.arange(mw)).reshape(1, mw), mh, axis=0)).reshape(mh, mw) + flowforw[:, :, 0]), 0,
#     #                mw - 1)
#     #
#     # backindxi = np.rint(indI)
#     # backindxj = np.rint(indJ)
#     #
#     # backindxi = backindxi.astype(np.int)
#     # backindxj = backindxj.astype(np.int)
#     #
#     # backindxi = np.clip(backindxi, 0, mh - 1)
#     # backindxj = np.clip(backindxj, 0, mw - 1)
#     #
#     #
#     # out = flowforw[backindxi, backindxj, :] + flowback
#     # mask = np.ones((mh, mw), dtype=np.uint8)
#     # diff = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2)
#     # mask[diff > 1] = 0
#     # maskforws.append(mask)
#
#
# WriteGray255(maskbacks, inputFolderFlow, 'maskwarpback')
# # WriteGray255(maskforws, inputFolderFlow, 'maskwarpforw')
# #
# #
# # maskresult = []
# # for maskback, maskforw in zip(maskbacks, maskforws):
# #
# #     mask = maskback + maskforw
# #     mask[mask < 2] = 0
# #     mask[mask == 2] = 1
# #     maskresult.append(mask)
# #
# # WriteGray255(maskresult, inputFolderFlow, 'masklast')
#
# maskresult = maskbacks
#
# rawpath = "/home/songxiaohong/tmpdir/align/0302/scratch_capture_pipeline_python-deghost/result/0225_testdeghost_motion_backfor/IMG_20220225_211002_AnchorFrame_1_align_1234"
#
# mipifiles = glob.glob(rawpath + '/*.algideal')  # bggr
# mipifiles.sort()
#
#
# baseraw = load_plainraw(mipifiles[0], 4096, 3072)
# baseraw = np.float32(baseraw)
#
#
#
# outraws = []
# outraws.append(baseraw)
#
#
# for i, mipifile in enumerate(mipifiles[1:]):
#
#     noiseimg = load_plainraw(mipifile, 4096, 3072)
#     output_raw = np.float32(noiseimg)
#
#
#     tmpalign = np.zeros((3072, 4096), dtype='uint16')
#
#     for c, (di, dj) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1])):
#         ref = output_raw[di::2, dj::2]
#         base = baseraw[di::2, dj::2]
#
#         remapped_image = maskresult[i] * ref + (1-maskresult[i]) * base
#
#         tmpalign[di::2, dj::2] = remapped_image
#
#     outraws.append(tmpalign)
#
# WriteRaw(outraws, rawpath, "alignlast")
# Raw2Jpg(outraws, "bggr", 2, 1024.0, 16383.0, rawpath, "alignlast")


