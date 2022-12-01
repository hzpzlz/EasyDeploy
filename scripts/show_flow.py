import os
import time
import argparse
import glob, cv2
import numpy as np
import math

import imageio
import argparse

import flow_viz

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
parser.add_argument('--result_path',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
parser.add_argument('--flag',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')
parser.add_argument('--ext',type=str,default='/home/hzp/datasets/night_scan/raw_base_align')

args=parser.parse_args()

def convertOptRGB(optical_flows, OutFolder, basename):
    '''
    :param optical_flows:  h*w*2
    :param OutFolder:
    :param basename:
    :return:
    '''

    for i, optical_flow in enumerate(optical_flows):
        #flow_img = flow_viz.flow_to_image(optical_flow)

        optical_flow = optical_flow.transpose(2, 0, 1)
        blob_x = optical_flow[0]
        blob_y = optical_flow[1]

        hsv = np.zeros((blob_x.shape[0], blob_y.shape[1], 3), np.uint8)
        mag, ang = cv2.cartToPolar(blob_x, blob_y)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        result = np.zeros((blob_x.shape[0], blob_x.shape[1],3),np.uint8)
        result[:, :, 0] = bgr[:, :, 2]
        result[:, :, 1] = bgr[:, :, 1]
        result[:, :, 2] = bgr[:, :, 0]

        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.jpg')
        cv2.imwrite(outputName, result)

def write_flow(optical_flows, OutFolder, basename):
    '''
    :param optical_flows:  h*w*2
    :param OutFolder:
    :param basename:
    :return:
    '''

    for i, optical_flow in enumerate(optical_flows):
        #flow_img = flow_viz.flow_to_image(optical_flow)

        result = flow_viz.flow_to_image(optical_flow)

        outputName = os.path.join(OutFolder, basename + "_" + str(i) + '.jpg')
        print(outputName, 'flow !!!!!!!')
        cv2.imwrite(outputName, result)

inputFolder = args.root_path
inputFolderFlow = args.root_path

burstList = glob.glob(os.path.join(inputFolder, '*'))
burstList.sort()


# Process all the bursts
for burstNumber, burstName in enumerate(burstList):
    burstPath = os.path.join(inputFolder, burstName)
    print(burstPath)


    FlowPath = os.path.join(inputFolderFlow, burstName.split("/")[-1])


    flowPathList = glob.glob(os.path.join(FlowPath, args.ext)) #GBRG
    #print(flowPathList, 'fffff')
    flowPathList.sort()
    for x in flowPathList:
        print(x, '\n')

    refIdx = int(burstName.split("/")[-1].split("_")[-3])-1

    flows = []
    if args.ext=='*.sdat':
        m=96
        n=128
    elif args.ext=='*.dat':
        m = 1536
        n = 2048
    for i in range(int(len(flowPathList)/2)):
        flowu = np.fromfile(flowPathList[2*i], dtype=np.float32, count=-1, sep='', offset=0)
        flowv = np.fromfile(flowPathList[2 * i + 1], dtype=np.float32, count=-1, sep='', offset=0)
        flowu = flowu.reshape(m, n)
        flowv = flowv.reshape(m, n)
        flows.append(np.stack((flowu, flowv), axis=2))


    flowssub = []
    i=0
    for flow in flows:
        m, n, c = flow.shape
        #print(flow)
        x = range(0, n)
        y = range(0, m)

        xx, yy = np.meshgrid(x, y)
        flow[:, :, 0] = flow[:, :, 0] - xx.astype(np.float32)
        flow[:, :, 1] = flow[:, :, 1] - yy.astype(np.float32)
        #flow[:, :, 0] = flow[:, :, 0] # - xx.astype(np.float32)
        #flow[:, :, 1] = flow[:, :, 1] #- yy.astype(np.float32)
        #if i>=5:
        #    distance = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        #    distance[distance<12] = 0
        #    cv2.imwrite(os.path.join(FlowPath, str(i) + '_flowT12.jpg'), distance*255)
        #i=i+1
        #print(flow)

        #flow = cv2.medianBlur(flow, 5)
        #flow = cv2.medianBlur(flow, 3)
        #flow = cv2.medianBlur(flow, 5)
        #print(flow)

        flowssub.append(flow)
    print(len(flowssub), 'ffffffff')

    if args.ext=='*.dat':
        convertOptRGB(flowssub, FlowPath, 'flow_' + args.flag)
    elif args.ext=='*.sdat':
        convertOptRGB(flowssub, FlowPath, 'flow_small' + args.flag)
    #write_flow(flowssub, FlowPath, 'flow_' + args.flag)
