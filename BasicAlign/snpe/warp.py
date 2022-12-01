import cv2
import numpy as np
import flow_viz

if __name__ == '__main__':
    data_path = '/home/hzp/codes/BasicAlign/demo-frames'
    h=436
    w=1024
    input_flow_path = '/home/hzp/codes/BasicAlign/result/pwcplus_onnx/output/Result_0/output.raw'
    flow_dlc = np.fromfile(input_flow_path, dtype=np.float32, count=h*w*2)

    flow_res = flow_dlc.reshape(1,h,w,2)[0]

    flo = flow_viz.flow_to_image(flow_res)
    print(flo.shape)
    cv2.imwrite('flow_noquant_phone.jpg', flo)
    #print(flow_dlc.shape, 'ssss')