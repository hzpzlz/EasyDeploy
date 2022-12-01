#!/bin/bash
#python warp.py -opt options/RAFT/test_RAFT.yml

#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT-chairs/*
#python warp.py -opt options/RAFT-chairs/test_RAFT-chairs.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT-things/*
#python warp.py -opt options/RAFT-things/test_RAFT-things.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT-things/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT-sintel/*
#python warp.py -opt options/RAFT-sintel/test_RAFT-sintel.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT-sintel/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp

#python warp.py -opt options/RAFT-things/test_RAFT-things.yml
#python warp.py -opt options/RAFT-sintel/test_RAFT-sintel.yml
#python warp.py -opt options/RAFT-kitti/test_RAFT-kitti.yml
#python warp.py -opt options/IRRPWC/test_IRRPWC.yml

#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-chairs/*
#python onnx_test.py -opt options/RAFT_STN-chairs/test_RAFT_STN-chairs.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-chairs/*.onnx hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-things/*
#python warp.py -opt options/RAFT_STN-things/test_RAFT_STN-things.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-things/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#rm /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-sintel/*
#python onnx_test.py -opt options/RAFT_STN-sintel/test_RAFT_STN-sintel.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-sintel/*.png hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp
#scp /home/hezhipeng/codes/result/BasicAlign/results/RAFT_STN-sintel/*.onnx hzp@10.241.17.204:/home/hzp/codes/BasicAlign/warp

python onnx_test.py -opt options/pwcplus/test_pwcplus.yml

#rm /home/hezhipeng/codes/result/BasicAlign/results/PWCNet-chairs/*
#python onnx_test.py -opt options/PWCNet/test_PWCNet.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/PWCNet-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/PWC-Net_pytorch/warp
#scp /home/hezhipeng/codes/result/BasicAlign/results/PWCNet-chairs/*.onnx hzp@10.241.17.204:/home/hzp/codes/PWC-Net_pytorch/warp

#rm /home/hezhipeng/codes/result/BasicAlign/results/RMOF/*
#rm /home/hezhipeng/codes/result/BasicAlign/results/RMOF_l1/*
#python warp.py -opt options/RMOF/test_RMOF.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RMOF/*.png hzp@10.241.17.204:/home/hzp
#scp /home/hezhipeng/codes/result/BasicAlign/results/RMOF_l1/*.png hzp@10.241.17.204:/home/hzp

#rm /home/hezhipeng/codes/result/BasicAlign/results/RMOF_v1/*
#python warp.py -opt options/RMOF_v1/test_RMOF_v1.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/RMOF_v1/*.png hzp@10.241.17.204:/home/hzp

#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-chairs/*
#python warp.py -opt options/MaskFlowNet-chairs/test_MaskFlowNet-chairs.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 
#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-things/*
#python warp.py -opt options/MaskFlowNet-things/test_MaskFlowNet-things.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-things/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 
#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-sintel/*
#python warp.py -opt options/MaskFlowNet-sintel/test_MaskFlowNet-sintel.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet-sintel/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 

#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-chairs/*
#python warp.py -opt options/MaskFlowNet_v2-chairs/test_MaskFlowNet_v2-chairs.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 
#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-things/*
#python warp.py -opt options/MaskFlowNet_v2-things/test_MaskFlowNet_v2-things.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-things/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 
#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-sintel/*
#python warp.py -opt options/MaskFlowNet_v2-sintel/test_MaskFlowNet_v2-sintel.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v2-sintel/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 

#rm /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v3-chairs/*
#python warp.py -opt options/MaskFlowNet_v3-chairs/test_MaskFlowNet_v3-chairs.yml
#scp /home/hezhipeng/codes/result/BasicAlign/results/MaskFlowNet_v3-chairs/*.png hzp@10.241.17.204:/home/hzp/codes/MaskFlownet_pytorch/warp 
