
#rm -r /home/hezhipeng/codes/result/BasicAlign/results/Unet/*
#python test_denoise.py -opt options/Unet/test_Unet.yml
#scp -r /home/hezhipeng/codes/result/BasicAlign/results/Unet/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/denoise/' 

#rm -r /home/hezhipeng/codes/result/BasicAlign/results/STN/*
#python test_stn.py -opt options/STN/test_STN.yml
#scp -r /home/hezhipeng/codes/result/BasicAlign/results/STN/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/' 
rm -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v1/*
python test_stn.py -opt options/STN_v1/test_STN_v1.yml
scp -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v1/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/stn_v1' 
rm -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v2/*
python test_stn.py -opt options/STN_v2/test_STN_v2.yml
scp -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v2/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/stn_v2' 
rm -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v3/*
python test_stn.py -opt options/STN_v3/test_STN_v3.yml
scp -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_v3/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/stn_v3' 
#rm -r /home/hezhipeng/codes/result/BasicAlign/results/STN_Unet/*
#python test_stn.py -opt options/STN_Unet/test_STN_Unet.yml
#scp -r /home/hezhipeng/codes/result/BasicAlign/results/STN_Unet/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/' 
#rm -r /home/hezhipeng/codes/result/BasicAlign/results/STN_Unet_v1/*
#python test_stn.py -opt options/STN_Unet_v1/test_STN_Unet_v1.yml
#scp -r /home/hezhipeng/codes/result/BasicAlign/results/STN_Unet_v1/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/' 
#rm -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_Unet_v2/*
#python test_stn.py -opt options/STN_Unet_v2/test_STN_Unet_v2.yml
#scp -r /home/work/ssd1/hezhipeng/result/BasicAlign/results/STN_Unet_v2/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/' 

#rm -r /home/hezhipeng/codes/result/BasicAlign/results/Unet/*
#python test_stn.py -opt options/Unet/test_Unet.yml
#scp -r /home/hezhipeng/codes/result/BasicAlign/results/Unet/davis/* hzp@10.241.17.204:'/home/hzp/codes/BasicAlign/stn/' 
