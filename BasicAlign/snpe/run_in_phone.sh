#!/system/bin/bash
root_dir=/data/local/optflow
cd $root_dir
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root_dir/arm64/lib
export PATH=$PATH:$root_dir/arm64/bin
snpe-net-run --version
snpe-net-run --container pwcplus_95000_G.dlc --input_list rawdata.txt --output_dir output

