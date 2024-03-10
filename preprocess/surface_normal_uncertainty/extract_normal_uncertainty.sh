# 用法：sh extract_normal_uncertainty.sh <data_dir>
# 要求data_dir下存在rgb目录存储图片
# BN-scannet或GN-nyu
data_dir=$1
python run.py -i $data_dir/rgb -n $data_dir/mono_normal -u $data_dir/uncertainty --architecture BN --pretrained scannet