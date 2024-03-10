data_dir=$1
prompt=${2:-ceiling.wall.floor}
python sam.py -i $data_dir/rgb -o $data_dir/mask --prompt $prompt