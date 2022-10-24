script_dir="$(dirname "${BASH_SOURCE[0]}")"
script_dir="$(realpath "${script_dir}")"

export LD_LIBRARY_PATH=$script_dir:$LD_LIBRARY_PATH

if [ ! -f $script_dir/daai ]; then
    mkdir $script_dir/build
    cd $script_dir/build
    cmake ..
    make
    mv daai ..
    cd ..
fi
chmod 777 ./daai

if [ ! -x ./daai ]; then
echo "move to a location that can run daai"
exit
fi

model_dir=$script_dir/../models/ncnn/yolov5s.opt
img_dir=$script_dir/../models/resources

./daai $model_dir $img_dir