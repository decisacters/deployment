# apt update
# apt install build-essential gdb \
#             gcc-arm-linux-gnueabi gcc-arm-linux-gnueabihf \
#             g++-arm-linux-gnueabi g++-arm-linux-gnueabihf -y

script_dir="$(dirname "${BASH_SOURCE[0]}")"
script_dir="$(realpath "${script_dir}")"

github_dir="$script_dir/../.tmp_versions/.github"

if [ ! -d $github_dir ]; then
    mkdir $github_dir
fi

# pull_repo github_dir repo_name repo_url
source $script_dir/functions.sh

# pull_repo $github_dir https://github.com/DefTruth/lite.ai.toolkit

# pull_repo $github_dir https://github.com/nothings/stb
# cp $github_dir/stb/stb_image.h /usr/include
# cp $github_dir/stb/stb_image_write.h /usr/include

# pull_repo $github_dir https://github.com/opencv/opencv
bash $script_dir/opencv.sh $github_dir

# pull_repo $github_dir https://github.com/tencent/ncnn
# bash $script_dir/ncnn.sh $github_dir


# # docker run -it --cap-add sys_ptrace --ipc host -w /home --gpus all --name decisacter_deployment -v /home://wsl.localhost/Ubuntu/home/decisacter/deploy ultralytics/yolov5