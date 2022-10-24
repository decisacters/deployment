
if [ $# -ne 1 ]; then
    echo "Usage: opencv.sh github_dir"
    exit
fi

github_dir=$1

tmp_dir=$github_dir/../../.tmp_versions

opencv_dir=$github_dir/opencv

for prefix in x86_64 # arm-openwrt-linux-muslgnueabi # arm-ca9-linux-gnueabihf # 
do
    build_dir=$github_dir/opencv/bin/$prefix

    if [[ $prefix == x86_64 ]]; then
        install_dir=/usr/local
    else
        compiler_dir=$tmp_dir/tools/$prefix-gnueabi
        bin_dir=$compiler_dir/bin
        install_dir=$compiler_dir
    fi

    libs=$build_dir/lib/libopencv*

    if [[ "`echo $libs`" == "$libs" ]]; then

        mkdir -p $build_dir
        cd $build_dir

        cmake_command="cmake $tmp_dir/.github/opencv -DBUILD_LIST=core"

        cmake_command+="imgproc,highgui,imgcodecs,videoio"

        if [[ $prefix == x86_64 ]]; then
            eval $cmake_command
        else
               
            # libs_3rd=$build_dir/3rdparty/lib/*
            # arm_gnueabi_toolchain=$tmp_dir/.github/opencv/platforms/linux/arm-gnueabi.toolchain.cmake
    
            # if [[ "`echo $libs_3rd`" == "$libs_3rd" ]]; then
            #     cmake_command+=" -DCMAKE_TOOLCHAIN_FILE=$arm_gnueabi_toolchain"
            #     eval $cmake_command
            #     make -j$(nproc)
            # fi
            
            # arm_toolchain=$tmp_dir/.github/opencv/platforms/linux/arm.toolchain.cmake
            # sed -i "s/arm-linux-gnueabi/$prefix/" $arm_gnueabi_toolchain
            # sed -i "s/mthumb/mthumb -march=armv7-a/" $arm_toolchain

            # libexec=$compiler_dir/libexec/gcc/$prefix/6.4.1/
            # export PATH=$PATH:$bin_dir:
            # export STAGING_DIR=""
            # chmod +x 777 $bin_dir/$prefix*
            # chmod +x 777 $libexec/*
            
            # cmake_command+=" -DCMAKE_C_FLAGS=\"-march=armv7-a\""
            # cmake_command+=" -DCMAKE_CXX_FLAGS=\"-march=armv7-a\""
            
            # cmake_command+=" -DCMAKE_SYSTEM_PROCESSOR=arm"
            # cmake_command+=" -DCMAKE_C_COMPILER=$bin_dir/$prefix-gcc"
            # cmake_command+=" -DCMAKE_CXX_COMPILER=$bin_dir/$prefix-g++"

            # cmake_command="CC=$bin_dir/$prefix-gcc CXX=$bin_dir/$prefix-g++ $cmake_command"
            # cmake_command="CC=$prefix-gcc CXX=$prefix-g++ $cmake_command"

            cmake_command+=" -DCMAKE_C_COMPILER=$prefix-gcc"
            cmake_command+=" -DCMAKE_CXX_COMPILER=$prefix-g++"
            cmake_command+=" -DCMAKE_TOOLCHAIN_FILE=$arm_gnueabi_toolchain"
            eval $cmake_command
        fi

        make -j$(nproc)
        make install
    fi

    rm -f $install_dir/lib/libopencv*
    rm -rf $install_dir/include/opencv2

    cp $libs $install_dir/lib
    cp -r $build_dir/include/opencv4/opencv2 $install_dir/include
done

# # https://zhuanlan.zhihu.com/p/85758253
# os.system('apt update')
# os.system('apt install gcc-arm-linux-gnueabi gcc-arm-linux-gnueabih ffmpeg -y')
# # apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev

# # build ffmpeg (optional) for video capture
# os.system(f'git clone https://github.com.cnpmjs.org/FFmpeg/FFmpeg.git')
# os.system(f'mkdir -p {download_dir}/FFmpeg/build')
# os.system(f'cd {download_dir}/FFmpeg')

# os.system(f'./configure --enable-shared --disable-static -prefix=/usr/local/ffmpeg --cross-prefix=arm-openwrt-linux-muslgnueabi- --arch=armv7l --target-os=linux')
# os.system('make -j$(nproc)')
# os.system('/etc/ld.so.conf.d/ffmpeg.conf')
# os.system('make install')

# # sudo vim /etc/ld.so.conf.d/ffmpeg.conf add:/usr/local/ffmpeg/lib
# # sudo ldconfig
# # sudo vim /etc/profile append export PATH=$PATH:/usr/local/ffmpeg/bin
# # source /etc/profile
# os.system(f'export LD_LIBRARY_PATH={download_dir}/FFmpeg/build/lib:${{LD_LIBRARY_PATH}}')
# os.system(f'export C_INCLUDE_PATH={download_dir}/FFmpeg/build/include:${{C_INCLUDE_PATH}}')
# os.system(f'export CPLUS_INCLUDE_PATH={download_dir}/FFmpeg/build/include:${{CPLUS_INCLUDE_PATH}}')

# os.system('cmake -D CMAKE_TOOLCHAIN_FILE=../openwrt-opencv.toolchain.cmake -D WITH_FFMPEG=ON ../../..') # -D WITH_FFMPEG=ON -D BUILD_opencv_world=ON
# os.system('make -j$(nproc)')

# os.system('make install')
# os.system('ldconfig')