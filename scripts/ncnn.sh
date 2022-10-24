
if [ $# -ne 1 ]; then
    echo "Usage: ncnn.sh github_dir"
    exit
fi

github_dir=$1

tmp_dir=$github_dir/../../.tmp_versions

ncnn_dir=$github_dir/ncnn

# ncnn
# ${CMAKE_CURRENT_SOURCE_DIR}/.tmp_versions/libs/${dir}/lib/cmake/ncnn/ncnnConfig.cmake
# set(NCNN_VULKAN OFF)
# ${CMAKE_CURRENT_SOURCE_DIR}/.tmp_versions/libs/${dir}/lib/cmake/ncnn/ncnn.cmake
# INTERFACE_LINK_LIBRARIES "OpenMP::OpenMP_CXX;Threads::Threads"
# ${CMAKE_CURRENT_SOURCE_DIR}/.tmp_versions/libs/${dir}/include/ncnn/platform.h
# #define NCNN_VULKAN 0

for prefix in x86_64 # ndk-21.4.7075529 # arm-openwrt-linux-muslgnueabi # arm-ca9-linux-gnueabihf
do
    build_dir=$github_dir/ncnn/build/$prefix
    rm -rf $build_dir
    mkdir -p $build_dir
    cd $build_dir

    cmake_command="cmake $ncnn_dir -DNCNN_OPENMP=OFF"
    # cmake_command+=" -DNCNN_SIMPLEOCV=ON"
    if [[ $prefix == x86_64 ]]; then
        eval $cmake_command
    elif [[ $prefix == ndk* ]]; then
        export ANDROID_NDK=$tmp_dir/tools/$prefix
        android_gnueabi_toolchain=$ANDROID_NDK/build/cmake/android.toolchain.cmake
        cmake_command+=" -DANDROID_ABI=\"armeabi-v7a\""
        cmake_command+=" -DANDROID_ARM_NEON=ON"
        cmake_command+=" -DANDROID_PLATFORM=android-29"
        # cmake_command+=" -DNCNN_VULKAN=ON"
        echo $cmake_command
        eval $cmake_command
    else
        compiler_dir=$tmp_dir/tools/$prefix-gnueabi
        arm_gnueabi_toolchain=$ncnn_dir/toolchains/arm-linux-gnueabi.toolchain.cmake
        cmake_command+=" -DCMAKE_C_FLAGS=\"--pipe -march=armv7-a -mtune=cortex-a7 -mfpu=neon  -fno-caller-saves -Wno-unused-result -mfloat-abi=hard -fpermissive -ffunction-sections -fdata-sections  -fdiagnostics-color=auto -mthumb\""
        cmake_command+=" -DCMAKE_CXX_FLAGS=\"--pipe -march=armv7-a -mtune=cortex-a7 -mfpu=neon  -fno-caller-saves -Wno-unused-result -mfloat-abi=hard -fpermissive -ffunction-sections -fdata-sections  -fdiagnostics-color=auto -mthumb\""
        cmake_command+=" -DCMAKE_C_COMPILER=$bin_dir/$prefix-gcc"
        cmake_command+=" -DCMAKE_CXX_COMPILER=$bin_dir/$prefix-g++"
        cmake_command+=" -DCMAKE_TOOLCHAIN_FILE=$arm_gnueabi_toolchain"
        eval $cmake_command
    fi

    make -j$(nproc)
    cmake --install $build_dir
    make install

    if [[ $prefix == x86_64 ]]; then
        install_dir=/usr/local
        rm -f $install_dir/bin/*ncnn*
        cp $build_dir/install/bin/* $install_dir/bin
    else
        install_dir=$compiler_dir
    fi

    rm -f $install_dir/lib/libncnn*
    rm -rf $install_dir/include/ncnn

    cp $build_dir/install/lib/libncnn* $install_dir/lib
    cp -r $build_dir/install/include/ncnn $install_dir/include
done