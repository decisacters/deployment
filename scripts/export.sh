script_dir="$(dirname "${BASH_SOURCE[0]}")"
script_dir="$(realpath "${script_dir}")"

tmp_dir=$script_dir/../.tmp_versions
for prefix in x86_64 # arm-ca9-linux-gnueabihf # arm-openwrt-linux-muslgnueabi # 
do
    if [[ $prefix == x86_64 ]]; then
        dst=$tmp_dir/export/$prefix
        mkdir -p $dst
        cp /usr/local/lib/libncnn.a $dst
        cp $script_dir/run.sh $dst
        cp $script_dir/CMakeLists.txt $dst
        cp $script_dir/../src/daai.h $dst
        cp $script_dir/../src/daai.cpp $dst
        cp $script_dir/../build/src/libDAAI.a $dst
    else
        mkdir -p $tmp_dir/export/$prefix-gnueabi
        cd $tmp_dir/export/$prefix-gnueabi

        compiler_dir=$tmp_dir/tools/$prefix-gnueabi
        bin_dir=$compiler_dir/bin
        libexec=$compiler_dir/libexec/gcc/$prefix/6.4.1/
        export PATH=$PATH:$bin_dir:$libexec
        chmod 777 $bin_dir/$prefix*
        chmod 777 $libexec/*

        export STAGING_DIR="" # arm-openwrt-linux

        readarray -d - -t parts <<< $prefix
        platform=${parts[1]}

        # CC=$bin_dir/$prefix-gcc CXX=$bin_dir/$prefix-g++ cmake $tmp_dir/../
        export LD_LIBRARY_PATH=$compiler_dir/lib:$LD_LIBRARY_PATH

        cmake_command="cmake $tmp_dir/../ -DOPENWRT=$compiler_dir"
        cmake_command+=" -DCMAKE_C_FLAGS=\"--pipe -march=armv7-a -mtune=cortex-a7 -mfpu=neon  -fno-caller-saves -Wno-unused-result -mfloat-abi=hard -fpermissive -ffunction-sections -fdata-sections  -fdiagnostics-color=auto -mthumb\""
        cmake_command+=" -DCMAKE_CXX_FLAGS=\"--pipe -march=armv7-a -mtune=cortex-a7 -mfpu=neon  -fno-caller-saves -Wno-unused-result -mfloat-abi=hard -fpermissive -ffunction-sections -fdata-sections  -fdiagnostics-color=auto -mthumb\""
        cmake_command+=" -DCMAKE_C_COMPILER=$prefix-gcc"
        cmake_command+=" -DCMAKE_CXX_COMPILER=$prefix-g++"
        # cmake_command+=" -DCMAKE_TOOLCHAIN_FILE=$arm_gnueabi_toolchain"
        eval $cmake_command
        make
    fi
    # if [[ $prefix == x86_64 ]]; then
    #     install_dir=/usr/local
    #     rm -f $install_dir/bin/*ncnn*
    #     cp $build_dir/install/bin/* $install_dir/bin
    # else
    #     install_dir=$compiler_dir
    # fi

    # rm -f $install_dir/lib/libncnn*
    # rm -rf $install_dir/include/ncnn

    # cp $build_dir/install/lib/libncnn* $install_dir/lib
    # cp -r $build_dir/install/include/ncnn $install_dir/include

done
