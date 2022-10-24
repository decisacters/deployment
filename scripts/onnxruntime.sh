cp /home/decisacter/deployment/.tmp_versions/onnxruntime-linux-x64-gpu-1.11.1/lib/libonnxruntime* /usr/local/lib
cp /home/decisacter/deployment/.tmp_versions/onnxruntime-linux-x64-gpu-1.11.1/include/onnxruntime* /usr/local/include

rm /usr/local/lib/libonnx*
rm /usr/local/include/onnxruntime*

# ${CMAKE_CURRENT_SOURCE_DIR}/.tmp_versions/libs/${dir}/lib/cmake/ONNXConfig.cmake
add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_LIST_DIR}/../libonnxruntime.so.1.11.1
)