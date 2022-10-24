# ${CMAKE_CURRENT_SOURCE_DIR}/.tmp_versions/libs/${dir}/ONNXConfig.cmake
add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
if(WIN32)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_LIST_DIR}/lib/onnxruntime.dll
)
elseif(UNIX)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_LIST_DIR}/lib/libonnxruntime.so.1.11.1
)
endif()