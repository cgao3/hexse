file(GLOB nnevaluate_SRC *.cpp *.h *.hpp)
include_directories(${tensorflow_root}/
        ${tensorflow_root}/third_party/eigen-eigen-install/
        ${tensorflow_root}/bazel-genfiles
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util)

add_executable(nntest ${nnevaluate_SRC} Test.cxx)
add_library(nnevaluate STATIC ${nnevaluate_SRC})

target_link_libraries(nnevaluate
        protobuf
        tensorflow_cc
        pthread
        )

target_link_libraries(
        nntest
        protobuf
        tensorflow_cc
        pthread
        hex
        util
)