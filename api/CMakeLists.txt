include_directories(${OpenCV_INCLUDE_DIRS})

add_library(embedded_bgsub_api STATIC)

target_sources(
    embedded_bgsub_api
        PRIVATE
            "src/VibeBGS.cpp" "include/VibeBGSUtils.hpp" "src/WeightedMovingVariance.cpp"
        PUBLIC
            "include/VibeBGS.hpp" "include/WeightedMovingVariance.hpp"
)

target_include_directories(
    embedded_bgsub_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include"
)

