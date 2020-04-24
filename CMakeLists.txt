cmake_minimum_required(VERSION 2.8)
#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
set(CMAKE_BUILD_TYPE "Release")
project(zhuangjiaban)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(X11 REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
	${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/anglesolver.h
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/armordector.h
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/kalmanfilter.h
        ${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/numeric_rm.h
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/opencv_extended.h
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/predictor.h
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/include/serial.h
)
	
set(SOURCE_FILES
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/src/opencv_extended.cpp
        ${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/src/numeric_rm.cpp
	${PROJECT_SOURCE_DIR}/home/jiing/zhuangjiaban/src/serial.cpp
        )
if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_executable(zhuangjiaban main.cpp)
target_link_libraries(zhuangjiaban 
		${OpenCV_LIBS}
		-lpthread
		-lMVSDK
		-lm
)
