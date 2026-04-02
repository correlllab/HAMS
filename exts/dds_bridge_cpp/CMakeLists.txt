cmake_minimum_required(VERSION 3.12)
project(dds_bridge_cpp CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(dds_bridge_cpp
  src/main.cpp
)

# target_link_libraries(CycloneDDS-CXX)
target_include_directories(dds_bridge_cpp PUBLIC
  /home/mateo/cyclonedds-cxx/src/ddscxx/include
  /home/mateo/cyclonedds/build/src/core/include
  /home/mateo/cyclonedds-cxx/build/src/ddscxx/include
  
)
