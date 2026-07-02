cmake_minimum_required(VERSION 3.20)

# The MJPC static libraries in the runtime build tree are clang-13 + LTO (LLVM
# bitcode archives), which GNU ld cannot read ("file format not recognized").
# When that build tree is present, compile + link this whole package with
# clang-13 to match the mjpc toolchain. Must run BEFORE project() picks a compiler.
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/home/code/mujoco_mpc/build")
  find_program(_H12_CLANGXX clang++-13)
  find_program(_H12_CLANG clang-13)
  if(_H12_CLANGXX AND _H12_CLANG)
    set(CMAKE_C_COMPILER "${_H12_CLANG}")
    set(CMAKE_CXX_COMPILER "${_H12_CLANGXX}")
  endif()
endif()

project(h12_deploy_mjpc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ---- Python estimator node (proprioceptive base estimator) ----
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS ${PROJECT_NAME}/estimator_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME estimator_node)

# ---- shared resources ----
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

# ---- C++ MJPC controller (embeds mjpc::Agent; auto-skipped where absent) ----
# The controller links the fork's static libs + headers directly from the warm
# build tree the deploy image keeps at runtime: docker-compose mounts
# container_cache/mjpc_build -> /home/code/mujoco_mpc/build and launch_ros.sh
# hydrates it from the baked /opt/mjpc-build-seed BEFORE colcon runs. So at build
# time here MJPC_BUILD holds libmjpc/threadpool/absl *.a + the fetched mujoco/absl
# headers under _deps. The SKIP guard keys off that build tree existing, so a dev
# checkout (no hydrated build) still builds the estimator.
option(H12_BUILD_CONTROLLER "Build the C++ MJPC controller (needs the mjpc build tree)" ON)
set(MJPC_ROOT "/home/code/mujoco_mpc" CACHE PATH "mujoco_mpc fork root (bind-mounted source)")
set(MJPC_BUILD "${MJPC_ROOT}/build" CACHE PATH "MJPC CMake build tree (hydrated at runtime)")

if(H12_BUILD_CONTROLLER AND EXISTS "${MJPC_BUILD}")
  find_package(rclcpp REQUIRED)
  find_package(unitree_hg REQUIRED)
  find_package(unitree_go REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(Threads REQUIRED)

  # Includes: mjpc source headers, plus the fetched mujoco + absl headers. mujoco
  # C++ headers reliably ship in the pip package (version-matched); fall back to the
  # build's _deps. absl headers come from the build's _deps (source + generated).
  set(MJPC_INCLUDE_DIRS "${MJPC_ROOT}")
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    PATHS /usr/local/lib/python3.10/dist-packages/mujoco/include NO_DEFAULT_PATH)
  if(NOT MUJOCO_INCLUDE_DIR)
    file(GLOB _mj_dep "${MJPC_BUILD}/_deps/mujoco-src/include")
    set(MUJOCO_INCLUDE_DIR ${_mj_dep})
  endif()
  file(GLOB _absl_src "${MJPC_BUILD}/_deps/abseil-cpp-src")
  file(GLOB _absl_bld "${MJPC_BUILD}/_deps/abseil-cpp-build")
  list(APPEND MJPC_INCLUDE_DIRS ${MUJOCO_INCLUDE_DIR} ${_absl_src} ${_absl_bld})

  # libmujoco.so (staged in /usr/local/lib by the image; also in the build tree).
  find_library(MUJOCO_LIBRARY mujoco PATHS /usr/local/lib "${MJPC_BUILD}/lib")
  if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "h12_deploy_mjpc: libmujoco not found "
                        "(set -DMUJOCO_LIBRARY=/path/to/libmujoco.so)")
  endif()

  # Link the mjpc library + threadpool + abseil (the upstream deploy's set), found
  # anywhere under the build tree, in --start-group so intra-mjpc / mjpc->absl
  # cross-references resolve. Narrowed from "all *.a" so lld's LTO codegen doesn't
  # drag in the grpc/protobuf/gtest bitcode the controller never references.
  file(GLOB_RECURSE _mjpc_core "${MJPC_BUILD}/libmjpc.a" "${MJPC_BUILD}/libthreadpool.a")
  file(GLOB_RECURSE _mjpc_absl "${MJPC_BUILD}/libabsl_*.a")
  set(MJPC_LIBRARIES ${_mjpc_core} ${_mjpc_absl})

  add_executable(mjpc_deploy_lowerbody_controller src/mjpc_deploy_lowerbody_controller.cpp)
  target_include_directories(mjpc_deploy_lowerbody_controller PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" ${MJPC_INCLUDE_DIRS})
  ament_target_dependencies(mjpc_deploy_lowerbody_controller rclcpp unitree_hg unitree_go std_srvs)
  target_link_libraries(mjpc_deploy_lowerbody_controller
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS})
  # lld (LTO-aware) is required to link the clang-13 bitcode archives above.
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_deploy_lowerbody_controller PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  set_target_properties(mjpc_deploy_lowerbody_controller PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib")
  install(TARGETS mjpc_deploy_lowerbody_controller DESTINATION lib/${PROJECT_NAME})
  message(STATUS "h12_deploy_mjpc: mjpc_deploy_lowerbody_controller ENABLED (MJPC build tree ${MJPC_BUILD})")
else()
  message(STATUS "h12_deploy_mjpc: mjpc_deploy_lowerbody_controller SKIPPED "
                 "(no MJPC build tree at '${MJPC_BUILD}'; run in the ros container after "
                 "launch_ros.sh hydrates it, or pass -DH12_BUILD_CONTROLLER=OFF). "
                 "Estimator + launch still build.")
endif()

ament_package()
