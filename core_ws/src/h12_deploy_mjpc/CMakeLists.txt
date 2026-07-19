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
  # Match the fork's toolchain (cmake/MpcOptions.cmake sets C++20): the
  # controller compiles the fork's deploy_common.cc against mjpc headers.
  set(CMAKE_CXX_STANDARD 20)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ---- Python nodes: estimator shell + controller launcher ----
# Both are rclpy PARAM SHELLS around fork cores. The controller core runs as a
# SEPARATE process (see controller_launcher.py: rclcpp's rmw_cyclonedds and
# unitree_sdk2's bundled CycloneDDS both export libddsc/libddscxx.so.0 -- one
# process loading both corrupts the heap, so rclcpp never shares a process
# with the raw-DDS control loop).
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS ${PROJECT_NAME}/estimator_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME estimator_node)
install(PROGRAMS ${PROJECT_NAME}/controller_launcher.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME mjpc_deploy_lowerbody_controller)
# Same launcher as above but with the upper-body pause toggle
# (mjpc_deploy/toggle_pause_upperbody Trigger service) and the dynamic startup
# handshake with frame_task. Distinct node/executable so it can be selected in
# place of the plain lower-body controller.
install(PROGRAMS ${PROJECT_NAME}/split_body_controller.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME mjpc_deploy_splitbody_controller)
# Debug-only: renders the measured robot + a blue ghost of the planner's current
# best trajectory (rt/mjpc/plan) to an mp4. Same param-shell-around-a-fork-worker
# arrangement as the estimator.
install(PROGRAMS ${PROJECT_NAME}/debug_visualizer_node.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME mjpc_debug_visualizer)

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

# unitree_sdk2 C++ (raw-DDS ChannelFactory/Publisher/Subscriber used by the
# fork's deploy core). /opt/unitree_install is the container_cache mount that
# launch_ros.sh hydrates (prebuilt libs, SHA-pinned to the fork's host build).
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_install" "$ENV{HOME}/unitree_install")
find_package(unitree_sdk2 QUIET)

if(H12_BUILD_CONTROLLER AND EXISTS "${MJPC_BUILD}" AND unitree_sdk2_FOUND)
  find_package(Threads REQUIRED)

  # Includes: mjpc source headers, plus the fetched mujoco + absl headers.
  # ABI CRITICAL (found 2026-07-07): the controller MUST compile against the
  # SAME mujoco headers the build tree's libmujoco.so + libmjpc.a were built
  # with -- the _deps/mujoco-src pinned commit. The pip package reports the
  # same release (3.2.3) but its mjModel/mjData struct layout differs from the
  # pinned commit (extra flex_*/eq_objtype/nC fields before the actuator
  # section), so compiling THIS package's translation units against the pip
  # headers while linking the build-tree libs skews every direct
  # mjModel/mjData member access: PatchActuators wrote kp/kv into actuator_gear
  # of the PLANNER model (gear 1 -> 200), the CEM optimum on a frozen stand
  # state became the straight-knee jackknife (hipP +0.09 vs the agent server's
  # -0.20 on a bit-identical state), and every DDS handoff fell 1-4 s after
  # release. Prefer the _deps headers; the pip headers are only a FALLBACK for
  # dev checkouts without a hydrated build tree.
  set(MJPC_INCLUDE_DIRS "${MJPC_ROOT}")
  file(GLOB _mj_dep "${MJPC_BUILD}/_deps/mujoco-src/include")
  if(_mj_dep)
    set(MUJOCO_INCLUDE_DIR ${_mj_dep})
  else()
    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
      PATHS /usr/local/lib/python3.10/dist-packages/mujoco/include NO_DEFAULT_PATH)
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

  # The controller core = the fork's OWN CLI thin main + shared deploy core,
  # compiled STRAIGHT FROM THE SUBMODULE (no copy; the fork stays the single
  # source of truth for tables, flags, control loop, DDS I/O, safety behavior).
  # Deliberately NOT an rclcpp target -- see the CycloneDDS soname-collision
  # note above; the mjpc_deploy_lowerbody_controller entry point is the rclpy
  # launcher installed above, which execs this binary.
  add_executable(mjpc_lowerbody_core
    "${MJPC_ROOT}/mjpc/deploy/h12_lower_body_controller.cc"
    "${MJPC_ROOT}/mjpc/deploy/deploy_common.cc")
  target_include_directories(mjpc_lowerbody_core PRIVATE ${MJPC_INCLUDE_DIRS})
  target_link_libraries(mjpc_lowerbody_core
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
  # lld (LTO-aware) is required to link the clang-13 bitcode archives above.
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_lowerbody_core PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  # unitree_sdk2's bundled CycloneDDS .so lives next to the SDK; bake its dir
  # into the rpath so libddsc/libddscxx resolve at runtime regardless of
  # LD_LIBRARY_PATH (same arrangement as the fork's own deploy build).
  get_target_property(_ddsc_loc ddsc IMPORTED_LOCATION)
  get_filename_component(_dds_libdir "${_ddsc_loc}" DIRECTORY)
  set_target_properties(mjpc_lowerbody_core PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}")
  install(TARGETS mjpc_lowerbody_core DESTINATION lib/${PROJECT_NAME})

  # FULL-BODY core (h12_control_node.cc, nu=27, Lean task) -- same deploy_common
  # + link recipe as the lower-body core, just the other thin main. Lets the
  # full-body controller run through the SAME in-container DDS/estimator path as
  # the lower body (host-side chains hit host<->container DDS state-feed stalls).
  # Publishes the FULL channel rt/safety/lowcmd_in -> needs safety in full mode.
  add_executable(mjpc_fullbody_core
    "${MJPC_ROOT}/mjpc/deploy/h12_control_node.cc"
    "${MJPC_ROOT}/mjpc/deploy/deploy_common.cc")
  target_include_directories(mjpc_fullbody_core PRIVATE ${MJPC_INCLUDE_DIRS})
  target_link_libraries(mjpc_fullbody_core
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_fullbody_core PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  set_target_properties(mjpc_fullbody_core PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}")
  install(TARGETS mjpc_fullbody_core DESTINATION lib/${PROJECT_NAME})

  # SPLIT-BODY core (h12_split_controller.cc) -- the core the split-body launcher
  # (mjpc_deploy_splitbody_controller / split_body_controller.py) starts. Thin-main
  # scaffolding copied from the lower-body core (same deploy_common + link recipe),
  # but a WHOLE-BODY nu=27 controller (default task "Lean H12 Magpie Split") that
  # publishes BOTH split safety channels: legs on rt/safety/lowcmd_lower_in, arms
  # on rt/safety/lowcmd_upper_in. Kept a distinct target/binary so split-specific
  # C++ changes never touch the plain lower-body core.
  add_executable(mjpc_split_core
    "${MJPC_ROOT}/mjpc/deploy/h12_split_controller.cc"
    "${MJPC_ROOT}/mjpc/deploy/deploy_common.cc")
  target_include_directories(mjpc_split_core PRIVATE ${MJPC_INCLUDE_DIRS})
  target_link_libraries(mjpc_split_core
    -Wl,--start-group ${MJPC_LIBRARIES} -Wl,--end-group
    "${MUJOCO_LIBRARY}" unitree_sdk2 Threads::Threads ${CMAKE_DL_LIBS})
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_split_core PRIVATE -fuse-ld=/usr/bin/ld.lld-13)
  endif()
  set_target_properties(mjpc_split_core PROPERTIES
    BUILD_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}"
    INSTALL_RPATH "/usr/local/lib;${MJPC_BUILD}/lib;${_dds_libdir}")
  install(TARGETS mjpc_split_core DESTINATION lib/${PROJECT_NAME})

  message(STATUS "h12_deploy_mjpc: mjpc_lowerbody_core + mjpc_fullbody_core + mjpc_split_core ENABLED "
                 "(MJPC build tree ${MJPC_BUILD}; unitree_sdk2 dds at ${_dds_libdir})")
else()
  message(STATUS "h12_deploy_mjpc: mjpc_lowerbody_core SKIPPED "
                 "(needs the MJPC build tree at '${MJPC_BUILD}' AND unitree_sdk2 "
                 "[found=${unitree_sdk2_FOUND}] -- run in the ros container after "
                 "launch_ros.sh hydrates both, or pass -DH12_BUILD_CONTROLLER=OFF). "
                 "Estimator + launcher still build.")
endif()

ament_package()
