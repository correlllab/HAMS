cmake_minimum_required(VERSION 3.20)

# MJPC deploy layer for the Unitree H1-2 (rebuilt on the mjpc_reintegration
# branch against pristine google-deepmind upstream mjpc).
#
# Integration architecture: DIRECT C++ LINK. This package consumes the
# standalone MJPC CMake build tree (built by docker/scripts/rebuild_mjpc.sh
# into /home/code/mujoco_mpc/build, bind-mounted from
# container_cache/mjpc_build) — libmjpc.a + threadpool + abseil archives + the
# build's own libmujoco.so — so future deploy nodes can embed mjpc::Agent
# in-process. The fork's deploy layer already proved the alternative (gRPC
# agent_server + python client) starves the planner with per-tick round trips;
# rebuild_mjpc.sh still builds agent_server, so switching to the gRPC client
# later only means dropping this file's C++ section and talking to
# build/bin/agent_server instead.
#
# When the mjpc build tree hasn't been built yet (fresh host, or an
# interrupted rebuild_mjpc.sh) the C++ section skips itself with a warning so
# the launch-time colcon build stays green. Detection uses CONFIGURE_DEPENDS
# globs on the mjpc build artifacts, so once rebuild_mjpc.sh has run, the next
# colcon build reconfigures this package automatically — no wipe needed.

option(H12_BUILD_MJPC_CPP "Build the C++ targets that link the mjpc build tree" ON)
set(MJPC_ROOT "/home/code/mujoco_mpc" CACHE PATH "mujoco_mpc submodule root (bind-mounted source)")
set(MJPC_BUILD "${MJPC_ROOT}/build" CACHE PATH "MJPC CMake build tree (container_cache/mjpc_build mount)")

# Upstream mjpc leaves CMAKE_INTERPROCEDURAL_OPTIMIZATION ON for Release, so
# its static libs are clang-13 LTO bitcode that GNU ld/gcc cannot read. Pin the
# whole package to clang-13 UNCONDITIONALLY (and before project()) — not just
# when the mjpc tree already exists — so a package first configured on a fresh
# host (mjpc not built yet) never caches gcc and then fails at link once the
# tree appears. NOTE: CMake silently ignores a compiler change on an
# already-configured build dir, so a core_ws/build/h12_deploy_mjpc that
# predates this pin can be stuck on gcc — the compiler guard below detects
# that and soft-skips with instructions.
if(NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(_H12_CLANGXX clang++-13)
  find_program(_H12_CLANG clang-13)
  if(_H12_CLANGXX AND _H12_CLANG)
    set(CMAKE_C_COMPILER "${_H12_CLANG}")
    set(CMAKE_CXX_COMPILER "${_H12_CLANGXX}")
  endif()
endif()

project(h12_deploy_mjpc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Empty Python package kept as the seam for the deploy nodes to come.
ament_python_install_package(${PROJECT_NAME})

# Detect the mjpc build by its ARTIFACTS (not its CMakeCache): a
# half-configured or in-flight mjpc build stays in the soft-skip path instead
# of a hard configure error. CONFIGURE_DEPENDS makes the generated build
# system re-run cmake whenever these glob results change (libs appearing after
# rebuild_mjpc.sh, or the archive set changing after a dep bump + cache wipe).
file(GLOB _libmjpc CONFIGURE_DEPENDS "${MJPC_BUILD}/lib/libmjpc.a")
file(GLOB _libmujoco CONFIGURE_DEPENDS "${MJPC_BUILD}/lib/libmujoco.so")
file(GLOB _mjpc_support CONFIGURE_DEPENDS
  "${MJPC_BUILD}/lib/libthreadpool.a"
  "${MJPC_BUILD}/lib/liblodepng.a"
  "${MJPC_BUILD}/lib/libglfw3.a")
file(GLOB _mjpc_absl CONFIGURE_DEPENDS "${MJPC_BUILD}/lib/libabsl_*.a")

set(_mjpc_ready FALSE)
if(H12_BUILD_MJPC_CPP AND _libmjpc AND _libmujoco AND _mjpc_absl
   AND EXISTS "${MJPC_BUILD}/_deps/mujoco-src/include/mujoco/mujoco.h")
  set(_mjpc_ready TRUE)
endif()

if(_mjpc_ready AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Stale build dir from before the unconditional clang pin (or an image
  # without clang-13): linking the LTO bitcode with ${CMAKE_CXX_COMPILER_ID}
  # would die cryptically at link time. Skip, stay green, say how to fix.
  message(WARNING
    "h12_deploy_mjpc: mjpc build tree found, but this package's cached "
    "compiler is ${CMAKE_CXX_COMPILER_ID} — the clang-13-LTO mjpc archives "
    "need clang-13 + lld. C++ targets skipped. Fix (one-time):\n"
    "  rm -rf /home/code/core_ws/build/h12_deploy_mjpc && "
    "colcon build --packages-select h12_deploy_mjpc")
  set(_mjpc_ready FALSE)
endif()

if(_mjpc_ready)
  find_package(Threads REQUIRED)

  # Headers: the mjpc source root plus the PINNED deps from the build tree.
  # ABI-critical: mujoco headers MUST come from build/_deps/mujoco-src (the
  # exact commit the libs were compiled from), never from the pip mujoco wheel
  # — a different 3.2.3-era commit skews mjModel/mjData member offsets and
  # silently corrupts field access (this bit the old deploy stack).
  set(MJPC_INCLUDE_DIRS
    "${MJPC_ROOT}"
    "${MJPC_BUILD}/_deps/mujoco-src/include"
    "${MJPC_BUILD}/_deps/abseil-cpp-src"
    "${MJPC_BUILD}/_deps/abseil-cpp-build")
  # nlohmann json is header-only and reachable through some task headers.
  if(EXISTS "${MJPC_BUILD}/_deps/json-src/include")
    list(APPEND MJPC_INCLUDE_DIRS "${MJPC_BUILD}/_deps/json-src/include")
  endif()

  # Infrastructure smoke test: registry lookup + model load through libmjpc
  # (see src/mjpc_smoketest.cc). Future deploy nodes reuse exactly this
  # include/link recipe. Static archives are linked as one --start-group so
  # cross-references resolve regardless of order; the set is deliberately
  # narrowed (not lib*.a) so lld's LTO codegen never chews through the
  # grpc/protobuf/gtest archives these targets don't need. The build's own
  # libmujoco.so comes from the glob above — never from pip.
  add_executable(mjpc_smoketest src/mjpc_smoketest.cc)
  target_include_directories(mjpc_smoketest PRIVATE ${MJPC_INCLUDE_DIRS})
  target_compile_definitions(mjpc_smoketest PRIVATE
    MJSIMULATE_STATIC
    H12_DEFAULT_MJPC_TASKS_DIR="${MJPC_BUILD}/mjpc/tasks")
  target_link_libraries(mjpc_smoketest
    -Wl,--start-group ${_libmjpc} ${_mjpc_support} ${_mjpc_absl} -Wl,--end-group
    ${_libmujoco} Threads::Threads ${CMAKE_DL_LIBS})
  # lld is required to link the LTO bitcode archives. --disable-new-dtags
  # forces DT_RPATH (which beats LD_LIBRARY_PATH), so the pip mujoco wheel's
  # same-soname, different-ABI libmujoco.so.3.2.3 can never be substituted at
  # run time.
  if(EXISTS "/usr/bin/ld.lld-13")
    target_link_options(mjpc_smoketest PRIVATE
      -fuse-ld=/usr/bin/ld.lld-13 -Wl,--disable-new-dtags)
  endif()
  set_target_properties(mjpc_smoketest PROPERTIES
    BUILD_RPATH "${MJPC_BUILD}/lib"
    INSTALL_RPATH "${MJPC_BUILD}/lib")
  install(TARGETS mjpc_smoketest DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING
    "h12_deploy_mjpc: mjpc build artifacts not found under ${MJPC_BUILD}/lib — "
    "C++ targets skipped (python package still installs). Run "
    "/home/code/h12_sim_scripts/rebuild_mjpc.sh, then rebuild this package "
    "(`colcon build --packages-select h12_deploy_mjpc` — it reconfigures "
    "itself automatically once the artifacts exist).")
endif()

ament_package()
