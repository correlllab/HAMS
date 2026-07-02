// ControllerNode: MJPC embedded lower-body control node for the Unitree H1-2.
//
// Class declaration only -- see mjpc_deploy_lowerbody_controller.cpp for the
// implementation and the full port notes (audit fixes H1/H2/M4/M5, the bring-up
// ramp, etc.).
//
// ROS interface:
//   sub  /lowstate                    (unitree_hg/LowState)
//   sub  /h12_deploy_mjpc/sportstate_est (unitree_go/SportModeState, from estimator_node)
//   pub  /safety/lowcmd_lower_in      (unitree_hg/LowCmd) -> h12_safety_layer
//
// Strategy is HARDCODED to the stable stand (6); planning is SYNCHRONOUS in the
// control callback (no async planner thread, hence no data race).
#ifndef H12_DEPLOY_MJPC_MJPC_DEPLOY_LOWERBODY_CONTROLLER_HPP_
#define H12_DEPLOY_MJPC_MJPC_DEPLOY_LOWERBODY_CONTROLLER_HPP_

#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/threadpool.h"

#include <rclcpp/rclcpp.hpp>
#include "unitree_go/msg/sport_mode_state.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

#include "mjpc_glue.hpp"

using LowState = unitree_hg::msg::LowState;
using LowCmd = unitree_hg::msg::LowCmd;
using SportModeState = unitree_go::msg::SportModeState;
using h12::kNU;

// ---- hardcoded operating config (were upstream CLI flags) ----
inline constexpr char kTaskId[] = "Stabilize H12 Magpie";  // lower-body nu=12 stabilize task
inline constexpr int kStrategy = 6;                          // stable stand (no live switch)
inline constexpr double kCtrlHz = 200.0;
inline constexpr double kWarmupSec = 1.0;
inline constexpr double kStartRampSec = 5.0;   // measured -> stance ramp on all leg joints
inline constexpr double kRampHoldSec = 3.0;    // hold the stance scripted while CEM converges
inline constexpr double kPolicyBlendSec = 2.0; // ease scripted stance -> live policy target
inline constexpr double kStaleSec = 0.05;      // H1 watchdog: state older than this -> safe-hold
inline constexpr float kSafeHoldKd = 2.0f;     // damping-stop kd on safe-hold (kp=0, tau=0)

class ControllerNode : public rclcpp::Node {
 public:
  ControllerNode();
  ~ControllerNode() override;

 private:
  void InitMjpc();
  // Fill the planner state from the latest LowState (joints + IMU) + SportModeState
  // (IMU-site base pose/vel), backing the site pose out to the pelvis.
  void FillState(const LowState& ls, const SportModeState& sp);
  void EmitSafeHold();
  void ControlStep();
  // Self-contained standing demo: drop the sim's elastic support tether once the
  // controller has been running for band_drop_after_secs (fixed dwell timer).
  void MaybeReleaseBand(double wall);
  // M4: grade torque headroom against TAU_ESTOP (the actual trip point).
  void MaybeLogStatus(double wall, bool warming, const LowState& ls,
                      const double* tgt_q, const double* tau, int nmotor);

  // params
  double gff_ = 0.85;
  double plan_rate_hz_ = 80.0;
  double imu_pitch_off_ = 0.0;
  double imu_roll_off_ = 0.0;
  double clamp_ratio_ = 0.9;
  // controller-managed elastic-band drop (sim standing demo)
  bool drop_band_ = true;
  double band_drop_after_secs_ = 20.0;  // fixed dwell: drop the tether this long after t0
  bool band_released_ = false;
  long band_ticks_ = 0;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr band_cli_;
  // mjpc scratch
  int nq_ = 0, nv_ = 0, nu_ = 0, nact_ = 0, home_key_ = -1;
  mjData* sd_ = nullptr;
  mjData* gd_ = nullptr;
  std::unique_ptr<mjpc::ThreadPool> plan_pool_;
  std::vector<double> action_;
  double home_q_[kNU] = {0};
  double q_init_[kNU] = {0};
  double base_quat_[4] = {1, 0, 0, 0};  // planner base orientation, stashed by FillState
  double base_pos_z_ = 0.0;             // pelvis world z, stashed by FillState
  bool init_set_ = false;
  double ramp_eff_ = kStartRampSec;
  double plan_accum_ = 0.0;
  // loop state
  rclcpp::Time t0_;
  bool have_t0_ = false;
  bool stale_warned_ = false;
  long ticks_ = 0;
  // latest inputs (guarded by state_mu_; the subscription group is reentrant)
  std::mutex state_mu_;
  LowState::SharedPtr last_ls_;
  SportModeState::SharedPtr last_base_;
  rclcpp::Time last_ls_time_;
  rclcpp::Time last_base_time_;
  // ros i/o
  rclcpp::CallbackGroup::SharedPtr sub_group_;
  rclcpp::Subscription<LowState>::SharedPtr ls_sub_;
  rclcpp::Subscription<SportModeState>::SharedPtr base_sub_;
  rclcpp::Publisher<LowCmd>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

#endif  // H12_DEPLOY_MJPC_MJPC_DEPLOY_LOWERBODY_CONTROLLER_HPP_
