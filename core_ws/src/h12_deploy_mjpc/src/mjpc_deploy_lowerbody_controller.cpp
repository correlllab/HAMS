// MJPC embedded lower-body control node for the Unitree H1-2, ROS 2 port.
//
// Ports mjpc/deploy/h12_lower_body_controller.cc (badinkajink/mujoco_mpc
// @extended_hw) into a rclcpp node that fits the HAMS stack:
//   * transport is ROS topics with unitree_hg messages (not raw CycloneDDS):
//       sub  /lowstate                    (unitree_hg/LowState)
//       sub  /h12_deploy_mjpc/sportstate_est (unitree_go/SportModeState, from estimator_node)
//       pub  /safety/lowcmd_lower_in      (unitree_hg/LowCmd) -> h12_safety_layer
//   * the safety layer computes the outgoing CRC, so this node emits none.
//   * the estimator publishes the IMU-site base pose/velocity (SportModeState); this
//     node backs it out to the pelvis using the /lowstate IMU quaternion + IMU_OFFSET
//     (orientation + gyro also come from /lowstate). Velocity is published directly,
//     so no finite-diff / twin_dt machinery is needed.
//
// The validated MJPC core is preserved: Agent init, GetTasks(), the residual
// sensor callback, per-strategy PlannerNumericOverrides, Task::Transition each
// tick, PatchActuators (gains + estop forceranges), gravity feedforward, and the
// bring-up ramp -> hold -> policy-blend handoff.
//
// Strategy is HARDCODED to the stable stand (6); there is no live switch.
// Planning is SYNCHRONOUS in the (single-threaded) control callback -- the audit
// mode that holds balance; there is no async planner thread, hence no data race.
//
// Audit fixes folded in:
//   H1  input-freshness watchdog -> safe-hold (damping stop) on stale state.
//   H2  target clamp bounds the FULL commanded torque (tau_ff + KP*e + KV*dq).
//   M4  torque headroom graded against TAU_ESTOP (the real trip point).
//   M5  MuJoCo error handler emits a safe-hold command before terminating; it
//       never leaves the robot latched at a stale target via a bare std::exit.

#include "mjpc_deploy_lowerbody_controller.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

using h12::KP;
using h12::KV;
using h12::TAU_ESTOP;
using h12::TAU_LIMIT;
using h12::JOINT_NAMES;

namespace {

// Globals for the MuJoCo sensor callback (mirror grpc/agent_service.cc). A single
// controller instance owns these; set once at construction.
mjpc::Agent g_agent;
const mjModel* g_agent_model = nullptr;
mjModel* g_model = nullptr;
mjpc::Task* g_task = nullptr;

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  if ((m == g_agent_model || m == g_model) && stage == mjSTAGE_ACC) {
    g_task->Residual(m, d, d->sensordata);
  }
}

// Patch the loaded model's <position> actuators to the node's authoritative gains
// (single source of truth; keeps planner + node KP/KV == twin PD). Leg forceranges
// are left at the model default (tightening them regressed the hold upstream).
void PatchActuators(mjModel* m) {
  for (int i = 0; i < kNU && i < m->nu; i++) {
    m->actuator_gainprm[i * mjNGAIN + 0] = KP[i];
    m->actuator_biasprm[i * mjNBIAS + 1] = -KP[i];
    m->actuator_biasprm[i * mjNBIAS + 2] = -KV[i];
  }
}

// M5: MuJoCo fatal-error handler. Rather than a bare std::exit that abandons the
// robot at its last latched command, publish a damping safe-hold first, then
// terminate. (Threads cannot be joined from inside mju_error, so we still exit --
// but the robot is left in a damped stop, not driving a stale target.)
std::function<void()> g_emit_safe_hold;  // set by the node ctor
void FatalMjuError(const char* msg) {
  std::fprintf(stderr, "[node] [mju_error] %s -- emitting safe-hold then aborting\n", msg);
  if (g_emit_safe_hold) g_emit_safe_hold();
  std::fflush(stderr);
  std::_Exit(1);
}
void LogMjuWarning(const char* msg) { std::fprintf(stderr, "[node] [mju_warning] %s\n", msg); }

}  // namespace

ControllerNode::ControllerNode() : rclcpp::Node("mjpc_deploy_lowerbody_controller") {
  gff_ = declare_parameter<double>("gravity_ff", 0.85);
  plan_rate_hz_ = declare_parameter<double>("plan_rate_hz", 80.0);
  imu_pitch_off_ = declare_parameter<double>("imu_pitch_offset_deg", 0.0) * M_PI / 180.0;
  imu_roll_off_ = declare_parameter<double>("imu_roll_offset_deg", 0.0) * M_PI / 180.0;
  // torque-clamp headroom as a fraction of the estop. 0.9 = the safe real-robot
  // default; the sim safety layer trips at 2.0x URDF, so a looser ratio (e.g.
  // 1.8) lets the legs use the balance authority the real estop would deny.
  clamp_ratio_ = declare_parameter<double>("clamp_ratio", 0.9);
  // Self-contained standing demo: band_drop_after_secs after the controller starts
  // (t0), drop the sim's elastic support tether via /elastic_band/toggle (once).
  // Disable on the real robot (no band); if the service is absent it simply warns
  // and never fires.
  drop_band_ = declare_parameter<bool>("drop_band", true);
  band_drop_after_secs_ = declare_parameter<double>("band_drop_after_secs", 20.0);
  const std::string lowstate_topic = declare_parameter<std::string>("lowstate_topic", "/lowstate");
  const std::string base_topic =
      declare_parameter<std::string>("base_state_topic", "/h12_deploy_mjpc/sportstate_est");
  const std::string cmd_topic =
      declare_parameter<std::string>("lowcmd_topic", "/safety/lowcmd_lower_in");

  mju_user_error = FatalMjuError;    // headless: emit safe-hold + abort, never block on getchar
  mju_user_warning = LogMjuWarning;

  InitMjpc();

  g_emit_safe_hold = [this]() { EmitSafeHold(); };

  // Subscriptions run in a REENTRANT group so /lowstate + /sportstate_est keep being
  // received even while ControlStep is mid-plan; the timer stays in its own
  // (mutually-exclusive) group so ControlStep never overlaps itself. state_mu_
  // guards the shared latest-state pointers across the groups. Requires a
  // MultiThreadedExecutor (see main()).
  sub_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = sub_group_;
  auto qos = rclcpp::SensorDataQoS();  // BEST_EFFORT, KEEP_LAST
  ls_sub_ = create_subscription<LowState>(
      lowstate_topic, qos, [this](LowState::SharedPtr m) {
        std::lock_guard<std::mutex> lk(state_mu_);
        last_ls_ = m;
        last_ls_time_ = now();
      }, sub_opts);
  base_sub_ = create_subscription<SportModeState>(
      base_topic, qos, [this](SportModeState::SharedPtr m) {
        std::lock_guard<std::mutex> lk(state_mu_);
        last_base_ = m;
        last_base_time_ = now();
      }, sub_opts);
  cmd_pub_ = create_publisher<LowCmd>(cmd_topic, 10);
  if (drop_band_) band_cli_ = create_client<std_srvs::srv::Trigger>("/elastic_band/toggle");

  const int hw = mjpc::NumAvailableHardwareThreads();
  const int n_threads = std::max(1, std::min(8, hw));
  plan_pool_ = std::make_unique<mjpc::ThreadPool>(n_threads);

  // ROS-time timer (honours use_sim_time via /clock) so the control cadence, the
  // planner clock, and the freshness watchdog all share one time base -- and it
  // does not fire until /clock is valid, so t0_ never latches against time 0.
  control_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / kCtrlHz),
      [this]() { ControlStep(); });

  RCLCPP_INFO(get_logger(),
              "controller ready: task='%s' strategy=%d(stand) gravity_ff=%.2f "
              "plan_rate_hz=%.0f ctrl_hz=%.0f threads=%d | sub %s + %s -> pub %s",
              kTaskId, kStrategy, gff_, plan_rate_hz_, kCtrlHz, n_threads,
              lowstate_topic.c_str(), base_topic.c_str(), cmd_topic.c_str());
}

ControllerNode::~ControllerNode() {
  g_emit_safe_hold = nullptr;
  if (sd_) mj_deleteData(sd_);
  if (gd_) mj_deleteData(gd_);
  if (g_model) { mj_deleteModel(g_model); g_model = nullptr; }
}

void ControllerNode::InitMjpc() {
  g_agent.SetTaskList(mjpc::GetTasks());
  const int task_index = g_agent.GetTaskIdByName(kTaskId);
  if (task_index < 0) {
    RCLCPP_FATAL(get_logger(), "unknown MJPC task '%s'", kTaskId);
    throw std::runtime_error("unknown MJPC task");
  }
  g_agent.gui_task_id = task_index;
  g_agent.SetTaskByIndex(task_index);
  auto lm = g_agent.LoadModel();
  if (!lm.model) {
    RCLCPP_FATAL(get_logger(), "LoadModel failed: %s", lm.error.c_str());
    throw std::runtime_error("LoadModel failed");
  }
  // per-strategy planner numeric overrides (node stays strategy-agnostic).
  {
    mjModel* sm = lm.model.get();
    for (const auto& kv : g_agent.ActiveTask()->PlannerNumericOverrides(kStrategy)) {
      int id = mj_name2id(sm, mjOBJ_NUMERIC, kv.first.c_str());
      if (id >= 0) sm->numeric_data[sm->numeric_adr[id]] = kv.second;
    }
  }
  PatchActuators(lm.model.get());
  g_agent.Initialize(lm.model.get());
  g_agent.Allocate();
  g_agent.Reset();
  g_task = g_agent.ActiveTask();
  g_agent_model = g_agent.GetModel();
  g_model = mj_copyModel(nullptr, g_agent_model);
  PatchActuators(g_model);

  nq_ = g_model->nq;
  nv_ = g_model->nv;
  nu_ = g_model->nu;
  nact_ = std::min(nu_, kNU);

  home_key_ = mj_name2id(g_model, mjOBJ_KEY, "home");
  sd_ = mj_makeData(g_model);
  gd_ = mj_makeData(g_model);
  mjData* init = mj_makeData(g_model);
  if (home_key_ >= 0) {
    mj_resetDataKeyframe(g_model, sd_, home_key_);
    mj_resetDataKeyframe(g_model, gd_, home_key_);
    mj_resetDataKeyframe(g_model, init, home_key_);
  }
  mjcb_sensor = residual_sensor_callback;
  g_agent.SetState(init);
  mj_deleteData(init);
  g_agent.plan_enabled = true;
  g_agent.action_enabled = true;
  g_agent.SetParamByName("residual_Strategy", static_cast<double>(kStrategy));

  // bring-up ramp destination = the "stand" stance (bent-knee), not the singular
  // straight-knee "home" (ramping to knee=0 hands the policy the hyperextension basin).
  int hk = mj_name2id(g_model, mjOBJ_KEY, "stand");
  if (hk < 0) hk = home_key_;
  if (hk < 0) hk = 0;
  for (int i = 0; i < kNU; i++) {
    home_q_[i] = (g_model->nkey > 0) ? g_model->key_qpos[hk * nq_ + 7 + i] : 0.0;
  }
  action_.assign(nu_, 0.0);
}

// Fill the planner state from the latest LowState (joints + IMU) and the
// SportModeState (IMU-site base pose/velocity). Orientation + gyro come from
// /lowstate; the site pose is backed out to the pelvis with the SAME raw IMU
// quaternion + IMU_OFFSET the estimator used, so the reconstruction is exact.
void ControllerNode::FillState(const LowState& ls, const SportModeState& sp) {
  // raw base orientation from the robot IMU (wxyz).
  double bq_raw[4] = {ls.imu_state.quaternion[0], ls.imu_state.quaternion[1],
                      ls.imu_state.quaternion[2], ls.imu_state.quaternion[3]};
  mju_normalize4(bq_raw);
  double R[9];
  mju_quat2Mat(R, bq_raw);
  double gyro[3] = {ls.imu_state.gyroscope[0], ls.imu_state.gyroscope[1],
                    ls.imu_state.gyroscope[2]};
  // site (IMU) -> pelvis: p_pelvis = p_site - R*off; v_pelvis = v_site - (R*gyro) x (R*off).
  double roff[3];
  mju_rotVecMat(roff, h12::IMU_OFFSET, R);
  double omega_w[3];
  mju_rotVecMat(omega_w, gyro, R);
  double wxr[3];
  mju_cross(wxr, omega_w, roff);
  sd_->qpos[0] = sp.position[0] - roff[0];
  sd_->qpos[1] = sp.position[1] - roff[1];
  sd_->qpos[2] = sp.position[2] - roff[2];
  // free-joint velocity: qvel[0:3] = WORLD pelvis linvel, qvel[3:6] = BODY angvel (gyro).
  sd_->qvel[0] = sp.velocity[0] - wxr[0];
  sd_->qvel[1] = sp.velocity[1] - wxr[1];
  sd_->qvel[2] = sp.velocity[2] - wxr[2];
  sd_->qvel[3] = gyro[0];
  sd_->qvel[4] = gyro[1];
  sd_->qvel[5] = gyro[2];

  // planner orientation = IMU quaternion with the balance pitch/roll zero-offset.
  // A constant bias in the perceived vertical makes the planner balance around a
  // false vertical -> a steady lean/creep; a small body-frame post-multiply cancels
  // it. Applied to the PLANNER orientation only -- NOT the back-out geometry above.
  double bq[4];
  mju_copy4(bq, bq_raw);
  if (imu_pitch_off_ != 0.0) {
    double d[4], ax[3] = {0, 1, 0}, t[4];
    mju_axisAngle2Quat(d, ax, imu_pitch_off_);
    mju_mulQuat(t, bq, d); mju_copy4(bq, t);
  }
  if (imu_roll_off_ != 0.0) {
    double d[4], ax[3] = {1, 0, 0}, t[4];
    mju_axisAngle2Quat(d, ax, imu_roll_off_);
    mju_mulQuat(t, bq, d); mju_copy4(bq, t);
  }
  mju_normalize4(bq);
  for (int k = 0; k < 4; k++) sd_->qpos[3 + k] = bq[k];
  // stash for MaybeReleaseBand / MaybeLogStatus (message carries no orientation/z).
  mju_copy4(base_quat_, bq);
  base_pos_z_ = sd_->qpos[2];

  const int nmotor = static_cast<int>(ls.motor_state.size());
  // Feed EVERY joint the model has (legs + torso + arms + wrists), not just the
  // 12 actuated legs. frame_task IK drives the arms to a home pose the planner
  // must SEE: leaving qpos[19..] frozen at the model keyframe gives the planner a
  // wrong CoM/inertia, so when full-authority policy engages it tips the robot
  // over (observed forward fall ~1-2 s after policy blend-in). The /lowstate
  // motor order matches the model joint order (legs 0-11, torso 12, arms 13+).
  const int njnt = nq_ - 7;
  for (int i = 0; i < njnt && i < nmotor; i++) {
    sd_->qpos[7 + i] = ls.motor_state[i].q;
    sd_->qvel[6 + i] = ls.motor_state[i].dq;
  }
}

void ControllerNode::EmitSafeHold() {
  // damping stop: kp=0, small kd, tau=0. Hold at the last measured pose if we
  // have one (harmless with kp=0), else zeros. May be called from the control
  // thread or from FatalMjuError, so snapshot last_ls_ under the mutex.
  LowState::SharedPtr ls_ptr;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    ls_ptr = last_ls_;
  }
  LowCmd cmd;
  if (ls_ptr) cmd.mode_machine = ls_ptr->mode_machine;
  const int nmotor = ls_ptr ? static_cast<int>(ls_ptr->motor_state.size()) : 0;
  for (int i = 0; i < kNU; i++) {
    auto& mc = cmd.motor_cmd[i];
    mc.mode = 1;
    mc.q = (i < nmotor) ? ls_ptr->motor_state[i].q : 0.0f;
    mc.dq = 0.0f;
    mc.tau = 0.0f;
    mc.kp = 0.0f;
    mc.kd = kSafeHoldKd;
  }
  if (cmd_pub_) cmd_pub_->publish(cmd);
}

void ControllerNode::ControlStep() {
  // Snapshot the latest inputs under the mutex (sub callbacks run concurrently
  // in the reentrant group).
  LowState::SharedPtr ls_ptr;
  SportModeState::SharedPtr base_ptr;
  rclcpp::Time ls_time, base_time;
  {
    std::lock_guard<std::mutex> lk(state_mu_);
    ls_ptr = last_ls_;
    base_ptr = last_base_;
    ls_time = last_ls_time_;
    base_time = last_base_time_;
  }
  // Wait for the first full state before commanding anything.
  if (!ls_ptr || !base_ptr) return;

  const rclcpp::Time t = now();
  if (t.nanoseconds() <= 0) return;   // wait for a valid sim clock

  // H1: input-freshness watchdog. If either stream is stale, damp and bail.
  if ((t - ls_time).seconds() > kStaleSec || (t - base_time).seconds() > kStaleSec) {
    if (!stale_warned_) {
      RCLCPP_WARN(get_logger(), "state stale (>%.0f ms) -> safe-hold (damping stop)",
                  kStaleSec * 1e3);
      stale_warned_ = true;
    }
    EmitSafeHold();
    return;
  }
  stale_warned_ = false;

  const LowState& ls = *ls_ptr;
  const SportModeState& sp = *base_ptr;
  const int nmotor = static_cast<int>(ls.motor_state.size());

  if (!have_t0_) { t0_ = t; have_t0_ = true; }
  const double wall = (t - t0_).seconds();
  const bool warming = wall < kWarmupSec;

  FillState(ls, sp);

  // latch the measured power-on pose + rescale the ramp by distance from home.
  if (!init_set_) {
    for (int i = 0; i < kNU; i++) q_init_[i] = (i < nmotor) ? ls.motor_state[i].q : 0.0;
    double d0 = 0.0;
    for (int i = 0; i < kNU; i++) d0 = std::fmax(d0, std::fabs(q_init_[i] - home_q_[i]));
    ramp_eff_ = kStartRampSec * std::fmin(1.0, d0 / 0.5);
    RCLCPP_INFO(get_logger(), "latched pose %.2f rad from home -> effective ramp %.1fs", d0,
                ramp_eff_);
    init_set_ = true;
  }

  // advance the planner clock (sim time); mj_forward; Transition (loads strategy
  // keyframe/weights + advances multi-phase); SetState.
  sd_->time = wall;
  mj_forward(g_model, sd_);
  g_agent.ActiveTask()->Transition(g_model, sd_);
  g_agent.SetState(sd_);

  // synchronous fractional planning on THIS fresh state (audit-M1 mode).
  plan_accum_ += plan_rate_hz_ / kCtrlHz;
  int niter = static_cast<int>(plan_accum_);
  plan_accum_ -= niter;
  for (int i = 0; i < niter; i++) g_agent.PlanIteration(plan_pool_.get());

  if (!warming) {
    g_agent.ActivePlanner().ActionFromPolicy(action_.data(), g_agent.state.state().data(),
                                             g_agent.state.time());
  }

  // gravity feedforward: tau = gff * qfrc_bias evaluated at qvel = 0.
  double tau[kNU] = {0};
  if (gff_ != 0.0) {
    mju_copy(gd_->qpos, sd_->qpos, nq_);
    mju_zero(gd_->qvel, nv_);
    mj_forward(g_model, gd_);
    for (int i = 0; i < kNU; i++) tau[i] = gff_ * gd_->qfrc_bias[6 + i];
  }

  // per-joint target with the bring-up ramp: measured -> stance -> (blend) policy.
  const double t_ho = ramp_eff_ + kRampHoldSec;
  double tgt_q[kNU];
  for (int i = 0; i < kNU; i++) {
    double tgt;
    if (ramp_eff_ > 0.0) {
      const double aa = std::fmin(1.0, std::fmax(0.0, wall / ramp_eff_));
      double policy_tgt;
      if (aa < 1.0 || warming || i >= nact_ || wall < t_ho) {
        policy_tgt = home_q_[i];                       // rising / warmup / scripted hold
      } else if (kPolicyBlendSec > 0.0 && wall < t_ho + kPolicyBlendSec) {
        const double bb = (wall - t_ho) / kPolicyBlendSec;
        policy_tgt = (1.0 - bb) * home_q_[i] + bb * action_[i];
      } else {
        policy_tgt = action_[i];                       // full policy authority
      }
      tgt = (1.0 - aa) * q_init_[i] + aa * policy_tgt;
    } else {
      tgt = (warming || i >= nact_) ? ((i < nmotor) ? ls.motor_state[i].q : 0.0) : action_[i];
    }
    tgt_q[i] = tgt;
  }

  // H2: torque-aware clamp on the FULL commanded torque budget.
  for (int i = 0; i < kNU; i++) {
    const double q = (i < nmotor) ? ls.motor_state[i].q : 0.0;
    const double dq = (i < nmotor) ? ls.motor_state[i].dq : 0.0;
    tgt_q[i] = h12::ClampTargetFullBudget(i, tgt_q[i], q, dq, tau[i], clamp_ratio_);
  }

  // build + publish the 12-leg LowCmd (safety layer computes the CRC downstream).
  LowCmd cmd;
  cmd.mode_pr = 0;
  cmd.mode_machine = ls.mode_machine;
  for (int i = 0; i < kNU; i++) {
    auto& mc = cmd.motor_cmd[i];
    mc.mode = 1;
    mc.q = static_cast<float>(tgt_q[i]);
    mc.dq = 0.0f;
    mc.tau = static_cast<float>(tau[i]);
    mc.kp = static_cast<float>(KP[i]);
    mc.kd = static_cast<float>(KV[i]);
  }
  cmd_pub_->publish(cmd);

  MaybeReleaseBand(wall);
  MaybeLogStatus(wall, warming, ls, tgt_q, tau, nmotor);
}

// Self-contained standing demo: once the controller has run for band_drop_after_secs
// (a fixed dwell timer off t0), drop the sim's elastic support tether exactly once.
// No-op on the real robot (the service is absent -> a periodic warning, never a
// spurious command).
void ControllerNode::MaybeReleaseBand(double wall) {
  if (!drop_band_ || band_released_) return;
  if (++band_ticks_ % 20 != 0) return;          // ~10 Hz check, off the 200 Hz hot loop
  if (wall < band_drop_after_secs_) return;    // fixed dwell: not enough run time yet
  if (!band_cli_ || !band_cli_->service_is_ready()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "t=%.1fs (>= %.1fs) but /elastic_band/toggle unavailable -- band NOT dropped",
                         wall, band_drop_after_secs_);
    return;
  }
  band_cli_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  band_released_ = true;
  RCLCPP_INFO(get_logger(),
              "t=%.1fs (>= %.1fs dwell) -> dropping elastic band", wall, band_drop_after_secs_);
}

// M4: grade torque headroom against TAU_ESTOP (the actual trip point), not TAU_LIMIT.
void ControllerNode::MaybeLogStatus(double wall, bool warming, const LowState& ls,
                                    const double* tgt_q, const double* tau,
                                    int nmotor) {
  if (++ticks_ % static_cast<long>(kCtrlHz) != 0) return;
  double worst_pct = 0.0; int worst = 0;
  for (int i = 0; i < kNU; i++) {
    const double q = (i < nmotor) ? ls.motor_state[i].q : 0.0;
    const double dq = (i < nmotor) ? ls.motor_state[i].dq : 0.0;
    const double total = std::fabs(tau[i] + KP[i] * (tgt_q[i] - q) + KV[i] * (0.0 - dq));
    const double pct = 100.0 * total / TAU_ESTOP[i];   // vs ESTOP, not LIMIT
    if (pct > worst_pct) { worst_pct = pct; worst = i; }
  }
  double R[9]; double bq[4]; mju_copy4(bq, base_quat_);
  mju_normalize4(bq); mju_quat2Mat(R, bq);
  const double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, R[8]))) * 57.29578;
  // body-frame lean of the world-up vector (row 2 of R): +pitch and +roll tell
  // which way it creeps, so the imu_*_offset sign can be chosen to counter it.
  const double lean_pitch = std::atan2(R[6], R[8]) * 57.29578;
  const double lean_roll = std::atan2(R[7], R[8]) * 57.29578;
  RCLCPP_INFO(get_logger(),
              "t=%5.1fs %s z=%.3f tilt=%4.1f lean(p/r)=%+.1f/%+.1f  worst tau=%.0f%%estop(%s)  plan=%.0f/s",
              wall, warming ? "WARMUP" : "policy", base_pos_z_, tilt,
              lean_pitch, lean_roll, worst_pct, JOINT_NAMES[worst], plan_rate_hz_);
  if (worst_pct > 90.0)
    RCLCPP_WARN(get_logger(), "%s at %.0f%% of its safety estop (>90 = near-trip)",
                JOINT_NAMES[worst], worst_pct);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int rc = 0;
  try {
    auto node = std::make_shared<ControllerNode>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[node] fatal: %s\n", e.what());
    rc = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return rc;
}
