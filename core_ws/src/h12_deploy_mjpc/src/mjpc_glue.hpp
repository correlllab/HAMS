// Pure, ROS/MJPC-independent helpers for the H1-2 lower-body MJPC controller:
// the authoritative per-joint gains + safety limits, and the torque-aware target
// clamp (audit H2). Header-only so it can be unit-tested without MuJoCo or ROS.
#ifndef H12_DEPLOY_MJPC_MJPC_GLUE_HPP_
#define H12_DEPLOY_MJPC_MJPC_GLUE_HPP_

#include <algorithm>
#include <cmath>

namespace h12 {

// LEGS-ONLY lower-body controller: 12 actuated joints below the pelvis
// (L/R hip yaw/pitch/roll, knee, ankle pitch/roll). qpos[7..18] order.
inline constexpr int kNU = 12;

// Per-joint gains == h1_2_modified actuator classes == real LowCmd kp/kd
// (must match the safety-layer / twin PD). Patched into the planner model.
inline constexpr double KP[kNU] = {150, 200, 200, 200, 80, 80, 150, 200, 200, 200, 80, 80};
inline constexpr double KV[kNU] = {5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 4, 4};

// Safety-layer tau-ESTOP thresholds (estop torque_ratio x URDF torque limit).
// The clamp keeps the FULL commanded torque under 0.9x these.
inline constexpr double TAU_ESTOP[kNU] = {60, 130, 200, 300, 54, 36, 60, 130, 200, 300, 54, 36};

// Operational H1-2 joint torque limits (Nm) = URDF actuatorfrcrange. Used only
// for the operator torque readout; the estop trips at TAU_ESTOP (below this).
inline constexpr double TAU_LIMIT[kNU] = {200, 200, 200, 300, 60, 40, 200, 200, 200, 300, 60, 40};

inline constexpr const char* JOINT_NAMES[kNU] = {
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR"};

// audit H2: bound the emitted position target so the FULL commanded torque the
// onboard/safety PD applies -- tau_ff + KP*(tgt-q) + KV*(0-dq) -- stays within
// 0.9x the safety estop. Upstream only bounded the KP*(tgt-q) term and ignored
// the gravity feedforward tau_ff and the KV*dq transient, so the "estop
// impossible" guarantee was false. Here the PD headroom is reduced by |tau_ff|
// and KV*|dq| before converting to a position delta.
inline double ClampTargetFullBudget(int i, double tgt, double q, double dq, double tau_ff,
                                    double ratio = 0.9) {
  const double budget = ratio * TAU_ESTOP[i];
  const double pd_headroom = budget - std::fabs(tau_ff) - KV[i] * std::fabs(dq);
  const double dmax = (pd_headroom > 0.0) ? pd_headroom / KP[i] : 0.0;
  const double lo = q - dmax, hi = q + dmax;
  return std::min(hi, std::max(lo, tgt));
}

}  // namespace h12

#endif  // H12_DEPLOY_MJPC_MJPC_GLUE_HPP_
