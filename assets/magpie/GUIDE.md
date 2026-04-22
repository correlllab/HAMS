# Magpie Gripper MuJoCo Model Guide

This document describes how the Magpie gripper's 4-bar linkage mechanism is modeled in MuJoCo, including coordinate transformations from Onshape CAD and the closed kinematic chain setup.

## CAD Pivot Positions (Onshape)

The gripper has two mirrored 4-bar linkages (left and right). Each linkage has 4 pivot points:

### Right Finger (mm)
| Pivot | Description | X | Y | Z |
|-------|-------------|---|---|---|
| Crank | crank-base | 38.0 | 1.75 | 69.0 |
| Rocker | rocker-base | 60.0 | -0.750 | 65.0 |
| Finger (outer) | finger-outer | 99.824 | -6.750 | 85.954 |
| Finger (inner) | finger-inner | 77.824 | -6.750 | 89.953 |

### Left Finger (mm)
| Pivot | Description | X | Y | Z |
|-------|-------------|---|---|---|
| Crank | crank-base | -38.0 | 1.75 | 69.0 |
| Rocker | rocker-base | -60.0 | -0.750 | 65.0 |
| Finger (outer) | finger-outer | -99.824 | -6.750 | 85.954 |
| Finger (inner) | finger-inner | -77.824 | -6.750 | 89.953 |

## Coordinate Transformation: Onshape to MuJoCo

The STL meshes exported from Onshape use a different coordinate system than MuJoCo. To align them correctly, we apply a rotation via the `xyaxes` attribute in the default geom settings:

```xml
<default>
    <geom type="mesh" xyaxes="0 -1 0 1 0 0"/>
</default>
```

This specifies:
- MuJoCo local X-axis = (0, -1, 0) in parent frame (points in -Y direction)
- MuJoCo local Y-axis = (1, 0, 0) in parent frame (points in +X direction)
- MuJoCo local Z-axis = cross product = (0, 0, 1) (unchanged)

### Position Transformation Formula

To convert CAD coordinates to MuJoCo world coordinates:

```
MuJoCo X = CAD Y / 1000
MuJoCo Y = -CAD X / 1000
MuJoCo Z = CAD Z / 1000
```

(Division by 1000 converts mm to meters)

### Transformed Pivot Positions (MuJoCo, meters)

#### Left Finger
| Pivot | Name | MuJoCo (X, Y, Z) |
|-------|------|------------------|
| 1 | crank-base | (0.00175, 0.038, 0.069) |
| 2 | finger-inner | (-0.00675, 0.077824, 0.089953) |
| 3 | finger-outer | (-0.00675, 0.099824, 0.085954) |
| 4 | rocker-base | (-0.00075, 0.060, 0.065) |

#### Right Finger
| Pivot | Name | MuJoCo (X, Y, Z) |
|-------|------|------------------|
| 1 | crank-base | (0.00175, -0.038, 0.069) |
| 2 | finger-inner | (-0.00675, -0.077824, 0.089953) |
| 3 | finger-outer | (-0.00675, -0.099824, 0.085954) |
| 4 | rocker-base | (-0.00075, -0.060, 0.065) |

## 4-Bar Linkage Structure

### Physical Layout

Each finger mechanism is a 4-bar linkage with this topology:

```
Left Finger:                          Right Finger:

finger-outer -------- finger-inner    finger-inner -------- finger-outer
(pivot 3)             (pivot 2)       (pivot 2)             (pivot 3)
    |                     |               |                     |
    |      [FINGER]       |               |      [FINGER]       |
    |                     |               |                     |
[ROCKER]              [CRANK]         [CRANK]              [ROCKER]
    |                     |               |                     |
rocker-base --------- crank-base      crank-base --------- rocker-base
(pivot 4)             (pivot 1)       (pivot 1)             (pivot 4)
    |_____________________|               |_____________________|
            [BASE]                               [BASE]
```

### Link Descriptions

| Link | Connects | Role |
|------|----------|------|
| Base | pivot 1 to pivot 4 | Ground link (fixed) |
| Crank | pivot 1 to pivot 2 | Input link (actuated) |
| Finger | pivot 2 to pivot 3 | Coupler (output motion) |
| Rocker | pivot 3 to pivot 4 | Closes the loop |

## MuJoCo Kinematic Chain

MuJoCo uses a tree structure for kinematics, so we can't directly model a closed loop. Instead, we build an open chain and close it with an equality constraint.

### Kinematic Tree Structure

```
base_top
    └── crank (body origin at pivot 1)
            └── finger (body origin at pivot 2)
                    └── rocker (body origin at pivot 3)
```

This matches the pattern from `example_4bar.xml`:
- link_1 (red/ground) → link_2 (blue/crank) → link_3 (green/coupler) → link_4 (white/rocker)

### Joint Placement

Each joint is placed at the body origin (pos="0 0 0"):

| Joint | Connects | Location | Pivot |
|-------|----------|----------|-------|
| hinge_1 | base → crank | crank origin | pivot 1 (crank-base) |
| hinge_2 | crank → finger | finger origin | pivot 2 (finger-inner) |
| hinge_3 | finger → rocker | rocker origin | pivot 3 (finger-outer) |

All joints rotate about the X-axis: `axis="1 0 0"`

### Closing the Loop with Equality Constraint

The `<connect>` equality constraint closes the 4-bar loop by constraining the rocker to the base at pivot 4:

```xml
<equality>
    <connect name="left_4bar_close" body1="base_top" body2="left_rocker"
             anchor="-0.00075 0.060 0.065"/>
    <connect name="right_4bar_close" body1="base_top" body2="right_rocker"
             anchor="-0.00075 -0.060 0.065"/>
</equality>
```

The `anchor` specifies pivot 4 (rocker-base) position in the base_top frame. The constraint pulls the rocker's origin toward this anchor, and the solver adjusts joint angles to satisfy the closed-loop kinematics.

### Body Positions (Relative to Parent)

| Body | Parent | Relative Position | Calculation |
|------|--------|-------------------|-------------|
| crank | base_top | pivot 1 | directly from CAD |
| finger | crank | pivot 2 - pivot 1 | (-0.0085, ±0.039824, 0.020953) |
| rocker | finger | pivot 3 - pivot 2 | (0, ±0.022, -0.003999) |

### Geom Offsets

Each mesh geom has a position offset that places it correctly relative to the body origin. The offset is the negative of the body's absolute position, which places the mesh at the base_top origin:

```
geom_offset = -(body_absolute_position)
```

## Actuation

The 4-bar linkage is driven by actuating the crank joint (hinge_1). This is the standard approach for 4-bar mechanisms - rotating the crank causes the entire linkage to move, with the finger (coupler) following a specific trajectory.

### Joint Range

The crank joints have a configurable range limit (in radians):

```xml
<joint name="left_hinge_1" ... range="-0.5 0.5"/>
<joint name="right_hinge_1" ... range="-0.5 0.5"/>
```

**Tuning the range:**
- Positive values open the gripper (fingers move apart)
- Negative values close the gripper (fingers move together)
- Adjust based on physical limits of your mechanism
- Current default: ±0.5 rad (±28.6°)

### Position Actuators

Position actuators provide PD control to drive each finger:

```xml
<actuator>
    <position name="left_finger_actuator" joint="left_hinge_1"
              ctrlrange="-0.5 0.5" kp="10"/>
    <position name="right_finger_actuator" joint="right_hinge_1"
              ctrlrange="-0.5 0.5" kp="10"/>
</actuator>
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ctrlrange` | Control input range (should match joint range) | -0.5 to 0.5 rad |
| `kp` | Position gain (stiffness) | 10 |

**Tuning tips:**
- Increase `kp` for stiffer, faster response
- Decrease `kp` for softer, more compliant grip
- `ctrlrange` should match the joint `range`

### Control Interface

In Python with MuJoCo:

```python
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path("magpie.xml")
data = mujoco.MjData(model)

# Set finger positions (in radians)
data.ctrl[0] = 0.3   # left finger
data.ctrl[1] = 0.3   # right finger

# Step simulation
mujoco.mj_step(model, data)
```

## Modular File Structure

The magpie gripper is organized into modular files for reuse:

### File Organization

| File | Purpose |
|------|---------|
| `magpie.xml` | Standalone gripper model (complete with worldbody) |
| `magpie_gripper.xml` | Body hierarchy only (for inclusion in other models) |
| `ur5_magpie.xml` | UR5E arm + magpie gripper combined |

### Using `<include>` for Reuse

The `magpie_gripper.xml` file contains only the body tree (no `<mujoco>` or `<worldbody>` wrapper), allowing it to be included as a child of any body:

```xml
<!-- In your robot model, attach gripper to end effector -->
<body name="end_effector">
  <body name="gripper_attachment" pos="0 0.1 0" quat="0.707 -0.707 0 0">
    <include file="magpie_gripper.xml"/>
  </body>
</body>
```

### Attachment Considerations

When attaching to a robot arm:

1. **Position**: Offset the gripper mount from the wrist flange
2. **Orientation**: Rotate to align gripper Z-axis (through fingers) with arm's end-effector direction
3. **Assets**: Include magpie mesh assets with `magpie_` prefix to avoid name conflicts
4. **Equality constraints**: Add the 4-bar loop closure constraints
5. **Actuators**: Add position actuators for the crank joints

### Example: UR5E Integration

```xml
<!-- In ur5_magpie.xml -->
<body name="wrist_3_link">
  ...
  <!-- Rotate gripper: UR5E Y-out → Magpie Z-out (-90° about X) -->
  <body name="gripper_attachment" pos="0 0.1 0" quat="0.7071068 -0.7071068 0 0">
    <include file="magpie_gripper.xml"/>
  </body>
</body>

<!-- Don't forget equality constraints and actuators! -->
<equality>
  <connect name="left_4bar_close" body1="base_top" body2="left_rocker"
           anchor="-0.00075 0.060 0.065"/>
  ...
</equality>
```

### Keeping Models in Sync

When you modify `magpie_gripper.xml`:
- Changes automatically propagate to both `magpie.xml` and `ur5_magpie.xml` (via include)
- Only one source file to maintain for the body hierarchy

When you modify pivot positions or add links:
- Update `magpie_gripper.xml` (single source of truth for body tree)
- Update equality constraint anchors in `magpie.xml` and `ur5_magpie.xml`
- Update actuator definitions if joint names change

## Summary

1. **Export meshes** from Onshape as STL
2. **Apply xyaxes transform** to rotate meshes into MuJoCo frame
3. **Convert pivot positions** using: MuJoCo = (CAD_Y, -CAD_X, CAD_Z) / 1000
4. **Build kinematic tree**: base → crank → finger → rocker
5. **Place joints at body origins** (pivot 1, 2, 3)
6. **Close loop with connect constraint** anchored at pivot 4
7. **Add actuators** on crank joints (hinge_1) to drive the mechanism
8. **Use modular includes** for robot arm integration
