import random
import subprocess
import time

import cv2
from ultralytics import YOLO

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage

import omni.graph.core as og
import numpy as np
import omni.usd
import omni.kit.commands

from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim

from isaacsim.robot.manipulators import SingleManipulator
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.core.utils.types import ArticulationAction

from pxr import Gf, UsdGeom, Usd, UsdPhysics


# ================================
# 환경 / 컨베이어 설정
# ================================
WAREHOUSE_USD_PATH = "/home/rokey/Documents/assets/warehouse_ver20.usd"
WAREHOUSE_PRIM_PATH = "/World/Conveyor"

CONVEYOR_PATHS = [
    "/World/Conveyor/conveyor/ConveyorTrack_2_3/ConveyorBeltGraph",  # 오른쪽 (idx=0)
    "/World/Conveyor/conveyor/ConveyorTrack_1_3/ConveyorBeltGraph",  # 왼쪽 (idx=1)
]

PHYSICS_DT = 1.0 / 60.0

# ================================
# 배터리 스폰 설정
# ================================
SPAWN_USD_PATH = "/home/rokey/Documents/assets/batteryModule2.usdc"
SPAWN_INTERVAL_SEC = 8.0
SPAWN_MASS_KG = 2.5
ENABLE_CCD = True
COLLIDER_APPROXIMATION = "convexDecomposition"
MAX_SPAWN_COUNT = 6
SPAWN_SCALE = (0.8, 0.8, 1.0)

SPAWN_POSES = [
    {
        "pos": (5.19132, -24.5, 3.20497),
        "rot_deg": (0.0, 0.0, 0.0),
    },
    {
        "pos": (-26.0, -24.3, 3.20497),
        "rot_deg": (0.0, 0.0, 0.0),
    },
]

# ================================
# 로봇 설정 (Robot 1~4)
# ================================
ROBOT2_PRIM_PATH = "/World/Conveyor/robot/ur10_2"
EE2_PRIM_PATH = f"{ROBOT2_PRIM_PATH}/ee_link"

ROBOT_PRIM_PATH = "/World/Conveyor/robot/ur10_4"
EE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/ee_link"

ROBOT4_PRIM_PATH = "/World/Conveyor/robot/ur10_3"
EE4_PRIM_PATH = f"{ROBOT4_PRIM_PATH}/ee_link"

ROBOT3_PRIM_PATH = "/World/Conveyor/robot/ur10_1"
EE3_PRIM_PATH = f"{ROBOT3_PRIM_PATH}/ee_link"

INITIAL_JOINTS = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64)
HOME_JOINT_TOL = 0.02

# ================================
# 배터리 케이스 / 조립체 설정
# ================================
BATTCASE_USD_PATH = "/home/rokey/Documents/assets/battcase2.usdc"
BATTEND_USD_PATH = "/home/rokey/Documents/assets/battend2.usdc"

BATTCASE2_PRIM_PATH = "/World/Battcase2"
BATTCASE2_2_PRIM_PATH = "/World/Battcase2_2"

BATTCASE2_WORLD_POS = np.array([-19.5, -5.0, 1.1], dtype=np.float64)
BATTCASE2_2_WORLD_POS = np.array([-19.7, -2.05, 1.16], dtype=np.float64)

BATTCASE2_ROT_DEG = np.array([0.0, 0.0, 180.0], dtype=np.float64)
BATTCASE2_2_ROT_DEG = np.array([180.0, 0.0, 0.0], dtype=np.float64)

BATTCASE1_USD_PATH = "/home/rokey/Documents/assets/battcase1.usdc"
BATTEND1_USD_PATH = "/home/rokey/Documents/assets/battend1.usdc"

BATTCASE1_PRIM_PATH = "/World/Battcase1"
BATTCASE1_2_PRIM_PATH = "/World/Battcase1_2"

BATTFULL1_PATH = "/home/rokey/Documents/assets/battfull1.usdc"
BATTFULL2_PATH = "/home/rokey/Documents/assets/battfull2.usdc"

BATTFULL1_PRIM_PATH = "/World/Battfull1"
BATTFULL2_PRIM_PATH = "/World/Battfull2"

BATTCASE1_WORLD_POS = np.array([-0.85, -4.86, 1.01417], dtype=np.float64)
BATTCASE1_2_WORLD_POS = np.array([-0.9, -2.45, 1.1], dtype=np.float64)

BATTCASE1_ROT_DEG = np.array([0.0, 0.0, -90.0], dtype=np.float64)
BATTCASE1_2_ROT_DEG = np.array([180.0, 0.0, 90.0], dtype=np.float64)

BATTCASE_SCALE = (0.8, 0.8, 0.8)
BATTCASE_MASS_KG = 0.01

# ================================
# 뚜껑 로봇(Cover) 회전 설정
# ================================
R1_APPROACH_Z_WORLD = 1.6
R1_JOINT1_ROTATE_DEG = -180.0
R1_POST_ROTATE_WAIT_STEPS = int(0.5 / PHYSICS_DT)
R1_POST_OPEN_WAIT_STEPS = int(0.5 / PHYSICS_DT)
R1_ARRIVE_JOINT_TOL = 0.001

# ================================
# YOLO / 차량 스폰 설정
# ================================
YOLO_MODEL_PATH = "/home/rokey/Documents/best.pt"
CONVEYOR_STOP_WAIT_SEC = 10.0

CAR_SPAWN_DELAY_SEC = 10.0
CAR_SPAWN_POSITION = (-28.81936, 26.33664, 1.04249)
CAR_SPAWN_ROT_DEG = (0.0, 0.0, 90.0)
CAR_MASS_KG = 1800.0
CAR_USD_PATHS = [
    "/home/rokey/Documents/assets/ioniq5.usdc",
    "/home/rokey/Documents/assets/ioniq6.usdc",
]

# ================================
# 차량 컨베이어 / 정지 / 고정 설정
# ================================
# ★ 실제 Stage 경로에 맞게 수정 필요 ★
CAR_CONVEYOR_PATH = "/Root/Conveyor/ConveyorTrack_car/ConveyorBeltGraph"
CAR_DEFAULT_CONVEYOR_VELOCITY = -0.05
CAR_STOP_AFTER_SPAWN_SEC = 4.3
CAR_STABILIZE_WAIT_SEC = 1.0

# ================================
# NAV 실행 설정
# ================================
ROS_SETUP_BASH = "/opt/ros/humble/setup.bash"
WORKSPACE_SETUP_BASH = "/home/rokey/IsaacSim-ros_workspaces/humble_ws/install/setup.bash"

NAV_COMMAND_BY_LABEL = {
    "ioniq5": "ros2 run my_pkg nav_to_pose2",
    "ioniq6": "ros2 run my_pkg nav_to_pose",
}

# ================================
# 배터리 로봇 동작 튜닝
# ================================
ARRIVE_DIST_TOL = 0.05
ARRIVE_JOINT_TOL = 0.015
STUCK_JOINT_TOL = 0.005

PICK_ZONE_Y = -6.0
PICK_ZONE_THRESHOLD = 0.65

ROBOT2_PICK_Z = 1.32
ROBOT2_PICK_X_OFFSET = -0.01
ROBOT2_PICK_Y_OFFSET = 0.12

PICK_HOLD_STEPS = 30
POST_OPEN_WAIT_STEPS = 10
POST_ROTATE_WAIT_STEPS = int(0.5 / PHYSICS_DT)


# ================================
# RMPFlow 컨트롤러
# ================================
class RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = PHYSICS_DT,
        attach_gripper: bool = False,
    ):
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflow"
            )

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation,
            self.rmp_flow,
            physics_dt,
        )
        mg.MotionPolicyController.__init__(
            self,
            name=name,
            articulation_motion_policy=self.articulation_rmp,
        )

        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )


# ================================
# 메인 시뮬레이션 클래스
# ================================
class ConveyorTestInt(BaseSample):
    def __init__(self) -> None:
        super().__init__()

        self.yolo_model_path = YOLO_MODEL_PATH
        self.yolo = None
        self._yolo_interval = 5

        self.conveyor_attrs = []
        self.conveyor_running = [True, True]

        self.wait_time = CONVEYOR_STOP_WAIT_SEC
        self.stop_timer = [None, None]

        self.recent_i = [1, 0]
        self.send_flag = [False, False]
        self.r2_pos = [None, None]

        self.task_phase_r1 = [0, 0]
        self._r1_pause_timer = [0, 0]
        self._r1_joint1_rotate_target = [None, None]
        self._r2_joint1_target = [None, None]

        self.task_phase_r2 = [0, 0]
        self.r2_pause_timer = [0, 0]
        self.robot_flag = [False, False]

        self.i = [0, 0]
        self.j = [0, 0]

        self._elapsed_time = 0.0
        self._spawn_count = 0

        self._battend_created = [False, False]
        self._phase135_enter_time = [None, None]
        self._phase135_nav_forced = [False, False]
        # 차량 상태
        self._car_elapsed_time = 0.0
        self._car_spawned = False
        self.car_conveyor_attr = None
        self.car_spawned_car_prim_path = "/World/SpawnedObjects/random_car"
        self.car_spawned_gt_label = None
        self.car_stop_triggered = False
        self.car_stabilizing = False
        self.car_final_decision_done = False
        self.car_frozen = False
        self.car_moving_elapsed_time = 0.0
        self.car_stabilize_elapsed_sec = 0.0

        # 양쪽 암 종료 여부
        self.all_scenarios_done = [False, False]

        # NAV 중복 실행 방지
        self.car_nav_sent = False

    def setup_scene(self):
        add_reference_to_stage(usd_path=WAREHOUSE_USD_PATH, prim_path=WAREHOUSE_PRIM_PATH)
        world = self.get_world()

        self._spawn_battcase(BATTCASE2_PRIM_PATH, BATTCASE_USD_PATH, BATTCASE2_WORLD_POS, BATTCASE2_ROT_DEG)
        self._spawn_battcase(BATTCASE2_2_PRIM_PATH, BATTCASE_USD_PATH, BATTCASE2_2_WORLD_POS, BATTCASE2_2_ROT_DEG)
        self._spawn_battcase(BATTCASE1_PRIM_PATH, BATTCASE1_USD_PATH, BATTCASE1_WORLD_POS, BATTCASE1_ROT_DEG)
        self._spawn_battcase(BATTCASE1_2_PRIM_PATH, BATTCASE1_USD_PATH, BATTCASE1_2_WORLD_POS, BATTCASE1_2_ROT_DEG)

        stage = omni.usd.get_context().get_stage()
        for robot_path in [ROBOT_PRIM_PATH, ROBOT2_PRIM_PATH, ROBOT3_PRIM_PATH, ROBOT4_PRIM_PATH]:
            prim = stage.GetPrimAtPath(robot_path)
            if prim.IsValid() and prim.GetVariantSets().HasVariantSet("Gripper"):
                prim.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        self.gripper1 = SurfaceGripper(
            end_effector_prim_path=EE_PRIM_PATH,
            surface_gripper_path=f"{EE_PRIM_PATH}/SurfaceGripper",
        )
        self.gripper2 = SurfaceGripper(
            end_effector_prim_path=EE2_PRIM_PATH,
            surface_gripper_path=f"{EE2_PRIM_PATH}/SurfaceGripper",
        )
        self.gripper3 = SurfaceGripper(
            end_effector_prim_path=EE3_PRIM_PATH,
            surface_gripper_path=f"{EE3_PRIM_PATH}/SurfaceGripper",
        )
        self.gripper4 = SurfaceGripper(
            end_effector_prim_path=EE4_PRIM_PATH,
            surface_gripper_path=f"{EE4_PRIM_PATH}/SurfaceGripper",
        )

        world.scene.add(SingleManipulator(prim_path=ROBOT_PRIM_PATH,  name="my_ur10",   end_effector_prim_path=EE_PRIM_PATH,  gripper=self.gripper1))
        world.scene.add(SingleManipulator(prim_path=ROBOT2_PRIM_PATH, name="my_ur10_2", end_effector_prim_path=EE2_PRIM_PATH, gripper=self.gripper2))
        world.scene.add(SingleManipulator(prim_path=ROBOT3_PRIM_PATH, name="my_ur10_1", end_effector_prim_path=EE3_PRIM_PATH, gripper=self.gripper3))
        world.scene.add(SingleManipulator(prim_path=ROBOT4_PRIM_PATH, name="my_ur10_3", end_effector_prim_path=EE4_PRIM_PATH, gripper=self.gripper4))

    async def setup_post_load(self):
        self._world = self.get_world()

        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        self._stage = omni.usd.get_context().get_stage()
        self._Gf = Gf
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._PhysxSchema = PhysxSchema

        self.robots   = self._world.scene.get_object("my_ur10")    # Right Cover
        self.robots2  = self._world.scene.get_object("my_ur10_2")  # Right Batt
        self.robots_2 = self._world.scene.get_object("my_ur10_3")  # Left Cover
        self.robots2_2= self._world.scene.get_object("my_ur10_1")  # Left Batt

        try:
            self.yolo = YOLO(self.yolo_model_path)
            print("[INFO] YOLO 모델 로드 완료")
        except Exception as exc:
            print(f"[WARN] YOLO 모델 로드 실패: {exc}")

        for path in CONVEYOR_PATHS:
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("graph:variable:Velocity")
            if not attr.IsValid():
                continue
            attr.Set(-0.4)
            self.conveyor_attrs.append(attr)

        # 차량 컨베이어 초기화
        car_conv_prim = self._stage.GetPrimAtPath(CAR_CONVEYOR_PATH)
        if car_conv_prim.IsValid():
            attr = car_conv_prim.GetAttribute("graph:variable:Velocity")
            if attr.IsValid():
                attr.Set(CAR_DEFAULT_CONVEYOR_VELOCITY)
                self.car_conveyor_attr = attr
                print("[INFO] 차량 컨베이어 속도 설정 완료")
            else:
                print(f"[WARN] 차량 컨베이어 velocity attr 없음: {CAR_CONVEYOR_PATH}")
        else:
            print(f"[WARN] 차량 컨베이어 prim 없음: {CAR_CONVEYOR_PATH}")

        stage = omni.usd.get_context().get_stage()
        self.camera1 = Camera(prim_path="/World/BeltCamera1", resolution=(640, 640), frequency=20)
        self.camera2 = Camera(prim_path="/World/BeltCamera2", resolution=(640, 640), frequency=20)
        self.camera1.initialize()
        self.camera2.initialize()

        self.camera1.set_world_pose(
            position=np.array([-21.3, -7.0, 9.0]),
            orientation=np.array([0.5, -0.5, 0.5, 0.5]),
        )
        stage.GetPrimAtPath("/World/BeltCamera1").GetAttribute("focalLength").Set(25)

        self.camera2.set_world_pose(
            position=np.array([0.67, -7.0, 9.0]),
            orientation=np.array([0.5, -0.5, 0.5, 0.5]),
        )
        stage.GetPrimAtPath("/World/BeltCamera2").GetAttribute("focalLength").Set(25)

        self._world.add_physics_callback("spawn_usd_callback", self._on_physics_step)
        self._world.add_physics_callback("yolo_inference_callback",   self._run_yolo(0))
        self._world.add_physics_callback("yolo_inference_callback_1", self._run_yolo(1))
        self._world.add_physics_callback("robot_step",   self.physics_step(0))
        self._world.add_physics_callback("robot_step_1", self.physics_step(1))

        await self._world.play_async()

        for robot in [self.robots, self.robots2, self.robots_2, self.robots2_2]:
            robot.initialize()
            robot.set_joints_default_state(positions=INITIAL_JOINTS)
            robot.set_joint_positions(INITIAL_JOINTS)

        self.cspace_controller  = RMPFlowController(name="c_ctrl_1", robot_articulation=self.robots,    attach_gripper=True)
        self.cspace_controller2 = RMPFlowController(name="c_ctrl_2", robot_articulation=self.robots2,   attach_gripper=True)
        self.cspace_controller3 = RMPFlowController(name="c_ctrl_3", robot_articulation=self.robots_2,  attach_gripper=True)
        self.cspace_controller4 = RMPFlowController(name="c_ctrl_4", robot_articulation=self.robots2_2, attach_gripper=True)

    # ================================================================
    # 물리 / 스폰 헬퍼
    # ================================================================
    def _spawn_battcase(self, prim_path, usd_path, world_pos, rot_deg):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            return
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2])))
        xform.AddScaleOp().Set(Gf.Vec3f(*BATTCASE_SCALE))
        self._apply_rigid_body_and_collision_battcase(prim_path)

    def _apply_rigid_body_and_collision_battcase(self, root_prim_path):
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rigid_api = UsdPhysics.RigidBodyAPI(root_prim)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        rigid_api.CreateKinematicEnabledAttr(False)
        if not root_prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(root_prim)
        UsdPhysics.MassAPI(root_prim).CreateMassAttr(BATTCASE_MASS_KG)
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == "Mesh":
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("convexDecomposition")

    def _set_transform(self, prim, pos, rot_deg, scale=(1.0, 1.0, 1.0)):
        xform = self._UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(self._Gf.Vec3d(*pos))
        xform.AddRotateXYZOp().Set(self._Gf.Vec3f(*rot_deg))
        from pxr import UsdGeom as PxUsdGeom
        xform.AddScaleOp(precision=PxUsdGeom.XformOp.PrecisionDouble).Set(self._Gf.Vec3d(*scale))

    def _apply_rigid_body_and_colliders(self, root_prim, mass_kg, approximation):
        if not root_prim.HasAPI(self._UsdPhysics.RigidBodyAPI):
            self._UsdPhysics.RigidBodyAPI.Apply(root_prim)
        self._UsdPhysics.RigidBodyAPI(root_prim).CreateRigidBodyEnabledAttr(True)
        if not root_prim.HasAPI(self._UsdPhysics.MassAPI):
            self._UsdPhysics.MassAPI.Apply(root_prim)
        self._UsdPhysics.MassAPI(root_prim).CreateMassAttr(mass_kg)
        if not root_prim.HasAPI(self._PhysxSchema.PhysxRigidBodyAPI):
            self._PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        if ENABLE_CCD:
            self._PhysxSchema.PhysxRigidBodyAPI(root_prim).CreateEnableCCDAttr(True)
        for prim in self._Usd.PrimRange(root_prim):
            if not prim.IsValid() or not prim.IsA(self._UsdGeom.Gprim):
                continue
            if not prim.HasAPI(self._UsdPhysics.CollisionAPI):
                self._UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(self._UsdPhysics.MeshCollisionAPI):
                self._UsdPhysics.MeshCollisionAPI.Apply(prim)
            self._UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr(approximation)
            if not prim.HasAPI(self._PhysxSchema.PhysxCollisionAPI):
                self._PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision_api = self._PhysxSchema.PhysxCollisionAPI(prim)
            physx_collision_api.CreateContactOffsetAttr(0.005)
            physx_collision_api.CreateRestOffsetAttr(0.0)

    def _spawn_one_usd(self):
        prim_path = f"/World/SpawnedObjects/battery_{self._spawn_count}"
        pose_index = self._spawn_count % len(SPAWN_POSES)
        spawn_pose = SPAWN_POSES[pose_index]
        pos, rot_deg = spawn_pose["pos"], spawn_pose["rot_deg"]
        add_reference_to_stage(usd_path=SPAWN_USD_PATH, prim_path=prim_path)
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._set_transform(prim, pos, rot_deg, SPAWN_SCALE)
        self._apply_rigid_body_and_colliders(prim, mass_kg=SPAWN_MASS_KG, approximation=COLLIDER_APPROXIMATION)
        self._spawn_count += 1

    def _replace_case_with_full(self, idx):
        stage = omni.usd.get_context().get_stage()
        delete_paths = []
        if idx == 0:
            old_case_prim = BATTCASE2_PRIM_PATH
            full_usd = BATTFULL2_PATH
            full_prim = BATTFULL2_PRIM_PATH
            pos = BATTCASE2_WORLD_POS
            rot = BATTCASE2_ROT_DEG
        else:
            old_case_prim = BATTCASE1_PRIM_PATH
            full_usd = BATTFULL1_PATH
            full_prim = BATTFULL1_PRIM_PATH
            pos = BATTCASE1_WORLD_POS
            rot = BATTCASE1_ROT_DEG
        if stage.GetPrimAtPath(old_case_prim).IsValid():
            delete_paths.append(old_case_prim)
        target_mod = 1 if idx == 0 else 0
        for i in range(self._spawn_count):
            if i % 2 == target_mod:
                bat_path = f"/World/SpawnedObjects/battery_{i}"
                if stage.GetPrimAtPath(bat_path).IsValid():
                    delete_paths.append(bat_path)
        if delete_paths:
            omni.kit.commands.execute("DeletePrimsCommand", paths=delete_paths, destructive=True)
        add_reference_to_stage(usd_path=full_usd, prim_path=full_prim)
        new_prim = stage.GetPrimAtPath(full_prim)
        if not new_prim.IsValid():
            return
        xform = UsdGeom.Xformable(new_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rot[0]), float(rot[1]), float(rot[2])))
        xform.AddScaleOp().Set(Gf.Vec3f(*BATTCASE_SCALE))
        self._apply_rigid_body_and_collision_battcase(full_prim)
        print(f"[INFO] Side {idx} 꽉 찬 배터리 케이스 스폰 완료!")

    def _replace_full_and_case22_with_end(self, idx):
        if self._battend_created[idx]:
            return
        stage = omni.usd.get_context().get_stage()
        delete_paths = []
        if idx == 0:
            full_case_prim = BATTFULL2_PRIM_PATH
            cover_prim = BATTCASE2_2_PRIM_PATH
            end_usd = BATTEND_USD_PATH
            pos = BATTCASE2_WORLD_POS
            rot = BATTCASE2_ROT_DEG
        else:
            full_case_prim = BATTFULL1_PRIM_PATH
            cover_prim = BATTCASE1_2_PRIM_PATH
            end_usd = BATTEND1_USD_PATH
            pos = BATTCASE1_WORLD_POS
            rot = BATTCASE1_ROT_DEG
        if stage.GetPrimAtPath(full_case_prim).IsValid():
            delete_paths.append(full_case_prim)
        if stage.GetPrimAtPath(cover_prim).IsValid():
            delete_paths.append(cover_prim)
        target_mod = 1 if idx == 0 else 0
        for i in range(self._spawn_count):
            if i % 2 == target_mod:
                bat_path = f"/World/SpawnedObjects/battery_{i}"
                if stage.GetPrimAtPath(bat_path).IsValid():
                    delete_paths.append(bat_path)
        if delete_paths:
            omni.kit.commands.execute("DeletePrimsCommand", paths=delete_paths, destructive=True)
        add_reference_to_stage(usd_path=end_usd, prim_path=full_case_prim)
        new_prim = stage.GetPrimAtPath(full_case_prim)
        if not new_prim.IsValid():
            return
        xform = UsdGeom.Xformable(new_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rot[0]), float(rot[1]), float(rot[2])))
        xform.AddScaleOp().Set(Gf.Vec3f(*BATTCASE_SCALE))
        self._apply_rigid_body_and_collision_battcase(full_case_prim)
        self._battend_created[idx] = True
        print(f"[INFO] Side {idx} 최종 완성본 스폰 완료!")

    # ================================================================
    # 차량 관련
    # ================================================================
    def _spawn_random_car(self):
        usd_path = random.choice(CAR_USD_PATHS)
        prim_path = self.car_spawned_car_prim_path
        existing_prim = self._stage.GetPrimAtPath(prim_path)
        if existing_prim.IsValid():
            omni.kit.commands.execute("DeletePrimsCommand", paths=[prim_path], destructive=True)
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._set_transform(prim, CAR_SPAWN_POSITION, CAR_SPAWN_ROT_DEG, scale=(1.0, 1.0, 1.0))
        car_approximation = "convexDecomposition" if "ioniq5" in usd_path.lower() else COLLIDER_APPROXIMATION
        self._apply_rigid_body_and_colliders(prim, mass_kg=CAR_MASS_KG, approximation=car_approximation)
        self._car_spawned = True
        self.car_spawned_gt_label = "ioniq5" if "ioniq5" in usd_path.lower() else "ioniq6"
        self.car_moving_elapsed_time = 0.0
        self.car_stop_triggered = False
        self.car_stabilizing = False
        self.car_final_decision_done = False
        self.car_frozen = False
        self.car_stabilize_elapsed_sec = 0.0
        self.car_nav_sent = False
        if self.car_conveyor_attr is not None:
            self.car_conveyor_attr.Set(CAR_DEFAULT_CONVEYOR_VELOCITY)
        print(f"[INFO] 차량 스폰 완료 / GT = {self.car_spawned_gt_label}")

    def _get_spawned_car_root_prim(self):
        prim = self._stage.GetPrimAtPath(self.car_spawned_car_prim_path)
        return prim if prim.IsValid() else None

    def _force_stop_spawned_car(self):
        prim = self._get_spawned_car_root_prim()
        if prim is None or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        try:
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            vel_attr = rigid_api.GetVelocityAttr()
            if not vel_attr.IsValid():
                vel_attr = rigid_api.CreateVelocityAttr()
            ang_vel_attr = rigid_api.GetAngularVelocityAttr()
            if not ang_vel_attr.IsValid():
                ang_vel_attr = rigid_api.CreateAngularVelocityAttr()
            vel_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            ang_vel_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        except Exception:
            pass

    def _freeze_spawned_car(self):
        if self.car_frozen:
            return
        prim = self._get_spawned_car_root_prim()
        if prim is None:
            return
        try:
            from pxr import PhysxSchema
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            ccd_attr = PhysxSchema.PhysxRigidBodyAPI(prim).GetEnableCCDAttr()
            if not ccd_attr.IsValid():
                ccd_attr = PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr()
            ccd_attr.Set(False)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            kinematic_attr = rigid_api.GetKinematicEnabledAttr()
            if not kinematic_attr.IsValid():
                kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(True)
            self.car_frozen = True
        except Exception:
            pass

    def _car_step(self, step_size: float):
        if not self._car_spawned:
            self._car_elapsed_time += step_size
            if self._car_elapsed_time >= CAR_SPAWN_DELAY_SEC:
                self._spawn_random_car()
            return
        if not self.car_stop_triggered:
            self.car_moving_elapsed_time += step_size
            if self.car_moving_elapsed_time >= CAR_STOP_AFTER_SPAWN_SEC:
                if self.car_conveyor_attr is not None:
                    self.car_conveyor_attr.Set(0.0)
                self._force_stop_spawned_car()
                self.car_stop_triggered = True
                self.car_stabilizing = True
                self.car_stabilize_elapsed_sec = 0.0
                print(f"[CAR_INFO] 차량 컨베이어 정지 / GT = {self.car_spawned_gt_label}")
            return
        if self.car_stabilizing:
            self._force_stop_spawned_car()
            self.car_stabilize_elapsed_sec += step_size
            if self.car_stabilize_elapsed_sec >= CAR_STABILIZE_WAIT_SEC:
                self._freeze_spawned_car()
                self.car_stabilizing = False
                self.car_final_decision_done = True
                print(f"[CAR_INFO] 차량 고정 완료 / GT = {self.car_spawned_gt_label}")
            return

    def _check_and_launch_nav(self, force=False, reason=""):
        if self.car_nav_sent:
            return

        if not force and not all(self.all_scenarios_done):
            print("[NAV] 아직 양쪽 로봇암 동작이 모두 끝나지 않았습니다.")
            return

        if self.car_spawned_gt_label is None:
            print("[WARN] 차량 스폰 결과가 아직 없습니다. NAV 실행 불가.")
            return

        nav_cmd = NAV_COMMAND_BY_LABEL.get(self.car_spawned_gt_label)
        if nav_cmd is None:
            print(f"[WARN] 차량 라벨({self.car_spawned_gt_label})에 대응되는 NAV 명령이 없음")
            return

        try:
            command = (
                f"source {ROS_SETUP_BASH} && "
                f"source {WORKSPACE_SETUP_BASH} && "
                f"{nav_cmd}"
            )
            subprocess.Popen(["bash", "-lc", command])
            self.car_nav_sent = True

            if force:
                print(
                    f"[INFO] 강제 NAV 실행 / label={self.car_spawned_gt_label} "
                    f"/ cmd={nav_cmd} / reason={reason}"
                )
            else:
                print(
                    f"[INFO] 로봇암 동작 완료 → NAV 실행 "
                    f"/ label={self.car_spawned_gt_label} / cmd={nav_cmd}"
                )

        except Exception as exc:
            print(f"[ERROR] NAV 실행 실패: {exc}")

    def _on_physics_step(self, step_size):
        if self._spawn_count < MAX_SPAWN_COUNT:
            self._elapsed_time += step_size
            if self._elapsed_time >= SPAWN_INTERVAL_SEC:
                self._elapsed_time = 0.0
                self._spawn_one_usd()
        self._car_step(step_size)

    # ================================================================
    # 로봇 이동 / 로직
    # ================================================================
    def get_battery_pos(self, idx):
        target_pos = None
        for index in range(self.recent_i[idx], self._spawn_count, 2):
            path = f"/World/SpawnedObjects/battery_{index}"
            if not self._stage.GetPrimAtPath(path).IsValid():
                continue
            pick_path = path + "/PickPoint"
            if not self._stage.GetPrimAtPath(pick_path).IsValid():
                continue
            pos, _ = XFormPrim(pick_path).get_world_pose()
            if abs(pos[1] - PICK_ZONE_Y) < PICK_ZONE_THRESHOLD:
                target_pos = pos
                self.recent_i[idx] = index + 2
                break
        return target_pos

    def _move_to(self, robot, ctrl, pos, ori):
        action = ctrl.forward(target_end_effector_position=pos, target_end_effector_orientation=ori)
        robot.apply_action(action)
        curr_pos, _ = robot.end_effector.get_world_pose()
        dist = np.linalg.norm(curr_pos - pos)
        joint_diff = np.abs(robot.get_joint_positions()[:6] - action.joint_positions)
        is_stuck = np.all(joint_diff < STUCK_JOINT_TOL)
        is_arrived = dist < ARRIVE_DIST_TOL
        return (is_arrived and np.all(joint_diff < ARRIVE_JOINT_TOL)) or is_stuck

    def _r1_move_to(self, robot, ctrl, world_pos: np.ndarray, z_world: float = None):
        target = world_pos.copy()
        if z_world is not None:
            target[2] = z_world
        orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2.0, 0.0]))
        action = ctrl.forward(target_end_effector_position=target, target_end_effector_orientation=orientation)
        robot.apply_action(action)
        return action

    def _r1_rotate_joint1(self, robot, idx, delta_deg: float):
        current_positions = robot.get_joint_positions().copy()
        if self._r1_joint1_rotate_target[idx] is None:
            self._r1_joint1_rotate_target[idx] = current_positions.copy()
            self._r1_joint1_rotate_target[idx][0] = current_positions[0] + np.deg2rad(delta_deg)
        action = ArticulationAction(joint_positions=self._r1_joint1_rotate_target[idx])
        robot.apply_action(action)
        return action

    def _r1_arrived(self, robot, action) -> bool:
        current = robot.get_joint_positions()
        return bool(np.all(np.abs(current[:6] - action.joint_positions) < R1_ARRIVE_JOINT_TOL))

    def make_roi(self, camera):
        frame = camera.get_rgb()
        if frame is None:
            return None
        _, width, _ = frame.shape
        return frame[:, int(width * 0.42):int(width * 0.6)]

    def _run_yolo(self, idx):
        def yolo_callback(step_size):
            if self.yolo is None or not self.conveyor_running[idx]:
                return
            camera = self.camera1 if idx == 0 else self.camera2
            roi_frame = self.make_roi(camera)
            if roi_frame is None:
                return
            if roi_frame.shape[2] == 4:
                img_bgr = cv2.cvtColor(roi_frame, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR)
            results = self.yolo(img_bgr, verbose=False)
            height, _, _ = img_bgr.shape
            detect_trigger = False
            for box in results[0].boxes:
                _, y1, _, y2 = box.xyxy[0]
                cy = int((y1 + y2) / 2)
                if cy > height * 0.52:
                    detect_trigger = True
            if detect_trigger and self.conveyor_running[idx] and self.stop_timer[idx] is None:
                self.stop_timer[idx] = time.time() + self.wait_time
            if self.stop_timer[idx] is not None and self.conveyor_running[idx] and time.time() > self.stop_timer[idx]:
                self.conveyor_attrs[idx].Set(0.0)
                self.conveyor_running[idx] = False
            if not self.conveyor_running[idx] and not self.send_flag[idx]:
                pick_pos = self.get_battery_pos(idx)
                if pick_pos is not None:
                    pick_pos[0] += ROBOT2_PICK_X_OFFSET
                    pick_pos[1] += ROBOT2_PICK_Y_OFFSET
                    pick_pos[2] = ROBOT2_PICK_Z
                    self.r2_pos[idx] = np.array(pick_pos)
                    self.send_flag[idx] = not self.robot_flag[idx]
        return yolo_callback

    def physics_step(self, idx):
        def step_callback(step_size):
            down_ori = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))

            if idx == 0:
                robot_batt  = self.robots2
                ctrl_batt   = self.cspace_controller2
                gripper_batt = self.gripper2
                robot_cover  = self.robots
                ctrl_cover   = self.cspace_controller
                gripper_cover = self.gripper1
                r2_case  = np.array([-20.22, -4.95, 1.2445])
                r2_case2 = BATTCASE2_WORLD_POS
                r2_target2 = r2_case + np.array([0.5, 0, 0]) * self.i[0] + np.array([0, 0.7, 0]) * self.j[0]
                target_cover_pos = BATTCASE2_2_WORLD_POS
            else:
                robot_batt  = self.robots2_2
                ctrl_batt   = self.cspace_controller4
                gripper_batt = self.gripper3
                robot_cover  = self.robots_2
                ctrl_cover   = self.cspace_controller3
                gripper_cover = self.gripper4
                r2_case  = np.array([-0.44, -4.85, 1.3])
                r2_case2 = BATTCASE1_WORLD_POS
                r2_target2 = r2_case + np.array([-0.5, 0, 0]) * self.i[1] + np.array([0, 0.7, 0]) * self.j[1]
                target_cover_pos = BATTCASE1_2_WORLD_POS

            if self.r2_pos[idx] is not None:
                r2_target  = self.r2_pos[idx].copy()
                r2_approach = r2_target + np.array([0, 0, 0.4])
            else:
                r2_target, r2_approach = None, None

            r2_approach2 = r2_target2 + np.array([0, 0, 0.4])

            # ============================================================
            # Robot 2: 배터리 적재 로직
            # ============================================================
            if self.task_phase_r2[idx] == 0:
                if self.send_flag[idx] and self.r2_pos[idx] is not None and self.task_phase_r1[idx] == 0:
                    self.task_phase_r2[idx] = 1

            elif self.task_phase_r2[idx] == 1:
                self.robot_flag[idx] = True
                if self._move_to(robot_batt, ctrl_batt, r2_approach, down_ori):
                    self.task_phase_r2[idx] = 2

            elif self.task_phase_r2[idx] == 2:
                if self._move_to(robot_batt, ctrl_batt, r2_target, down_ori):
                    gripper_batt.close()
                    self.r2_pause_timer[idx] += 1
                    if self.r2_pause_timer[idx] > PICK_HOLD_STEPS:
                        self.r2_pause_timer[idx] = 0
                        self.task_phase_r2[idx] = 3

            elif self.task_phase_r2[idx] == 3:
                r2_lift = r2_target + np.array([0, 0, 0.4])
                if self._move_to(robot_batt, ctrl_batt, r2_lift, down_ori):
                    self.task_phase_r2[idx] = 4

            elif self.task_phase_r2[idx] == 4:
                if self._move_to(robot_batt, ctrl_batt, r2_approach2, down_ori):
                    self.task_phase_r2[idx] = 5

            elif self.task_phase_r2[idx] == 5:
                if self._move_to(robot_batt, ctrl_batt, r2_target2, down_ori):
                    self.r2_pause_timer[idx] += 1
                    if self.r2_pause_timer[idx] > POST_OPEN_WAIT_STEPS:
                        gripper_batt.open()
                        self.r2_pause_timer[idx] = 0
                        self.task_phase_r2[idx] = 6

            elif self.task_phase_r2[idx] == 6:
                if self._move_to(robot_batt, ctrl_batt, r2_approach2, down_ori):
                    self.send_flag[idx] = False
                    self.r2_pos[idx] = None
                    self.robot_flag[idx] = False
                    self.i[idx] += 1

                    if self.i[idx] == 2:
                        print(f"[Side {idx}] 완성본 소환 및 R1 기동")
                        self.conveyor_attrs[idx].Set(0.0)
                        self.conveyor_running[idx] = False
                        robot_batt.set_joint_positions(INITIAL_JOINTS)
                        ctrl_batt.reset()
                        # ★ R2는 phase 7(대기)로 → R1이 완료되면 phase 8로 전환해줌
                        self.task_phase_r2[idx] = 7
                        self._replace_case_with_full(idx)
                        self.task_phase_r1[idx] = 1

                    elif self.i[idx] == 3:
                        self.j[idx] += 1
                        self.i[idx] = 0

                    elif self.j[idx] == 2:
                        self.conveyor_attrs[idx].Set(0.0)

                    else:
                        self.task_phase_r2[idx] = 0
                        print(f"~~~ {self.i[idx]}번째 배터리 적재 완료 ~~~")
                        self.conveyor_attrs[idx].Set(-0.4)
                        self.conveyor_running[idx] = True
                        self.stop_timer[idx] = None

            elif self.task_phase_r2[idx] == 7:
                # R1(뚜껑 로봇) 작업 완료 대기
                # → R1 phase 10이 home에 도달하면 task_phase_r2[idx] = 8 로 전환해줌
                pass

            elif self.task_phase_r2[idx] == 8:
                r2_battendcase = BATTCASE2_WORLD_POS if idx == 0 else BATTCASE1_WORLD_POS
                r2_batt_approach = r2_battendcase + np.array([0, 0, 0.5])
                if self._move_to(robot_batt, ctrl_batt, r2_batt_approach, down_ori):
                    print(f"[Side {idx} - R2 Phase 8] 완성본 상공 도착, 하강 시작")
                    self.task_phase_r2[idx] = 9

            elif self.task_phase_r2[idx] == 9:
                r2_battendcase = BATTCASE2_WORLD_POS if idx == 0 else BATTCASE1_WORLD_POS + np.array([0, 0, 0.03])
                r2_batt_approach = r2_battendcase + np.array([0, 0, 0.5])
                if self._move_to(robot_batt, ctrl_batt, r2_battendcase, down_ori):
                    gripper_batt.close()
                    self.r2_pause_timer[idx] += 1
                    if self.r2_pause_timer[idx] > PICK_HOLD_STEPS:
                        print(f"[Side {idx} - R2 Phase 9] 완성본 흡착 완료!")
                        self.r2_pause_timer[idx] = 0
                        self.task_phase_r2[idx] = 10

            elif self.task_phase_r2[idx] == 10:
                r2_battendcase = BATTCASE2_WORLD_POS if idx == 0 else BATTCASE1_WORLD_POS
                r2_batt_approach = r2_battendcase + np.array([0, 0, 0.5])
                if self._move_to(robot_batt, ctrl_batt, r2_batt_approach, down_ori):
                    ctrl_batt.reset()
                    print(f"[Side {idx} - R2 Phase 10] 들어올리기 완료! 스윙 준비!")
                    self.task_phase_r2[idx] = 11

            elif self.task_phase_r2[idx] == 11:
                ROTATE_DEG = -90.0 if idx == 0 else 90.0
                current_positions = robot_batt.get_joint_positions().copy()
                if self._r2_joint1_target[idx] is None:
                    self._r2_joint1_target[idx] = current_positions.copy()
                    self._r2_joint1_target[idx][0] = current_positions[0] + np.deg2rad(ROTATE_DEG)
                action = ArticulationAction(joint_positions=self._r2_joint1_target[idx])
                robot_batt.apply_action(action)
                current_joints = robot_batt.get_joint_positions()[:6]
                target_joints  = self._r2_joint1_target[idx][:6]
                if np.all(np.abs(current_joints - target_joints) < ARRIVE_JOINT_TOL):
                    self._r2_joint1_target[idx] = None
                    self.r2_pause_timer[idx] = 0
                    print(f"[Side {idx} - R2 Phase 11] {ROTATE_DEG}도 스윙 완료!")
                    self.task_phase_r2[idx] = 12

            elif self.task_phase_r2[idx] == 12:
                self.r2_pause_timer[idx] += 1
                if self.r2_pause_timer[idx] >= POST_ROTATE_WAIT_STEPS:
                    self.r2_pause_timer[idx] = 0
                    print(f"[Side {idx} - R2 Phase 12] 안정화 완료, 그리퍼 열기!")
                    self.task_phase_r2[idx] = 13

            elif self.task_phase_r2[idx] == 13:
                # ★ 그리퍼 열고 충분히 대기 (케이스 안에 박혀 안 떨어지는 문제 방지)
                gripper_batt.open()
                self.r2_pause_timer[idx] += 1
                if self.r2_pause_timer[idx] >= POST_OPEN_WAIT_STEPS * 6:  # 약 1초(60step)로 연장
                    self.r2_pause_timer[idx] = 0
                    print(f"[Side {idx} - R2 Phase 13] 그리퍼 오픈 대기 완료, 위로 빠지기!")
                    self.task_phase_r2[idx] = 135  # 케이스에서 완전히 빠지는 단계 추가

            elif self.task_phase_r2[idx] == 135:
                if self._phase135_enter_time[idx] is None:
                    self._phase135_enter_time[idx] = time.time()
                    self._phase135_nav_forced[idx] = False
                    print(f"[Side {idx} - R2 Phase 135] 진입 시간 기록")

                elapsed_135 = time.time() - self._phase135_enter_time[idx]

                if elapsed_135 >= 10.0 and not self._phase135_nav_forced[idx]:
                    self._phase135_nav_forced[idx] = True
                    print(f"[WARN] Side {idx} phase 135가 10초 이상 지속됨 -> 강제 NAV 실행")
                    self._check_and_launch_nav(
                        force=True,
                        reason=f"side_{idx}_phase135_timeout_10s"
                    )

                r2_battendcase = BATTCASE2_WORLD_POS if idx == 0 else BATTCASE1_WORLD_POS
                lift_pos = r2_battendcase + np.array([0, 0, 0.5])

                if self._move_to(robot_batt, ctrl_batt, lift_pos, down_ori):
                    print(f"[Side {idx} - R2 Phase 135] 케이스 분리 완료!")
                    self._r2_joint1_target[idx] = None
                    self.r2_pause_timer[idx] = 0
                    self._phase135_enter_time[idx] = None
                    self._phase135_nav_forced[idx] = False
                    self.task_phase_r2[idx] = 136

            elif self.task_phase_r2[idx] == 136:
                action = ArticulationAction(joint_positions=INITIAL_JOINTS)
                robot_batt.apply_action(action)
                current_joints = robot_batt.get_joint_positions()[:6]
                if np.all(np.abs(current_joints - INITIAL_JOINTS) < HOME_JOINT_TOL):
                    print(f"[Side {idx} - R2 Phase 136] home 복귀 완료!")
                    self._phase135_enter_time[idx] = None
                    self._phase135_nav_forced[idx] = False
                    self.task_phase_r2[idx] = 14

            elif self.task_phase_r2[idx] == 14:
                action = ArticulationAction(joint_positions=INITIAL_JOINTS)
                robot_batt.apply_action(action)
                current_joints = robot_batt.get_joint_positions()[:6]
                if np.all(np.abs(current_joints - INITIAL_JOINTS) < HOME_JOINT_TOL):
                    print(f"[Side {idx}] 모든 시나리오 종료!")
                    self._phase135_enter_time[idx] = None
                    self._phase135_nav_forced[idx] = False
                    self.task_phase_r2[idx] = 0
                    self.all_scenarios_done[idx] = True
                    if all(self.all_scenarios_done):
                        print("[INFO] 양쪽 모든 로봇 동작 완료 → NAV 명령 실행 시도")
                        self._check_and_launch_nav()

            # ============================================================
            # Robot 1: Cover 로직
            # ============================================================
            r1_pick_z_world = target_cover_pos[2] + 0.01

            if self.task_phase_r1[idx] == 1:
                action = self._r1_move_to(robot_cover, ctrl_cover, target_cover_pos, z_world=R1_APPROACH_Z_WORLD)
                if self._r1_arrived(robot_cover, action):
                    ctrl_cover.reset()
                    print(f"[Side {idx} - R1 Phase 1] 뚜껑 상공 어프로치 완료")
                    self.task_phase_r1[idx] = 2

            elif self.task_phase_r1[idx] == 2:
                action = self._r1_move_to(robot_cover, ctrl_cover, target_cover_pos, z_world=r1_pick_z_world)
                if self._r1_arrived(robot_cover, action):
                    ctrl_cover.reset()
                    print(f"[Side {idx} - R1 Phase 2] 뚜껑 표면 도착")
                    self.task_phase_r1[idx] = 3

            elif self.task_phase_r1[idx] == 3:
                gripper_cover.close()
                print(f"[Side {idx} - R1 Phase 3] 뚜껑 흡착 완료")
                self.task_phase_r1[idx] = 4

            elif self.task_phase_r1[idx] == 4:
                action = self._r1_move_to(robot_cover, ctrl_cover, target_cover_pos, z_world=R1_APPROACH_Z_WORLD)
                if self._r1_arrived(robot_cover, action):
                    ctrl_cover.reset()
                    self._r1_joint1_rotate_target[idx] = None
                    print(f"[Side {idx} - R1 Phase 4] 뚜껑 들어올리기 완료")
                    self.task_phase_r1[idx] = 5

            elif self.task_phase_r1[idx] == 5:
                action = self._r1_rotate_joint1(robot_cover, idx, R1_JOINT1_ROTATE_DEG)
                if self._r1_arrived(robot_cover, action):
                    ctrl_cover.reset()
                    self._r1_joint1_rotate_target[idx] = None
                    self._r1_pause_timer[idx] = 0
                    print(f"[Side {idx} - R1 Phase 5] 턴 완료!")
                    self.task_phase_r1[idx] = 6

            elif self.task_phase_r1[idx] == 6:
                self._r1_pause_timer[idx] += 1
                if self._r1_pause_timer[idx] >= R1_POST_ROTATE_WAIT_STEPS:
                    self._r1_pause_timer[idx] = 0
                    self.task_phase_r1[idx] = 7

            elif self.task_phase_r1[idx] == 7:
                gripper_cover.open()
                self._r1_pause_timer[idx] = 0
                print(f"[Side {idx} - R1 Phase 7] 뚜껑 덮기 완료")
                self.task_phase_r1[idx] = 8

            elif self.task_phase_r1[idx] == 8:
                self._r1_pause_timer[idx] += 1
                if self._r1_pause_timer[idx] >= R1_POST_OPEN_WAIT_STEPS:
                    self.task_phase_r1[idx] = 9

            elif self.task_phase_r1[idx] == 9:
                self._replace_full_and_case22_with_end(idx)
                print(f"[Side {idx} - R1 Phase 9] 최종 완성본 소환 완료!")
                self.task_phase_r1[idx] = 10

            elif self.task_phase_r1[idx] == 10:
                # ★ 핵심 수정: home 복귀 완료 시 딱 한 번만 R2에게 phase 8 신호
                action = ArticulationAction(joint_positions=INITIAL_JOINTS)
                robot_cover.apply_action(action)
                current_cover_joints = robot_cover.get_joint_positions()[:6]
                if np.all(np.abs(current_cover_joints - INITIAL_JOINTS) < HOME_JOINT_TOL):
                    print(f"[Side {idx} - R1 Phase 10] 뚜껑 로봇 home 복귀 완료 → R2 완성본 픽업 시작")
                    self.task_phase_r1[idx] = 11        # R1 완전 종료
                    self.task_phase_r2[idx] = 8         # R2 픽업 시작 (딱 한 번만!)

            # task_phase_r1[idx] == 11: R1 완전 종료, 아무 동작 없음

        return step_callback
