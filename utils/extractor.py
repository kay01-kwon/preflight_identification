"""
ROS2 Bag extractor for moment excitation analysis

Changes from original:
  - Fixed _TYPE_CONVERTERS: PoseStamped was mapped to _convert_hexa_rpm
  - Fixed _convert_pose: PoseStamped has no child_frame_id
  - Fixed load_all: PoseStamped was not handled
  - t_rel now uses odom t[0] as global reference across all topics
"""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# =====================================
# Timestamp helper
# =====================================

def stamp_to_sec(stamp) -> float:
    """Convert builtin_interfaces/Time to seconds."""
    return stamp.sec + stamp.nanosec * 1e-9

# =====================================
# Data containers
# =====================================

@dataclass
class OdometryData:
    """Vectorized nav_msgs/msg/Odometry"""
    t: np.ndarray           # (N,) absolute time [s]
    position: np.ndarray    # (N,3) [x, y, z] [m]
    quaternion: np.ndarray  # (N,4) [qw, qx, qy, qz] Body -> World
    linear_vel: np.ndarray  # (N,3) [vx, vy, vz] [m/s] Expressed in body frame
    angular_vel: np.ndarray # (N,3) [wx, wy, wz] [rad/s] Expressed in body frame
    frame_id: str
    child_frame_id: str

    @property
    def N(self) -> int:
        return len(self.t)

    def t_rel(self, t0: float) -> np.ndarray:
        """Time relative to a global reference t0."""
        return self.t - t0


@dataclass
class PoseData:
    """Vectorized geometry_msgs/msg/PoseStamped"""
    t: np.ndarray           # (N,) absolute time [s]
    position: np.ndarray    # (N,3) [x, y, z] [m]
    quaternion: np.ndarray  # (N,4) [qw, qx, qy, qz]
    frame_id: str

    @property
    def N(self) -> int:
        return len(self.t)

    def t_rel(self, t0: float) -> np.ndarray:
        """Time relative to a global reference t0."""
        return self.t - t0


@dataclass
class ImuData:
    """Vectorized sensor_msgs/msg/Imu (e.g. /mavros/imu/data_raw)"""
    t: np.ndarray           # (N,) absolute time [s]
    angular_vel: np.ndarray # (N,3) [wx, wy, wz] [rad/s] body frame
    linear_acc: np.ndarray  # (N,3) [ax, ay, az] [m/s²] body frame
    quaternion: np.ndarray  # (N,4) [qw, qx, qy, qz] (may be all-zero for data_raw)
    frame_id: str

    @property
    def N(self) -> int:
        return len(self.t)

    def t_rel(self, t0: float) -> np.ndarray:
        """Time relative to a global reference t0."""
        return self.t - t0


@dataclass
class HexaRpmData:
    """Vectorized ros2_libcanard_msgs/msg/HexaActualRpm"""
    t: np.ndarray       # (N,) [s]
    rpm: np.ndarray     # (N,6) [rpm]
    acc: np.ndarray     # (N,6) [rpm/s]
    frame_id: str

    @property
    def N(self) -> int:
        return len(self.t)

    def t_rel(self, t0: float) -> np.ndarray:
        """Time relative to a global reference t0."""
        return self.t - t0


# Type alias for any data container
TopicData = Union[OdometryData, PoseData, HexaRpmData, ImuData]

# ===============================================
# Message -> dictionary converters
# ===============================================

def _convert_odometry(msg) -> dict:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    v = msg.twist.twist.linear
    w = msg.twist.twist.angular
    return dict(
        t=stamp_to_sec(msg.header.stamp),
        frame_id=msg.header.frame_id,
        child_frame_id=msg.child_frame_id,
        px=p.x, py=p.y, pz=p.z,
        qw=q.w, qx=q.x, qy=q.y, qz=q.z,
        vx=v.x, vy=v.y, vz=v.z,
        wx=w.x, wy=w.y, wz=w.z,
    )


def _convert_pose(msg) -> dict:
    p = msg.pose.position
    q = msg.pose.orientation
    return dict(
        t=stamp_to_sec(msg.header.stamp),
        frame_id=msg.header.frame_id,
        px=p.x, py=p.y, pz=p.z,
        qw=q.w, qx=q.x, qy=q.y, qz=q.z,
    )


def _convert_imu(msg) -> dict:
    w = msg.angular_velocity
    a = msg.linear_acceleration
    q = msg.orientation
    return dict(
        t=stamp_to_sec(msg.header.stamp),
        frame_id=msg.header.frame_id,
        wx=w.x, wy=w.y, wz=w.z,
        ax=a.x, ay=a.y, az=a.z,
        qw=q.w, qx=q.x, qy=q.y, qz=q.z,
    )


def _convert_hexa_rpm(msg) -> dict:
    rpms = np.array(msg.rpm[:6], dtype=np.int32)
    accs = np.array(msg.acceleration[:6], dtype=np.int32)
    return dict(
        t=stamp_to_sec(msg.header.stamp),
        frame_id=msg.header.frame_id,
        rpm=rpms,
        acc=accs,
    )


_TYPE_CONVERTERS = {
    "nav_msgs/msg/Odometry":                _convert_odometry,
    "geometry_msgs/msg/PoseStamped":        _convert_pose,
    "sensor_msgs/msg/Imu":                  _convert_imu,
    "ros2_libcanard_msgs/msg/HexaActualRpm": _convert_hexa_rpm,
}

# ====================================================
# Ros2 Bag extractor
# ====================================================

class RosBagExtractor:
    """
    Extract data from a single ROS2 bag file.

    Parameters
    ----------
    bag_path : str or Path
        Path to the bag directory (containing metadata.yaml and .db3)
        or directly to a .db3 file.
    """

    def __init__(self, bag_path: str | Path):
        bag_path = Path(bag_path)
        if bag_path.suffix == '.db3':
            bag_path = bag_path.parent
        if not bag_path.is_dir():
            raise FileNotFoundError(f'Bag path {bag_path} does not exist')

        self._bag_path = bag_path
        self._bag_name = bag_path.name

        # Open via Sequential reader
        reader = rosbag2_py.SequentialReader()
        storage_opts = rosbag2_py.StorageOptions(
            uri=str(bag_path), storage_id='sqlite3'
        )
        converter_opts = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )
        reader.open(storage_opts, converter_opts)

        # topic_name → type string
        self._topic_types: dict[str, str] = {}
        # topic_name → Python msg class
        self._msg_classes: dict[str, type] = {}

        for info in reader.get_all_topics_and_types():
            self._topic_types[info.name] = info.type
            try:
                self._msg_classes[info.name] = get_message(info.type)
            except Exception:
                pass  # custom msg not installed

        # ── Buffer all messages ──
        self._raw: dict[str, list[tuple[int, bytes]]] = {
            t: [] for t in self._topic_types
        }
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            self._raw[topic].append((timestamp, data))
        del reader

    # ── info ───────────────────────────────────────────────

    @property
    def bag_name(self) -> str:
        return self._bag_name

    @property
    def topic_names(self) -> list[str]:
        return list(self._topic_types.keys())

    @property
    def topic_types(self) -> dict[str, str]:
        return dict(self._topic_types)

    def message_count(self, topic: str) -> int:
        return len(self._raw[topic])

    def print_info(self):
        print(f"Bag : {self._bag_path}")
        print(f"{'Topic':<45} {'Type':<50} {'Count':>6}")
        print("-" * 105)
        for name, mtype in self._topic_types.items():
            print(f"{name:<45} {mtype:<50} {self.message_count(name):>6}")

    # ── deserialization ────────────────────────────────────

    def _deserialize_all(self, topic: str) -> list:
        if topic not in self._topic_types:
            raise KeyError(f"Topic '{topic}' not found. Available: {self.topic_names}")
        if topic not in self._msg_classes:
            raise RuntimeError(
                f"Message class for '{self._topic_types[topic]}' not available. "
                f"Is the package installed and sourced?"
            )
        cls = self._msg_classes[topic]
        return [deserialize_message(data, cls) for _, data in self._raw[topic]]

    def _read_converted(self, topic: str) -> list[dict]:
        mtype = self._topic_types[topic]
        converter = _TYPE_CONVERTERS.get(mtype)
        if converter is None:
            raise NotImplementedError(
                f"No converter for '{mtype}'. "
                f"Supported: {list(_TYPE_CONVERTERS.keys())}. "
                f"Use get_raw_messages() for unsupported types."
            )
        return [converter(m) for m in self._deserialize_all(topic)]

    # ── typed accessors ───────────────────────────────────

    def get_odometry(self, topic: str) -> OdometryData:
        msgs = self._read_converted(topic)
        if not msgs:
            raise ValueError(f"No messages on '{topic}'")
        return OdometryData(
            t=np.array([m["t"] for m in msgs]),
            position=np.array([[m["px"], m["py"], m["pz"]] for m in msgs]),
            quaternion=np.array([[m["qw"], m["qx"], m["qy"], m["qz"]] for m in msgs]),
            linear_vel=np.array([[m["vx"], m["vy"], m["vz"]] for m in msgs]),
            angular_vel=np.array([[m["wx"], m["wy"], m["wz"]] for m in msgs]),
            frame_id=msgs[0]["frame_id"],
            child_frame_id=msgs[0]["child_frame_id"],
        )

    def get_pose(self, topic: str) -> PoseData:
        msgs = self._read_converted(topic)
        if not msgs:
            raise ValueError(f"No messages on '{topic}'")
        return PoseData(
            t=np.array([m["t"] for m in msgs]),
            position=np.array([[m["px"], m["py"], m["pz"]] for m in msgs]),
            quaternion=np.array([[m["qw"], m["qx"], m["qy"], m["qz"]] for m in msgs]),
            frame_id=msgs[0]["frame_id"],
        )

    def get_imu(self, topic: str) -> ImuData:
        msgs = self._read_converted(topic)
        if not msgs:
            raise ValueError(f"No messages on '{topic}'")
        return ImuData(
            t=np.array([m["t"] for m in msgs]),
            angular_vel=np.array([[m["wx"], m["wy"], m["wz"]] for m in msgs]),
            linear_acc=np.array([[m["ax"], m["ay"], m["az"]] for m in msgs]),
            quaternion=np.array([[m["qw"], m["qx"], m["qy"], m["qz"]] for m in msgs]),
            frame_id=msgs[0]["frame_id"],
        )

    def get_hexa_rpm(self, topic: str) -> HexaRpmData:
        msgs = self._read_converted(topic)
        if not msgs:
            raise ValueError(f"No messages on '{topic}'")
        return HexaRpmData(
            t=np.array([m["t"] for m in msgs]),
            rpm=np.vstack([m["rpm"] for m in msgs]),
            acc=np.vstack([m["acc"] for m in msgs]),
            frame_id=msgs[0]["frame_id"],
        )

    # ── auto-load all ─────────────────────────────────────

    def load_all(self) -> dict[str, TopicData]:
        """
        Load all topics with known types.

        Time is kept as absolute; use BagData.t0 for relative time.
        """
        result = {}
        for name, mtype in self._topic_types.items():
            if mtype not in _TYPE_CONVERTERS:
                continue
            try:
                if mtype == "nav_msgs/msg/Odometry":
                    result[name] = self.get_odometry(name)
                elif mtype == "geometry_msgs/msg/PoseStamped":
                    result[name] = self.get_pose(name)
                elif mtype == "sensor_msgs/msg/Imu":
                    result[name] = self.get_imu(name)
                elif mtype == "ros2_libcanard_msgs/msg/HexaActualRpm":
                    result[name] = self.get_hexa_rpm(name)
            except Exception as e:
                print(f"  [WARN] Skipping '{name}': {e}")

        return result

    # ── raw access ────────────────────────────────────────

    def get_raw_messages(self, topic: str) -> list:
        """Deserialized ROS msg objects for unsupported types."""
        return self._deserialize_all(topic)

    def get_raw_bytes(self, topic: str) -> list[tuple[int, bytes]]:
        """Raw (timestamp_ns, cdr_bytes) for manual parsing."""
        if topic not in self._raw:
            raise KeyError(f"Topic '{topic}' not found.")
        return self._raw[topic]


# ═════════════════════════════════════════════════════════════
#  BagData  —  per-bag struct with named fields
# ═════════════════════════════════════════════════════════════

@dataclass
class BagData:
    """
    One excitation bag unpacked into named fields.

    The reference time t0 is taken from odom.t[0].
    Use data.t_rel(bag.t0) for time relative to odom start.

    Fields
    ------
    name       : str           bag folder name
    odom       : OdometryData  EKF2 fused odometry
    pose       : PoseData      mocap pose (/S550/pose)
    rpm        : HexaRpmData   actual per-motor RPM
    imu        : ImuData | None  raw IMU (/mavros/imu/data_raw), optional
    """
    name: str
    odom: OdometryData
    pose: PoseData
    rpm: HexaRpmData
    imu: Optional[ImuData] = None

    @property
    def t0(self) -> float:
        """Global reference time = odom.t[0]."""
        return self.odom.t[0]


# ═════════════════════════════════════════════════════════════
#  Dataset loader  —  scans a directory of bag folders
# ═════════════════════════════════════════════════════════════

def load_dataset(
    dataset_dir: str | Path,
) -> OrderedDict[str, dict[str, TopicData]]:
    """
    Scan *dataset_dir* for bag subdirectories, load each one.

    Returns
    -------
    OrderedDict  { bag_folder_name : { topic_name : Data } }
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    bag_dirs = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    ])

    if not bag_dirs:
        raise FileNotFoundError(
            f"No bag directories found in {dataset_dir}. "
            f"Each bag folder must contain a metadata.yaml file."
        )

    result = OrderedDict()
    for bag_dir in bag_dirs:
        name = bag_dir.name
        print(f"Loading {name} ...")
        ext = RosBagExtractor(bag_dir)
        ext.print_info()
        result[name] = ext.load_all()
        print()

    return result


def load_excitation_dataset(
    dataset_dir: str | Path,
) -> list[BagData]:
    """
    Load all bags in *dataset_dir* and return a list of BagData.

    Time reference (t0) is odom.t[0] per bag.
    Use bag.t0 to get relative times:
        t_rel = bag.odom.t_rel(bag.t0)
        t_rel = bag.pose.t_rel(bag.t0)
        t_rel = bag.rpm.t_rel(bag.t0)

    Parameters
    ----------
    dataset_dir : path
        e.g. "DataSet/2026_05_20/exp/My"

    Returns
    -------
    list[BagData]   sorted alphabetically

    Example
    -------
    >>> bags = load_excitation_dataset("DataSet/2026_05_20/exp/My")
    >>> for bag in bags:
    ...     t = bag.odom.t_rel(bag.t0)
    ...     plt.plot(t, bag.odom.angular_vel[:, 1], label=bag.name)
    """
    dataset = load_dataset(dataset_dir)

    result: list[BagData] = []
    for bag_name, topics in dataset.items():
        odom = topics.get("/mavros/local_position/odom")
        pose = topics.get("/S550/pose")
        rpm  = topics.get("/uav/actual_rpm")
        imu  = topics.get("/mavros/imu/data_raw")

        if odom is None:
            print(f"  [WARN] {bag_name}: missing /mavros/local_position/odom, skipping")
            continue
        if pose is None:
            print(f"  [WARN] {bag_name}: missing /S550/pose, skipping")
            continue
        if rpm is None:
            print(f"  [WARN] {bag_name}: missing /uav/actual_rpm, skipping")
            continue
        if imu is None:
            print(f"  [INFO] {bag_name}: no /mavros/imu/data_raw "
                  f"(IMU-based onset detection unavailable for this bag)")

        result.append(BagData(
            name=bag_name,
            odom=odom,
            pose=pose,
            rpm=rpm,
            imu=imu,
        ))

    return result