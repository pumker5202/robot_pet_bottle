#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    channel_type = LaunchConfiguration("channel_type", default="serial")
    serial_port = LaunchConfiguration("serial_port", default="/dev/ttyUSB1")
    serial_baudrate = LaunchConfiguration("serial_baudrate", default="115200")
    frame_id = LaunchConfiguration("frame_id", default="laser")
    inverted = LaunchConfiguration("inverted", default="false")
    angle_compensate = LaunchConfiguration("angle_compensate", default="true")
    scan_mode = LaunchConfiguration("scan_mode", default="Standard")

    sllidar_node = Node(
        package="slampibot_robot",
        executable="sllidar_node",
        name="sllidar_node",
        parameters=[
            {
                "channel_type": channel_type,
                "serial_port": serial_port,
                "serial_baudrate": serial_baudrate,
                "frame_id": frame_id,
                "inverted": inverted,
                "angle_compensate": angle_compensate,
                "scan_mode": scan_mode,
            }
        ],
        output="screen",
    )

    static_tf_cmd = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0.1",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "base_link",
            "--child-frame-id",
            "laser",
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "channel_type",
                default_value=channel_type,
                description="LiDAR communication channel type",
            ),
            DeclareLaunchArgument(
                "serial_port",
                default_value=serial_port,
                description="USB serial port connected to LiDAR",
            ),
            DeclareLaunchArgument(
                "serial_baudrate",
                default_value=serial_baudrate,
                description="Serial baudrate for LiDAR",
            ),
            DeclareLaunchArgument(
                "frame_id",
                default_value=frame_id,
                description="Frame ID used in LaserScan messages",
            ),
            DeclareLaunchArgument(
                "inverted",
                default_value=inverted,
                description="Whether scan data should be inverted",
            ),
            DeclareLaunchArgument(
                "angle_compensate",
                default_value=angle_compensate,
                description="Whether angle compensation is enabled",
            ),
            DeclareLaunchArgument(
                "scan_mode",
                default_value=scan_mode,
                description="Scan mode name supported by the device",
            ),
            sllidar_node,
            static_tf_cmd,
        ]
    )
