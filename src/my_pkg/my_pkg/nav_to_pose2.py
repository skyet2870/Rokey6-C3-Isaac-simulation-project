#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


class NavLiftController(Node):

    def __init__(self):
        super().__init__("nav_lift_controller")
        self.nav = BasicNavigator()
        self.lift_pub = self.create_publisher(JointState, "/lift_joint", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def make_pose(self, x, y, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        return pose

    def move_lift_slowly(self, start=0.0, target=0.2, steps=50, duration=3.0):
        current = start
        step_size = (target - start) / steps
        sleep_time = duration / steps

        for _ in range(steps + 1):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ["lift_joint"]
            msg.position = [current]

            self.lift_pub.publish(msg)
            self.get_logger().info(f"Lift position: {current:.4f}")

            current += step_size
            time.sleep(sleep_time)

    def set_lift_position(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["lift_joint"]
        msg.position = [position]

        self.lift_pub.publish(msg)
        self.get_logger().info(f"Lift instant set: {position:.4f}")

    def back_up(self, speed=-0.2, duration=3.0):
        msg = Twist()
        msg.linear.x = speed

        start = time.time()
        while time.time() - start < duration:
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.1)

        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info("후진 완료")

    def rotate_slowly(self, angular_speed=0.12, duration=2.5):
        msg = Twist()
        msg.angular.z = angular_speed

        start = time.time()
        while time.time() - start < duration:
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.1)

        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info("천천히 회전 완료")

    def wait_nav_result(self, label):
        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            if feedback:
                self.get_logger().info(
                    f"{label} distance remaining: {feedback.distance_remaining:.3f}"
                )
            time.sleep(0.1)

        result = self.nav.getResult()

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info(f"{label} 성공")
            return True
        if result == TaskResult.CANCELED:
            self.get_logger().error(f"{label} 취소됨")
            return False
        if result == TaskResult.FAILED:
            self.get_logger().error(f"{label} 실패")
            return False

        self.get_logger().error(f"{label} 알 수 없는 결과")
        return False

    def run(self):
        init_pose = self.make_pose(-10.0, -21.1, 1.57)

        self.nav.setInitialPose(init_pose)
        time.sleep(1.0)
        self.nav.waitUntilNav2Active()
        time.sleep(1.0)
        pose1 = self.make_pose(-18.5, -8.0, 1.83)
        pose2 = self.make_pose(-18.5, -6.2, 1.83)

        self.get_logger().info("첫번째 이동 시작")
        self.nav.goThroughPoses([pose1, pose2])

        ok = self.wait_nav_result("첫번째 이동")
        if not ok:
            self.get_logger().error("첫번째 이동 실패로 종료")
            return

        self.get_logger().info("첫번째 목적지 도착")
        time.sleep(1.0)

        self.get_logger().info("리프트 상승 시작")
        self.move_lift_slowly(start=0.0, target=0.2, steps=50, duration=3.0)
        time.sleep(1.0)

        self.get_logger().info("후진 시작")
        self.back_up(speed=-0.8, duration=10.0)
        time.sleep(1.0)

        self.get_logger().info("천천히 회전 시작")
        self.rotate_slowly(angular_speed=-0.5, duration=4)
        time.sleep(2.0)

        self.get_logger().info("두번째 이동 시작")
        final_pose = self.make_pose(-11.9, 22.0, 1.83)
        self.nav.goToPose(final_pose)

        ok = self.wait_nav_result("두번째 이동")
        if not ok:
            self.get_logger().error("두번째 이동 실패")
            return

        self.get_logger().info("두번째 목적지 도착")
        time.sleep(1.0)

        self.get_logger().info("리프트 즉시 하강")
        self.set_lift_position(0.0)
        time.sleep(0.5)

        self.get_logger().info("복귀 전 후진 시작")
        self.back_up(speed=-0.8, duration=10.0)
        time.sleep(1.0)

        self.get_logger().info("초기 위치로 복귀 시작")
        return_pose = self.make_pose(-10.0, -21.1, 0.0)
        self.nav.goToPose(return_pose)

        ok = self.wait_nav_result("복귀 이동")
        if not ok:
            self.get_logger().error("복귀 이동 실패")
            return

        self.get_logger().info("초기 위치 복귀 완료")
        self.get_logger().info("작업 완료")


def main():
    rclpy.init()
    node = NavLiftController()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
