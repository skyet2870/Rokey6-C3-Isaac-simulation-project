#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time

class LiftController(Node):
    def __init__(self):
        super().__init__('lift_controller')
        self.pub = self.create_publisher(JointState, '/lift_joint', 10)
    
    def lift_slowly(self, target=0.2, steps=50, duration=3.0):
        current = 0.0
        step_size = target / steps
        sleep_time = duration / steps
        
        for i in range(steps + 1):
            msg = JointState()
            msg.name = ['lift_joint']
            msg.position = [current]
            self.pub.publish(msg)
            print(f"Position: {current:.4f}")
            current += step_size
            time.sleep(sleep_time)

def main():
    rclpy.init()
    node = LiftController()
    node.lift_slowly(target=0.2, steps=50, duration=3.0)
    rclpy.shutdown()

if __name__ == '__main__':
    main()