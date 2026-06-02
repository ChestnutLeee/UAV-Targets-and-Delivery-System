#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

# 导入 Gazebo 与 AprilTag 核心依赖
try:
    from dt_apriltags import Detector
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10.image_pb2 import Image as GzImage
except ImportError as e:
    print(f"依赖缺失: {e}")
    exit(1)

class UnifiedUAVEvaluator(Node):
    def __init__(self):
        super().__init__('unified_uav_evaluator')
        
        # --- 1. 量化指标统计 (固定1000帧样本) ---
        self.n_total = 0      
        self.n_correct = 0    
        self.n_lost = 0       
        self.max_frames = 1000 
        self.eval_finished = False 
        
        # --- 2. 高精度检测器配置 ---
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,  
            quad_sigma=0.8,     
            refine_edges=1,     
            decode_sharpening=0.25,
            debug=0
        )
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # 画面缓存
        self.display_img = None
        self.any_tag_this_frame = False # 当前采样周期内是否有任意相机识别到
        
        # --- 3. 订阅两个摄像头 ---
        self.gz_node = GzTransportNode()
        self.gz_node.subscribe(GzImage, "/slot_machine_camera/front", self.front_callback)
        self.gz_node.subscribe(GzImage, "/slot_machine_camera/down", self.down_callback)
        
        # 统计与显示主循环
        self.timer = self.create_timer(0.04, self.main_loop)
        
        self.get_logger().info("综合视觉评估系统已启动：前视或下视任意识别即计入统计，目标1000帧")

    def front_callback(self, msg):
        self.process_incoming(msg, "FRONT")

    def down_callback(self, msg):
        self.process_incoming(msg, "DOWN")

    def process_incoming(self, msg, label):
        """通用图像处理：识别并标记"""
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.shape[2] == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray) 
        
        tags = self.detector.detect(gray)
        
        # 只要有一个相机识别到，就标记本轮识别成功
        if tags:
            self.any_tag_this_frame = True
            for tag in tags:
                corners = tag.corners.astype(np.int32)
                cv2.polylines(img, [corners], True, (0, 255, 0), 2)
                cv2.putText(img, f"{label}_ID:{tag.tag_id}", (int(tag.center[0]), int(tag.center[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(img, f"Active: {label}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        self.display_img = img # 实时更新显示画面

    def main_loop(self):
        """执行统计与UI渲染"""
        if self.display_img is None:
            return

        # --- 统计逻辑 ---
        if not self.eval_finished:
            self.n_total += 1
            if self.any_tag_this_frame:
                self.n_correct += 1
            else:
                self.n_lost += 1
            
            # 重置本帧标志位，等待下一轮相机回调更新
            self.any_tag_this_frame = False

            if self.n_total >= self.max_frames:
                self.eval_finished = True
                self.get_logger().info(f"评估完成！总帧数: {self.max_frames}, 识别成功: {self.n_correct}")

        # --- 绘制 UI ---
        canvas = self.display_img.copy()
        p_acc = (self.n_correct / self.n_total * 100) if self.n_total > 0 else 0.0
        
        # 统计面板
        cv2.rectangle(canvas, (10, 10), (320, 110), (40, 40, 40), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        status_txt = f"Progress: {self.n_total}/{self.max_frames}" if not self.eval_finished else "STATUS: FINISHED"
        cv2.putText(canvas, status_txt, (20, 35), font, 0.6, (0, 255, 255), 1)
        cv2.putText(canvas, f"P_acc: {p_acc:.2f}%", (20, 65), font, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Correct: {self.n_correct} | Lost: {self.n_lost}", (20, 95), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Multi-Camera Unified Evaluation", canvas)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedUAVEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()