#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math

try:
    from dt_apriltags import Detector
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10.image_pb2 import Image as GzImage
except ImportError as e:
    print(f"依赖缺失: {e}")
    exit(1)

class PrecisionEvaluator(Node):
    def __init__(self):
        super().__init__('precision_evaluator')
        
        # --- 1. 毕业设计指标统计 ---
        self.max_frames = 1000
        self.n_total = 0
        self.eval_finished = False
        
        # 统计变量
        self.sum_pos_error = 0.0  
        self.sum_att_error = 0.0  
        self.valid_samples = 0    
        
        # 锁定值缓存：用于采样完成后固定显示数值
        self.final_avg_pos = 0.0
        self.final_avg_att = 0.0
        self.final_valid_count = 0
        
        # --- 2. 视觉解算参数 ---
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0)
        # 使用你代码中的相机内参
        self.camera_matrix = np.array([[600.0, 0, 400.0], [0, 600.0, 300.0], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)
        self.tag_size = 0.8  
        s = self.tag_size / 2.0 
        self.tag_3d_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)

        # --- 3. 图像订阅 ---
        self.current_img = None
        self.gz_node = GzTransportNode()
        self.gz_node.subscribe(GzImage, "/slot_machine_camera/down", self.camera_callback)
        
        self.timer = self.create_timer(0.04, self.update_display)
        self.get_logger().info("精度评估系统：1000帧采样后指标将锁定，画面保持实时。")

    def camera_callback(self, msg):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR) if raw.shape[2] == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        tags = self.detector.detect(gray)
        
        # --- 统计逻辑 ---
        if not self.eval_finished:
            self.n_total += 1
            if tags:
                tag = max(tags, key=lambda x: x.decision_margin)
                success, rvec, tvec = cv2.solvePnP(self.tag_3d_pts, tag.corners.astype(np.float32), 
                                                  self.camera_matrix, self.dist_coeffs)
                if success:
                    # 计算位置误差 E_pos (cm)
                    tx, ty = tvec[0][0], tvec[1][0]
                    current_e_pos = math.sqrt(tx**2 + ty**2) * 10.0 
                    
                    # 计算角度误差 E_att (deg)
                    rmat, _ = cv2.Rodrigues(rvec)
                    yaw = math.atan2(rmat[1, 0], rmat[0, 0])
                    current_e_att = abs(math.degrees(yaw))
                    
                    self.sum_pos_error += current_e_pos
                    self.sum_att_error += current_e_att
                    self.valid_samples += 1

            # 检查是否刚达到采样终点，若是则锁定最终值
            if self.n_total >= self.max_frames:
                self.eval_finished = True
                self.final_avg_pos = self.sum_pos_error / self.valid_samples if self.valid_samples > 0 else 0.0
                self.final_avg_att = self.sum_att_error / self.valid_samples if self.valid_samples > 0 else 0.0
                self.final_valid_count = self.valid_samples
                self.get_logger().info(f"✅ 采样完成！指标已锁定。Avg E_pos: {self.final_avg_pos:.2f}cm")

        # --- 实时渲染逻辑 (即便采样完成也继续运行) ---
        if tags:
            for tag in tags:
                corners = tag.corners.astype(np.int32)
                cv2.polylines(img, [corners], True, (0, 255, 0), 2)
                cv2.circle(img, (int(tag.center[0]), int(tag.center[1])), 4, (0, 0, 255), -1)

        # 叠加 UI 数据面板
        self.current_img = self.render_overlay(img)

    def render_overlay(self, img):
        """绘制统计数据面板"""
        canvas = img.copy()
        
        # 决定显示实时值还是锁定值
        if self.eval_finished:
            disp_pos = self.final_avg_pos
            disp_att = self.final_avg_att
            disp_valid = self.final_valid_count
            status_text = "STATUS: EVALUATION FINISHED (LOCKED)"
            status_color = (0, 255, 255)
        else:
            disp_pos = self.sum_pos_error / self.valid_samples if self.valid_samples > 0 else 0.0
            disp_att = self.sum_att_error / self.valid_samples if self.valid_samples > 0 else 0.0
            disp_valid = self.valid_samples
            status_text = f"Sampling: {self.n_total}/{self.max_frames}"
            status_color = (255, 255, 255)

        # UI 半透明背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, status_text, (20, 35), f, 0.6, status_color, 1 if not self.eval_finished else 2)
        
        # 位置与角度指标展示
        c_pos = (0, 255, 0) if disp_pos <= 5.0 else (0, 0, 255)
        cv2.putText(canvas, f"Avg E_pos: {disp_pos:.2f} cm (Target <= 5)", (20, 70), f, 0.7, c_pos, 2)
        
        c_att = (0, 255, 0) if disp_att <= 3.0 else (0, 0, 255)
        cv2.putText(canvas, f"Avg E_att: {disp_att:.2f} deg (Target <= 3)", (20, 105), f, 0.7, c_att, 2)
        
        cv2.putText(canvas, f"Valid Count: {disp_valid} / {self.n_total if not self.eval_finished else 1000}", 
                    (20, 135), f, 0.5, (200, 200, 200), 1)
        return canvas

    def update_display(self):
        if self.current_img is not None:
            cv2.imshow("Real-time Precision Evaluator", self.current_img)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = PrecisionEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()