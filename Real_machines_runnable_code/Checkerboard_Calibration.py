import cv2
import numpy as np
import os
import glob

# ================= 针对你的棋盘格修改配置 =================
# 1. 内部角点规格: (25-1, 17-1)
CHECKERBOARD = (24, 16)

# 2. 物理尺寸: 1cm = 0.01 米 (用于获取物理位姿，对计算焦距fx, fy必不可少)
SQUARE_SIZE = 0.01 

# 3. GStreamer 管道 (Jetson Orin Nano CSI 相机)
def get_gst_pipeline(sensor_id=0):
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} ! '
        'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! '
        'nvvidconv ! video/x-raw, width=640, height=480, format=BGRx ! '
        'videoconvert ! video/x-raw, format=BGR ! appsink'
    )

# ================= 自动化标定核心逻辑 =================
objpoints = [] 
imgpoints = [] 

# 定义世界坐标系中的 3D 点 (0,0,0), (0.01,0,0), (0.02,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

def main():
    # 尝试打开相机 (下视通常是 0)
    cap = cv2.VideoCapture(get_gst_pipeline(0), cv2.CAP_GSTREAMER)
    
    print(f"检测到棋盘格规格: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} 角点")
    print("操作: [S] 采样, [C] 计算并保存, [Q] 退出")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        cv2.putText(display_frame, f"Saved: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 提高检测精度：对于 24x16 的密集型棋盘格，检测可能稍慢
            ret_corners, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            if ret_corners:
                # 亚像素精细化：窗口大小设为 (5,5) 适配 1cm 的小格子
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints.append(corners2)
                count += 1
                
                # 实时预览识别到的角点
                cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret_corners)
                cv2.imshow('Last Detection', frame)
                print(f"已捕获第 {count} 张有效样本")
            else:
                print("未发现完整角点，请确保 24x16 个内角点全部在画面内")

        elif key == ord('c') and count >= 15:
            print("正在计算内参...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            if ret:
                fx, fy = mtx[0,0], mtx[1,1]
                print("\n" + "="*40)
                print(f"标定完成！平均重投影误差: {ret:.4f}")
                print(f"相机内参 (focal_length_px): {(fx+fy)/2:.2f}")
                print(f"建议填入代码的值: {int((fx+fy)/2)}")
                print("="*40)
                # 保存为本地文件，方便以后直接加载
                np.savez("camera_calib_data.npz", mtx=mtx, dist=dist)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()