#!/usr/bin/env python3
"""
EVSR (EDSR + TensorRT) 基础演示
用法： python evsr_demo.py [--image 图片路径]  # 不传参数则使用摄像头
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os

class EDSR_TensorRT:
    """轻量版 EDSR TensorRT 推理器"""
    def __init__(self, engine_path, input_size=(480, 640)):
        self.input_size = input_size  # H, W
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 获取绑定信息
        self.input_idx = self.engine.get_binding_index('input')
        self.output_idx = self.engine.get_binding_index('output')
        self.input_shape = (1, 3, *self.input_size)
        self.output_shape = (1, 3, self.input_size[0]*2, self.input_size[1]*2)

        # 分配显存
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * np.float32().itemsize))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * np.float32().itemsize))
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

        # pinned memory 加速传输
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)

    def upscale(self, img_bgr):
        """输入 BGR 图像 (uint8)，返回 2倍超分 BGR 图像"""
        h, w = self.input_size
        img = cv2.resize(img_bgr, (w, h))
        # 预处理：BGR→RGB，HWC→CHW，归一化
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        np.copyto(self.h_input[0], img_chw)

        # 异步推理
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        # 后处理
        out_chw = self.h_output[0] * 255.0
        out_chw = np.clip(out_chw, 0, 255).astype(np.uint8)
        out_rgb = np.transpose(out_chw, (1, 2, 0))
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

def main():
    parser = argparse.ArgumentParser(description='EVSR Demo')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--engine', type=str, default='edsr_x2_fp16.engine', help='TensorRT engine 路径')
    args = parser.parse_args()

    if not os.path.exists(args.engine):
        print(f"错误: 找不到引擎文件 {args.engine}，请先转换模型。")
        return

    # 初始化超分引擎（默认 640x480 输入）
    sr = EDSR_TensorRT(args.engine, input_size=(480, 640))

    if args.image:
        # 图片模式
        img = cv2.imread(args.image)
        if img is None:
            print("无法读取图片")
            return
        t0 = cv2.getTickCount()
        hr = sr.upscale(img)
        t1 = cv2.getTickCount()
        time_ms = (t1 - t0) / cv2.getTickFrequency() * 1000
        print(f"超分完成，耗时: {time_ms:.2f} ms")

        cv2.imshow('Original', img)
        cv2.imshow('EVSR (EDSR TensorRT)', hr)
        cv2.imwrite('evsr_result.jpg', hr)
        print("结果已保存为 evsr_result.jpg")
        cv2.waitKey(0)
    else:
        # 摄像头实时模式
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("实时模式：按 q 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = cv2.getTickCount()
            hr = sr.upscale(frame)
            t1 = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (t1 - t0)

            cv2.putText(hr, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow('Original', frame)
            cv2.imshow('EVSR (EDSR TensorRT)', hr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()