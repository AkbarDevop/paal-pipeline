"""Official DepthAI ToF/IR -> RGB alignment demo."""

import cv2
import depthai as dai
import numpy as np


def to_vis_depth(frame):
    depth = frame.astype(np.float32)
    if np.max(depth) > 0:
        depth = depth / np.max(depth)
    depth = (depth * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_JET)


def to_vis_ir(frame):
    ir = frame.astype(np.float32)
    if np.max(ir) > 0:
        ir = ir / np.max(ir)
    ir = (ir * 255.0).astype(np.uint8)
    return cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)


def main():
    try:
        if not dai.Device.getAllAvailableDevices():
            print("No available OAK/DepthAI device found.")
            print("Check USB cable/power and run again.")
            return
    except Exception as e:
        print(f"Failed to query devices: {e}")
        return

    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setSize(1280, 800)
    cam_rgb.setFps(15)

    tof = pipeline.create(dai.node.ToF).build(dai.CameraBoardSocket.CAM_B)
    tof.setFps(15)

    align_depth = pipeline.create(dai.node.ImageAlign)
    align_depth.inputAlignTo.setBlocking(False)
    tof.depth.link(align_depth.input)
    cam_rgb.video.link(align_depth.inputAlignTo)

    align_ir = pipeline.create(dai.node.ImageAlign)
    align_ir.inputAlignTo.setBlocking(False)
    tof.irRaw.link(align_ir.input)
    cam_rgb.video.link(align_ir.inputAlignTo)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth_aligned")
    align_depth.outputAligned.link(xout_depth.input)

    xout_ir = pipeline.create(dai.node.XLinkOut)
    xout_ir.setStreamName("ir_aligned")
    align_ir.outputAligned.link(xout_ir.input)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth_aligned", maxSize=4, blocking=False)
        q_ir = device.getOutputQueue("ir_aligned", maxSize=4, blocking=False)

        print("Press q to quit.")
        while True:
            rgb = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()
            ir = q_ir.get().getFrame()

            depth_vis = to_vis_depth(depth)
            ir_vis = to_vis_ir(ir)

            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth aligned to RGB", depth_vis)
            cv2.imshow("IR aligned to RGB", ir_vis)
            cv2.imshow("Overlay RGB+Depth", cv2.addWeighted(rgb, 0.65, depth_vis, 0.35, 0.0))
            cv2.imshow("Overlay RGB+IR", cv2.addWeighted(rgb, 0.65, ir_vis, 0.35, 0.0))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
