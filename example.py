import json
import time

import open3d as o3d
import epiceye
import sys
import cv2

logger = epiceye.logger
divider = lambda x: "-" * 20 + f"[{x:^20}]" + "-" * 20


def epiceye_example(ip: str):
    logger.info(f"epiceye sdk version: {epiceye.get_sdk_version()}")
    if ip is None:
        logger.info(divider("Searching Camera"))
        found_camera = epiceye.search_camera()
        if found_camera is not None:
            ip = found_camera[0]["ip"]
            logger.info("using ip: ", ip)
        else:
            logger.info("No camera found!")
            return

    logger.info(divider("get info"))
    info = epiceye.get_info(ip)
    if info is not None:
        width = info['width']
        height = info['height']
        logger.info(f"SN        : {info['sn']}")
        logger.info(f"ip        : {info['ip']}")
        logger.info(f"model     : {info['model']}")
        logger.info(f"alias     : {info['alias']}")
        logger.info(f"resolution: {width} * {height}")
    else:
        logger.info("info is None")

    logger.info(divider("get config"))
    config = epiceye.get_config(ip)
    if config is not None:
        config_str = json.dumps(config, indent=1)
        for line in config_str.split("\n"):
            logger.info(line)
    else:
        logger.info("config is None")

    logger.info(divider("set config"))
    config_returned = epiceye.set_config(ip=ip, config=config)
    if config_returned is not None:
        config_str = json.dumps(config_returned, indent=1)
        for line in config_str.split("\n"):
            logger.info(line)
    else:
        logger.info("Failed to set config")

    logger.info(divider("get intrinsic"))
    camera_matrix = epiceye.get_camera_matrix(ip)
    distortion = epiceye.get_distortion(ip)
    logger.info(f"camera matrix: {camera_matrix}")
    logger.info(f"distortion   : {distortion}")

    logger.info(divider("trigger frame"))

    for i in range(10):
        frame_id = epiceye.trigger_frame(ip=ip, pointcloud=True)
        logger.info(f"frame_id: {frame_id}")

        logger.info(divider("get image"))
        image = epiceye.get_image(ip=ip, frame_id=frame_id)
        if image is not None:
            # convert 10bit raw image to 8bit for display
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
            cv2.imwrite("image.png", image)
            logger.info("Image saved to image.png")
        else:
            logger.info("image is None")

        logger.info(divider("get pointcloud"))
        pointcloud = epiceye.get_point_cloud(ip=ip, frame_id=frame_id)
        if pointcloud is not None:
            pcd_data = o3d.geometry.PointCloud()
            points = pointcloud.reshape(-1, 3)
            pcd_data.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(f"cloud_{i}.ply", pcd_data)
            logger.info(f"Pointcloud saved to cloud_{i}.ply")
        else:
            logger.info("pointcloud is None")

        logger.info(divider("get depth"))
        depth = epiceye.get_depth(ip=ip, frame_id=frame_id)
        if depth is not None:
            pseudo_color = cv2.applyColorMap(
                (depth / 10.0).astype("uint8"), cv2.COLORMAP_JET)
            cv2.imwrite("depth.png", pseudo_color)
            logger.info("Pseudo color map of depth image saved to depth.png")
        else:
            logger.info("depth is None")
        time.sleep(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_ip = sys.argv[1]
    else:
        input_ip = None
    epiceye_example(input_ip)
