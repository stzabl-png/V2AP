"""
Standalone script to visualize head camera feed.
Press 'q' to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.head_camera import create_robot_with_head_camera, wait_for_head_camera  # noqa: E402

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360


def main() -> None:
    logger.info("Initializing robot with head camera...")

    robot = create_robot_with_head_camera()
    try:
        wait_for_head_camera(robot, timeout=30.0, required_streams=("left_rgb",))
        logger.info("Press 'q' to quit")

        while True:
            head_obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])
            head_image = head_obs.get("left_rgb")

            if head_image is not None:
                if head_image.shape[0] != IMAGE_HEIGHT or head_image.shape[1] != IMAGE_WIDTH:
                    head_image = cv2.resize(head_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                head_image_bgr = cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Head Camera", head_image_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        robot.shutdown()
        logger.info("Done")


if __name__ == "__main__":
    main()
