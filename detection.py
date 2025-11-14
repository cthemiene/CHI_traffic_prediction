"""
Parking lot video processing and occupancy detection.

This module defines classes and functions to ingest frames from a webcam or
pre‑recorded video, detect vehicles using a deep learning model and determine
whether each parking space is free or occupied.  It relies on the Ultralytics
YOLOv8 model for object detection when available.  If the ``ultralytics``
package is not installed, the module will raise a ``RuntimeError`` when
instantiating the detector.  The intent is to keep the detection logic
isolated so that it can be unit tested separately from the video capture
pipeline.

Notes
-----
* The YOLOv8 architecture treats object detection as a single regression
  problem, predicting bounding boxes and class probabilities in one forward
  pass.  Compared to earlier YOLO versions it uses an anchor‑free design and
  improved backbone and neck modules to boost performance【465760557246635†L122-L139】.
* The PKLot dataset and subsequent research show that YOLO models excel at
  parking occupancy detection when combined with traditional image processing
  to map detections back to individual parking slots【650450892526941†L130-L168】.
* You must calibrate your parking lot by recording the pixel coordinates of
  each space under typical conditions.  Those coordinates are passed into
  ``ParkingSpot`` below.

The two core classes are ``ParkingSpot`` and ``ParkingLotMonitor``.  Each
``ParkingSpot`` represents a rectangular region in the image corresponding
to a single parking space.  The monitor loads a YOLO model and uses it
to detect vehicles, then checks the overlap between detected vehicles and
parking spots using Intersection over Union (IoU).

"""

from __future__ import annotations

import cv2  # type: ignore  # OpenCV is assumed to be installed by default
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    # The ultralytics package contains the official implementation of YOLOv8.
    from ultralytics import YOLO  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    YOLO = None  # type: ignore


@dataclass
class ParkingSpot:
    """Data structure representing a single parking space on a frame.

    Parameters
    ----------
    id : str
        Unique identifier for the parking space (e.g., "A1").
    top_left : Tuple[int, int]
        (x, y) coordinates of the top‑left corner of the rectangular region.
    bottom_right : Tuple[int, int]
        (x, y) coordinates of the bottom‑right corner of the region.

    The coordinates must be specified in pixels relative to the video frame.
    """

    id: str
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

    def iou(self, bbox: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between this parking spot and a bounding box.

        Parameters
        ----------
        bbox : Tuple[int, int, int, int]
            Bounding box represented as (x1, y1, x2, y2) in pixel coordinates.

        Returns
        -------
        float
            The IoU value in the range [0, 1].  A higher value indicates a
            greater overlap between the detected object and the parking space.
        """
        # Extract coordinates for clarity
        px1, py1 = self.top_left
        px2, py2 = self.bottom_right

        x1, y1, x2, y2 = bbox

        # Compute overlap region
        inter_x1 = max(px1, x1)
        inter_y1 = max(py1, y1)
        inter_x2 = min(px2, x2)
        inter_y2 = min(py2, y2)

        # If there is no overlap, IoU is zero
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        # Compute areas
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        spot_area = (px2 - px1) * (py2 - py1)
        bbox_area = (x2 - x1) * (y2 - y1)

        union_area = spot_area + bbox_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


class ParkingLotMonitor:
    """Process frames and detect occupancy across multiple parking spaces.

    The monitor encapsulates the YOLO model and exposes methods to process
    individual frames or live streams.  It does not perform any network
    operations; instead it runs locally on the host machine or server.
    """

    def __init__(
        self,
        model_weights: str,
        parking_spots: List[ParkingSpot],
        device: str = "cpu",
        iou_threshold: float = 0.3,
    ) -> None:
        """Initialize the parking lot monitor.

        Parameters
        ----------
        model_weights : str
            Path to the YOLO model weights (e.g., "yolov8n.pt").  See the
            ultralytics documentation for available pretrained weights.  If
            ``ultralytics`` is not installed, a ``RuntimeError`` is raised.
        parking_spots : list of ParkingSpot
            List of parking spaces to monitor.  Each must specify the pixel
            boundaries of the space.
        device : str, optional
            Device on which to run the model ("cpu" or "cuda").  Defaults to
            "cpu".  GPU acceleration will dramatically improve inference speed
            but requires a compatible CUDA setup.
        iou_threshold : float, optional
            Minimum Intersection over Union between a detected vehicle and a
            parking space to consider the space occupied.  Defaults to 0.3.
        """
        if YOLO is None:
            raise RuntimeError(
                "ultralytics package is not available. Install it with `pip install ultralytics`."
            )
        self.model = YOLO(model_weights)
        self.parking_spots = parking_spots
        self.device = device
        self.iou_threshold = iou_threshold

    def _detect_cars(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Run object detection on a frame and return car bounding boxes.

        Only vehicles of interest (e.g., cars and trucks) are retained.  The
        ultralytics model returns results with class IDs.  Class ID 2
        corresponds to 'car' in the COCO dataset; 7 corresponds to 'truck'.

        Parameters
        ----------
        frame : ndarray
            BGR image loaded via OpenCV.

        Returns
        -------
        list of tuples
            Each tuple is a bounding box (x1, y1, x2, y2) for a detected
            vehicle.
        """
        # Run inference; results list has one element per image in batch
        results = self.model.predict(frame, device=self.device, verbose=False)
        bboxes: List[Tuple[int, int, int, int]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                # Keep cars and trucks (COCO class IDs 2 and 7)
                if int(cls) in (2, 5, 7):  # 5 corresponds to bus; optional
                    x1, y1, x2, y2 = box.astype(int)
                    bboxes.append((x1, y1, x2, y2))
        return bboxes

    def analyze_frame(self, frame: np.ndarray) -> List[bool]:
        """Determine the occupancy state for each parking space in a frame.

        Parameters
        ----------
        frame : ndarray
            Image in BGR format.

        Returns
        -------
        list of bool
            A list of booleans indicating whether each parking spot is
            occupied (True) or free (False).  The order corresponds to
            ``self.parking_spots``.
        """
        car_boxes = self._detect_cars(frame)
        occupancy: List[bool] = []
        for spot in self.parking_spots:
            occupied = False
            for box in car_boxes:
                iou_val = spot.iou(box)
                if iou_val >= self.iou_threshold:
                    occupied = True
                    break
            occupancy.append(occupied)
        return occupancy

    def draw_overlay(
        self,
        frame: np.ndarray,
        occupancy: List[bool],
        color_free: Tuple[int, int, int] = (0, 255, 0),
        color_occupied: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """Draw colored rectangles on the frame to visualize occupancy.

        Parameters
        ----------
        frame : ndarray
            Frame to draw on (will not be modified in place).
        occupancy : list of bool
            Occupancy state for each parking spot.
        color_free : tuple, optional
            BGR color for free spaces (default green).
        color_occupied : tuple, optional
            BGR color for occupied spaces (default red).

        Returns
        -------
        ndarray
            New frame with rectangles drawn.
        """
        output = frame.copy()
        for spot, occ in zip(self.parking_spots, occupancy):
            color = color_occupied if occ else color_free
            cv2.rectangle(output, spot.top_left, spot.bottom_right, color, 2)
            cv2.putText(
                output,
                f"{spot.id}: {'Occupied' if occ else 'Free'}",
                (spot.top_left[0], spot.top_left[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return output

    def process_stream(
        self,
        video_source: int | str,
        callback: Optional[callable] = None,
        sample_every_n_frames: int = 30,
    ) -> None:
        """Process a live video stream and optionally invoke a callback.

        This method opens the video source and iterates through frames.  Every
        ``sample_every_n_frames`` frames it runs detection and occupancy
        analysis.  A callback can be provided to consume occupancy data (e.g.
        writing to a database or file).  The callback receives three
        arguments: timestamp, occupancy list, and annotated frame.

        Parameters
        ----------
        video_source : int or str
            Device index for webcams or path to a video file.  Pass 0 for the
            default webcam.
        callback : callable, optional
            Function called with (timestamp, occupancy, annotated_frame).
        sample_every_n_frames : int, optional
            Perform detection every N frames to reduce computation.  Increase
            this value if the processing is too slow.  Defaults to 30 (≈1
            second at 30 fps).
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {video_source}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every_n_frames == 0:
                    occupancy = self.analyze_frame(frame)
                    annotated = self.draw_overlay(frame, occupancy)
                    if callback is not None:
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds
                        callback(timestamp, occupancy, annotated)
                frame_idx += 1
        finally:
            cap.release()