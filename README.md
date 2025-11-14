# CHI_traffic_prediction
Using this project to learn more about using python for predictive analysis


Smart Parking Lot Monitoring and Forecasting
Overview

Urban parking shortages and long search times are common frustrations for drivers. Manual counting or expensive sensor networks do not scale well. Modern computer‑vision models can analyse video feeds to detect vehicles and infer which parking spaces are free or occupied. The data generated can feed time‑series models to forecast when parking demand is lowest. This report outlines a Python solution that ingests webcam video, determines space availability, and predicts the best time to visit a parking lot with operating hours between 7 AM and 7 PM (CST). It summarizes recent research on parking occupancy detection, compares forecasting techniques, and proposes an architecture suitable for web and mobile deployment.

Recent research on video‑based parking occupancy detection

Parking occupancy detection has benefited from advances in deep learning. Traditional algorithms used sliding windows or Haar cascades, but modern methods treat object detection as a single regression problem. The YOLO (You Only Look Once) family of models runs in real time by predicting bounding boxes and class probabilities in one forward pass. The latest version, YOLOv8, was released by Ultralytics in January 2023. It adopts an anchor‑free design and improved backbone and neck modules; the neck uses a spatial pyramid pooling block with a novel C2f module instead of a traditional feature pyramid network, reducing memory consumption and improving detection of objects of various sizes
arxiv.org
. YOLOv8’s backbone is based on a custom CSP‑Darknet53 architecture which uses cross‑stage partial connections to improve information flow while reducing computational complexity
arxiv.org
.

Comparative research has shown that customizing YOLOv8 with alternative backbones (ResNet‑18, EfficientNet V2, Ghost and VGG‑16) can further improve detection robustness in parking scenarios
arxiv.org
. However, even the nano variant (YOLOv8n) yields fast inference with high accuracy on parking datasets. A 2025 study on the PKLot dataset reported that YOLOv8’s anchor‑free design and three detection heads allow it to detect small objects (such as compact cars or motorcycles) more reliably than older YOLO versions
arxiv.org
. Occupancy detection systems typically combine YOLO with image analysis: researchers first delimit each parking space manually, then compute the intersection between detected vehicles and each space. A recent deep‑learning solution using YOLO and OpenCV monitored parking spaces at a bank’s car park and showed that this approach reduced drivers’ time spent searching for spaces and outperformed traditional sensor‑based methods
researchgate.net
.

Time‑series forecasting of parking demand

Once vehicle detections are aggregated into occupancy ratios (number of occupied spaces divided by total spaces) over time, forecasting models can predict future demand. A 2023 case study on Málaga and Birmingham tested four forecasting methods—Prophet, SARIMAX, LSTM and Neural Prophet—using six months of data. The study found that Prophet and Neural Prophet generally produced lower mean absolute errors than SARIMAX and LSTM, especially when forecasting per‑day occupancy
researchgate.net
. Prophet’s main advantage is its additive regression framework, which automatically models daily and weekly seasonality and provides confidence intervals; the study reported that Prophet achieved hit rates around 80 % for predicting whether the real occupancy fell within the forecast interval
researchgate.net
. While neural models such as LSTM excel on large datasets, simple statistical models can still be competitive when data are limited. The study concluded that Prophet and Neural Prophet were more accurate than SARIMAX and LSTM on the evaluated parking data
researchgate.net
.

Architectural design
Components

The proposed system consists of several components, illustrated in the conceptual diagram below:

Camera input: A fixed webcam captures the parking lot. Frames are sampled (e.g., once per second) to reduce computation. Higher sampling rates improve responsiveness but require more processing power.

Object detection: Each sampled frame is sent to the detection engine. The engine uses a pre‑trained YOLOv8 model to detect vehicles. Detected bounding boxes with classes corresponding to cars, buses or trucks are kept. The Intersection over Union (IoU) is calculated between each vehicle and manually defined parking spaces. If IoU exceeds a configurable threshold (e.g., 0.3), the space is considered occupied.

Database and logging: Occupancy observations are stored with timestamps. Each observation records the occupancy ratio (occupied spaces divided by total spaces). Data can be stored in a relational database (PostgreSQL, SQLite) or a time‑series database (InfluxDB).

Forecasting service: A time‑series forecasting module aggregates observations into regular intervals (e.g., 15 minutes). The module fits a forecasting model. When the prophet library is installed, Prophet (or Neural Prophet) is used because research suggests it yields competitive accuracy
researchgate.net
. If Prophet is unavailable, a SARIMAX model serves as a fallback. Forecasts are generated for the next 12 hours, and the time within the operating window (07:00–19:00) with the lowest predicted occupancy is recommended as the best time to visit.

API layer: A REST API (implemented with FastAPI) exposes endpoints for retrieving the current occupancy status, annotated images, forecasts and recommended visit times. This API can run locally or in the cloud.

User interface: A web/mobile application displays the current occupancy and best times. Cross‑platform frameworks such as Flutter allow development of a single codebase for Android, iOS, web and desktop. Flutter’s advantages include a single Dart codebase, hot reload for rapid iteration, a rich set of customizable widgets, near‑native performance, integrated testing tools and strong community support
redwerk.com
. These qualities make Flutter suitable for prototyping and deploying parking dashboards.

Data flow

Calibration: Define the coordinates of each parking space by annotating a sample image. Store these coordinates in a configuration file.

Frame capture: Acquire frames from the webcam using OpenCV. Every N frames, run detection to reduce load.

Detection: Use the YOLOv8 model to detect vehicles. Map detections to parking spaces via IoU. Produce a boolean occupancy vector.

Logging: Store timestamped occupancy ratios. Also keep optional annotated frames for auditing or debugging.

Forecasting: Periodically aggregate the logged data (e.g., every 15 minutes). Fit or update the forecasting model. Generate predictions for the next 12 hours and compute the best visiting time within the operating window.

API/UI: Serve the latest occupancy state and forecast via REST. The Flutter app can poll or subscribe to updates and visualize them.

Implementation details
Object detection module (detection.py)

The ParkingSpot dataclass represents each parking space. It stores the pixel coordinates of the top‑left and bottom‑right corners and exposes an iou method to compute the Intersection over Union with a bounding box. A ParkingLotMonitor class encapsulates the YOLO model and the list of parking spaces. During initialization, it loads the model weights using Ultralytics’ YOLO class. The _detect_cars method runs inference on a frame and returns bounding boxes for vehicles (cars, buses or trucks). The public method analyze_frame loops through all parking spaces, computes the IoU with each detected vehicle and returns a list of booleans indicating which spaces are occupied. The draw_overlay method annotates frames with colored rectangles and labels (green for free, red for occupied). A process_stream helper runs detection on a video source at a configurable sampling rate and invokes a callback with occupancy results and annotated frames.

Forecasting module (forecasting.py)

The OccupancyForecaster class loads a sequence of observations (timestamp and occupancy ratio) into a pandas DataFrame. Observations are resampled into regular intervals (default: 15 minutes) and missing values are interpolated. When the use_prophet flag is set and Prophet is available, the class fits a Prophet model with daily and weekly seasonality. Otherwise it falls back to a SARIMAX model with seasonal components. The forecast method generates predictions for a specified horizon (default: 12 hours). The best_time_to_visit method filters the forecast to the operating hours (07:00–19:00 CST) and returns the timestamp with the lowest predicted occupancy ratio. This function uses timezone‑aware timestamps via the zoneinfo module.

API (app.py)

For deployment, the package includes an optional FastAPI service. It exposes two endpoints:

GET /current captures a frame from the webcam, returns the current occupancy state and an annotated image encoded in base64. It also appends the observation to the forecaster’s log.

GET /forecast aggregates all observations, trains the forecaster if necessary, generates a 12‑hour forecast and returns the prediction series along with the recommended visit time.

Integrating the API with a database (e.g., PostgreSQL or InfluxDB) enables persistent storage and scalability. The Flutter client can call these endpoints to display live occupancy and future recommendations.

Coding best practices

Modularity: Separate detection, forecasting and API concerns into different modules (detection.py, forecasting.py, app.py). This improves testability and maintainability.

Configuration management: Store parking space coordinates, model paths and thresholds in configuration files or environment variables. Avoid hard‑coding values in code.

Error handling: The detection module raises a RuntimeError if the ultralytics package is unavailable, prompting the user to install it. The API layer checks for camera availability and returns appropriate HTTP error codes.

Timezone awareness: All timestamps use Python’s zoneinfo to avoid daylight‑saving errors. This is critical for the 07:00–19:00 CST operating window.

Unit tests: The package includes pytest tests. test_detection.py checks the Intersection over Union calculation and occupancy logic using a dummy monitor that bypasses YOLO. test_forecasting.py generates synthetic sinusoidal data, fits the forecaster and verifies that the forecast has the expected length and that the recommended time falls within operating hours.

Scalability: Run inference on a GPU where possible. YOLOv8’s design allows real‑time detection on modern GPUs
arxiv.org
. For edge devices, consider using a smaller model (YOLOv8n) or pruning/quantization. The API can be containerized (e.g., Docker) and scaled horizontally.

Privacy: Ensure camera placement does not capture personally identifiable information (e.g., vehicle license plates) or restrict detection to bounding boxes without storing raw images. Comply with local regulations.

Conclusion

The combination of YOLOv8 for real‑time vehicle detection and time‑series forecasting for demand prediction provides a practical solution to parking management. Research shows that YOLOv8’s anchor‑free architecture and multi‑scale detection heads offer state‑of‑the‑art accuracy for parking occupancy
arxiv.org
, while forecasting methods like Prophet can model daily and weekly patterns and produce reliable predictions
researchgate.net
. The Python package provided here demonstrates how to implement such a system, log data, forecast occupancy and expose results through a REST API. A Flutter front‑end can consume this API and provide users with an intuitive view of current availability and the best times to visit, taking advantage of Flutter’s single codebase and high performance
redwerk.com
.
