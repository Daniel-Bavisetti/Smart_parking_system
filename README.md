
# Smart Parking System

The **Smart Parking System** is a computer vision-based application that automates vehicle and license plate detection using YOLOv8 and OCR. It identifies individual vehicles, tracks them with unique IDs, extracts license plate information, and prepares data for automated billing based on parking duration.

---

## 📁 Project Structure

```
.
├── main.py                     # Runs YOLOv8 to detect vehicles and plates, assigns IDs
├── util.py                     # Helper functions for OCR and license plate processing
├── license_plate_detector.pt   # Custom model for detecting license plates
├── yolov8n.pt                  # YOLOv8n model for vehicle detection
├── video.mp4                   # Sample input video
├── test_interpolated.csv       # CSV output with detection and license data
├── visualise.py                # Visualization script for annotations
└── out.mp4                     # Visualized output (not used in deployment)
```

---

## ⚙️ How It Works

### `main.py`
- Detects vehicles using `yolov8n.pt`.
- Assigns a unique `car_id` to each vehicle.
- Stores bounding boxes for each detected vehicle.
- Feeds license plate regions to an OCR model (`license_plate_detector.pt`).
- Generates a CSV with the following per-frame information:
  - Frame number
  - Vehicle ID
  - Bounding boxes (car and license plate)
  - Detected license number and its confidence score

### `util.py`
- Contains helper functions for OCR and formatting:
  - **`read_license_plate(frame, bbox)`**: Extracts the license plate from a region and returns the text.
  - **`license_complies_format(plate_text)`**: Checks if the license follows valid formats.
  - **`format_license(plate_text)`**: Corrects common OCR errors and formats text to match Indian license plate standards.

> Indian license plate format supported:
>
> - `XX88 XX8888`
> - `XX88 X8888`
> - `XX88 8888`

---

## 🎯 Features

- **Vehicle detection** using YOLOv8.
- **License plate detection and recognition**.
- Assigns a **unique ID** to every vehicle.
- Filters and **validates license plate formats**.
- Generates a **CSV file** with all detection data for billing logic.
- Optional **visualization script (`visualise.py`)** for debugging/demo purposes.

---

## 🚀 Deployment Plan

- **Current Setup**: Runs on a laptop using a recorded `.mp4` video file.
- **Future Plan**: Optimize to run on an IoT device with live video input (e.g., Jetson Nano, Raspberry Pi with Coral/TPU).
- The final system will:
  - Process real-time streams
  - Use CSV logs to compute **parking duration**
  - Automatically **bill users** based on the time spent

---

## 🖥️ Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code> (example)</summary>

```
opencv-python
pandas
numpy
ultralytics
```

</details>

### 2. Run Detection

```bash
python main.py
```

- Generates `test_interpolated.csv` with detection and license data.

### 3. Optional Visualization

```bash
python visualise.py
```

- Creates `out.mp4` with annotations showing vehicle boxes and license plates.

---

## 📄 CSV Output Structure

| Column                  | Description                                 |
|-------------------------|---------------------------------------------|
| `frame_nmr`             | Frame number in the video                   |
| `car_id`                | Unique ID assigned to each vehicle          |
| `car_bbox`              | Bounding box of the vehicle `[x1, y1, x2, y2]` |
| `license_plate_bbox`    | Bounding box of the license plate           |
| `license_number`        | Text detected on the plate                  |
| `license_number_score`  | OCR confidence score                        |

---

## 📦 Output

- `test_interpolated.csv`: Main output used for billing logic.
- `out.mp4`: Visual output for testing (not needed during deployment).

---

## 💡 Example Use Cases

- Automated billing at mall or office parking.
- Real-time vehicle monitoring and registration.
- Integration with entry/exit boom barriers.

---

## 📌 Notes

- The system is trained and tested on **Indian license plate formats**.
- The detection is robust against common OCR misreads via formatting rules.
