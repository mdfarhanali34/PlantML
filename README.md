🌿

```
# Plant Severity Detection

Plant Severity Detection is an application that uses deep learning models for object detection and segmentation to analyze images of plants and calculate the severity of certain conditions based on color ratios in plant images.

## Features

- **Object Detection**: Uses YOLOv8x to detect wheat heads in plant images.
- **Segmentation**: Applies MobileSAM (Vision Transformer-based) for segmentation of detected wheat heads.
- **Severity Calculation**: Analyzes color ratios in segmented areas to determine severity levels of plant conditions.

## Installation

### Prerequisites

- Python 3.7 or higher
- Node.js (for frontend development)
- React Native (for mobile frontend development)

### Backend Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/yourusername/PlantSeverityDetection.git
   cd PlantSeverityDetection/backend
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

3. **Setup YOLO and MobileSAM models:**
   
   - Ensure YOLOv8x and MobileSAM models are correctly downloaded and placed in the backend folder.

### Frontend Installation (React Native)

1. **Navigate to the frontend directory:**

   ```
   cd ../frontend
   ```

2. **Install dependencies:**

   ```
   npm install
   ```

3. **Ensure React Native CLI is configured and environment is set up for Android/iOS development.**

## Configuration

- **Configure paths to YOLO and MobileSAM models in the backend.**
- **Adjust any necessary environment variables or configuration files as per your setup.**

## Usage

### Running the Backend

1. **Start the backend server:**

   ```
   cd ../backend
   python app.py
   ```

   The backend server will run at `http://localhost:5001`.

### Running the Frontend (React Native)

1. **Start the Metro server:**

   ```
   cd ../frontend
   npx react-native start
   ```

2. **Run the application on Android/iOS emulator or device:**

   ```
   npx react-native run-android
   ```

```
