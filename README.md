Plant Severity Detection

Plant Severity Detection is an application that uses deep learning models for object detection and segmentation to analyze images of plants and calculate the severity of certain conditions based on color ratios in plant images.
Features

    Object Detection: Uses YOLOv8x to detect wheat heads in plant images.
    Segmentation: Applies MobileSAM (Vision Transformer-based) for segmentation of detected wheat heads.
    Severity Calculation: Analyzes color ratios in segmented areas to determine severity levels of plant conditions.

Installation
Prerequisites

    Python 3.7 or higher
    Node.js (for frontend development)
    React Native (for mobile frontend development)

Backend Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/PlantSeverityDetection.git
cd PlantSeverityDetection/backend

Install dependencies:

bash

    pip install -r requirements.txt

    Setup YOLO and MobileSAM models:
        Ensure YOLOv8x and MobileSAM models are correctly downloaded and placed in the backend folder.

Frontend Installation (React Native)

    Navigate to the frontend directory:

    bash

cd ../frontend

Install dependencies:

bash

    npm install

    Ensure React Native CLI is configured and environment is set up for Android/iOS development.

Configuration

    Configure paths to YOLO and MobileSAM models in the backend.
    Adjust any necessary environment variables or configuration files as per your setup.

Usage
Running the Backend

    Start the backend server:

    bash

    cd ../backend
    python app.py

    The backend server will run at http://localhost:5001.

Running the Frontend (React Native)

    Start the Metro server:

    bash

cd ../frontend
npx react-native start

Run the application on Android/iOS emulator or device:

bash

npx react-native run-android

or

bash

    npx react-native run-ios

Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.
License

This project is licensed under the MIT License - see the LICENSE.md file for details.
Acknowledgments

    Ultralytics for YOLOv8x
    MobileSAM contributors
    React Native community
