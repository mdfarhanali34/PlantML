from quart import Quart, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

app = Quart(__name__)

# Load YOLO model
model = YOLO("/home/mohammadfarhanali/Downloads/plantML/backend/YOLOv8x_wheat_head_detection.onnx")
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load MobileSAM 
model_type = "vit_t"
sam_checkpoint = "/home/mohammadfarhanali/Downloads/plantML/backend/mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

async def calculate_severity(image_path):
    # Perform YOLO inference
    results = model(image_path)
    
    all_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes.xyxy:
            all_boxes.append(box.tolist())
    
    img_arr = cv2.imread(image_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(img_arr)
    # for box in all_boxes:
    #     show_box(box, plt.gca())
    # plt.show()
    # Set the image for the predictor
    # predictor.set_image(img_arr)

    # Set the image for the predictor
    predictor.set_image(img_arr)

    # Initialize a list to store the percentages
    percentages = []

    # Loop through each box and calculate the percentage of pixels with R/G ratio above a threshold for each detection
    for box in all_boxes:
        input_box = np.array(box)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask = masks[0]

        # Debug print to check mask values
        print(f"Mask values: {mask}")

        # Create a background with NA values
        na_background = np.full_like(img_arr, np.nan)

        # Create a function to divide the red band by the green band to create a new image
        img_na = na_background * (1 - mask[..., np.newaxis]) + img_arr * mask[..., np.newaxis]
        s_r = img_na[:, :, 0]
        s_g = img_na[:, :, 1]

        # Avoid division by zero
        s_rg_sum = s_g + s_r
        s_rg_sum[s_rg_sum == 0] = np.nan

        # Performing the red/green ratio calculation
        rg = s_r / s_rg_sum

        # Mask out the pixels that are not part of the detection
        rg[~mask] = np.nan

        # Debug print to check RG ratio values
        print(f"RG ratio values: {rg}")

        # Calculate the number of pixels above a threshold for each detection
        threshold = 0.495  # Adjust threshold based on the histogram above
        num_pixels = np.sum(rg > threshold)
        total_pixels = np.sum(mask)

        # Ensure we don't divide by zero
        if total_pixels > 0:
            percent_pixels = num_pixels / total_pixels * 100
        else:
            percent_pixels = 0
        
        percentages.append(percent_pixels)

    # Calculate the average percentage of pixels with R/G ratio above the threshold for all detections
    average_percentage = sum(percentages) / len(percentages) if percentages else 0
    
    return average_percentage

@app.route('/calculate_severity', methods=['POST'])
async def calculate_severity_endpoint():
    files = await request.files  # Await request.files
    if 'image' not in files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = files['image']
    image_path = "temp_image.jpg"
    await image_file.save(image_path)
    
    try:
        average_percentage = await calculate_severity(image_path)
        return jsonify({"average_percentage": average_percentage})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
