from quart import Quart, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import gaussian_kde

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

# define show box function
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=1))

# define mask and box functions for visualization
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image) 
    
async def calculate_severity(image_path):
    # Perform YOLO inference
    results = model(image_path)
    
    all_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes.xyxy:
            all_boxes.append(box.tolist())
    
    print(f"Detected boxes: {all_boxes}")
    
    img_arr = cv2.imread(image_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(img_arr)
    for box in all_boxes:
        show_box(box, plt.gca())
    # plt.show()

    # Set the image for the predictor
    predictor.set_image(img_arr)

    all_masks = []
    for box in all_boxes:
        input_box = np.array(box)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        all_masks.append(masks)

    # Initialize a combined mask with zeros
    combined_mask = np.zeros_like(all_masks[0][0], dtype=np.uint8)

    # Combine all masks
    for mask in all_masks:
        combined_mask = np.logical_or(combined_mask, mask[0])
    
    # Debug print to check combined mask values
    print(f"Combined mask values: {combined_mask}")

    # Convert the segmentation mask to a binary mask
    binary_mask = np.where(combined_mask > 0.5, 1, 0)
    
    # Debug print to check binary mask values
    print(f"Binary mask values: {binary_mask}")

    # Create a background with NA values
    na_background = np.full_like(img_arr, np.nan)
    
    # Debug print to check na_background values
    print(f"NA background values: {na_background}")

    # Create a function to divide the red band by the green band to create a new image
    img_na = na_background * (1 - binary_mask[..., np.newaxis]) + img_arr * binary_mask[..., np.newaxis]
    s_r = img_na[:,:,0]
    s_g = img_na[:,:,1]
    s_b = img_na[:,:,2]

    # Debug print to check image values
    print(f"Red channel values: {s_r}")
    print(f"Green channel values: {s_g}")
    print(f"Blue channel values: {s_b}")

    # Perform the red/green ratio calculation
    rg = s_r / (s_g + s_r)
    
    # Debug print to check RG ratio values
    print(f"RG ratio values: {rg}")

    # Set the red/green ratio values outside the mask to NaN
    rg[~combined_mask] = np.nan
    
    # Debug print to check RG ratio values after masking
    print(f"RG ratio values after masking: {rg}")

    # Define the color limits
    rg_min = 0.2
    rg_max = 0.6

    # Set the plot style to a dark theme
    plt.style.use('classic')

    # Create a new figure and arrange the subplots
    fig = plt.figure(figsize=(15, 15))

    # Define the grid layout
    grid = (2, 2)

    # Plot the original image on the top left
    ax1 = plt.subplot2grid(grid, (0, 0))
    img1 = ax1.imshow(img_arr)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot the image with the segmentation mask on the top right
    ax2 = plt.subplot2grid(grid, (0, 1))
    ax2.imshow(img_arr)
    for mask in all_masks:
        show_mask(mask[0], ax2, random_color=True)
    ax2.set_title('Segmentation Masks')
    ax2.axis('off')

    # Plot the rg image on the bottom left
    ax3 = plt.subplot2grid(grid, (1, 0))
    img3 = ax3.imshow(rg, cmap='Spectral', vmin=rg_min, vmax=rg_max)
    ax3.set_title('RG Transformation')

    # Display a colorbar for the rg image
    cbar = plt.colorbar(img3, ax=ax3)

    # Filter out invalid values (inf and NaN) from the rg array
    valid_rg = rg.ravel()[np.isfinite(rg.ravel())]
    
    # Debug print to check valid RG values
    print(f"Valid RG values: {valid_rg}")

    # Plot the smoothed density histogram on the bottom right (ax4)
    ax4 = plt.subplot2grid(grid, (1, 1))
    ax4.set_title('Smoothed Density Histogram')

    kde = gaussian_kde(valid_rg)

    # Set the range of values for the x-axis
    x_vals = np.linspace(rg_min, rg_max, 500)

    # Obtain colors from the 'Spectral' colormap for the histogram bars
    colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(x_vals)))

    # Plot the histogram as bars with colors based on x-value
    ax4.bar(x_vals, kde(x_vals), width=(x_vals[1]-x_vals[0]), color=colors, edgecolor='none')

    # Plot a line graph on top of the histogram
    ax4.plot(x_vals, kde(x_vals), color='black', lw=2)

    # add a dashed line at a value of 0.5
    ax4.axvline(x=0.5, color='black', linestyle='--')

    # Add labels and title
    ax4.set_xlabel('RG Values')
    ax4.set_ylabel('Density')
    ax4.set_title('Smoothed Density Histogram')

    # Set the x-axis limits to the specified range
    ax4.set_xlim(rg_min, rg_max)

    # Adjust spacing between subplots
    plt.tight_layout()

    # interact with the histogram to determine the ideal threshold
    mplcursors.cursor(hover=True)

    # Show the plot
    plt.show()
    
    # Calculate the percentage of pixels above a threshold
    threshold = 0.495  # Adjust threshold based on the histogram above
    num_pixels = np.sum(rg > threshold)
    total_pixels = np.sum(binary_mask)

    # Ensure we don't divide by zero
    if total_pixels > 0:
        percent_pixels = num_pixels / total_pixels * 100
    else:
        percent_pixels = 0
    
    print(f"Number of pixels above threshold: {num_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Percentage of pixels above threshold: {percent_pixels}")

    return percent_pixels

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
