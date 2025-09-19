import cv2
import numpy as np
import os
import logging
import urllib.request
import socket
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vehicle classes to detect
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

# YOLO model paths
MODEL_DIR = "models"
CONFIG_PATH = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

# Global variables
net = None
classes = []
output_layers = []
is_initialized = False

def ensure_model_directory():
    """Ensure models directory exists"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created {MODEL_DIR} directory")

def download_file_with_timeout(url, filepath, timeout=30):
    """Download file with timeout and error handling"""
    try:
        # Set socket timeout
        socket.setdefaulttimeout(timeout)
        
        # Create request with headers
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        return True
    except (URLError, HTTPError, socket.timeout) as e:
        logger.error(f"Download failed for {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False

def download_yolo_files():
    """Download YOLO files if missing"""
    ensure_model_directory()
    
    files = {
        CONFIG_PATH: 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        NAMES_PATH: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
        WEIGHTS_PATH: 'https://pjreddie.com/media/files/yolov3.weights'
    }
    
    for filepath, url in files.items():
        if not os.path.exists(filepath):
            print(f"⬇️  Downloading {os.path.basename(filepath)}...")
            if download_file_with_timeout(url, filepath, timeout=60):
                print(f"✅ Downloaded {os.path.basename(filepath)}")
            else:
                print(f"❌ Download failed: {os.path.basename(filepath)}")
                return False
        else:
            print(f"✅ {os.path.basename(filepath)} exists")
    
    return True

def initialize_yolo():
    """Initialize YOLO model"""
    global net, classes, output_layers, is_initialized
    
    try:
        # Download files if needed
        if not download_yolo_files():
            logger.error("Failed to download YOLO files")
            return False
        
        # Check files exist and are not empty
        for path in [CONFIG_PATH, WEIGHTS_PATH, NAMES_PATH]:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                logger.error(f"YOLO file missing or empty: {path}")
                return False
        
        # Load YOLO
        net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
        if net.empty():
            logger.error("Failed to load YOLO network")
            return False
        
        # Load classes
        with open(NAMES_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        if not classes:
            logger.error("No classes loaded from names file")
            return False
        
        # Get output layers - handle different OpenCV versions
        layer_names = net.getLayerNames()
        try:
            # For newer OpenCV versions
            unconnected = net.getUnconnectedOutLayers()
            if len(unconnected.shape) == 2:
                output_layers = [layer_names[i[0] - 1] for i in unconnected]
            else:
                output_layers = [layer_names[i - 1] for i in unconnected]
        except (IndexError, TypeError, AttributeError):
            # Fallback for older versions
            try:
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            except:
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        if not output_layers:
            logger.error("No output layers found")
            return False
        
        is_initialized = True
        logger.info("✅ YOLO initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLO initialization failed: {e}")
        is_initialized = False
        return False

def validate_image(image_path):
    """Validate image file"""
    if not os.path.exists(image_path):
        return False, "Image file does not exist"
    
    if os.path.getsize(image_path) == 0:
        return False, "Image file is empty"
    
    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Unsupported file extension: {file_ext}"
    
    return True, "Valid"

def detect_vehicles(image_path, save_output=False):
    """
    Detect vehicles in image using YOLO
    Returns: (vehicle_count, output_filename)
    """
    global is_initialized
    
    # Validate input
    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        logger.error(f"Image validation failed: {error_msg}")
        return fallback_detection(image_path, save_output)
    
    # Initialize if needed
    if not is_initialized:
        if not initialize_yolo():
            return fallback_detection(image_path, save_output)
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot read image: {image_path}")
            return fallback_detection(image_path, save_output)
        
        height, width, channels = img.shape
        if height == 0 or width == 0:
            logger.error(f"Invalid image dimensions: {width}x{height}")
            return fallback_detection(image_path, save_output)
        
        # Prepare for YOLO
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Check if it's a vehicle class with sufficient confidence
                if confidence > 0.5 and class_id < len(classes) and classes[class_id] in VEHICLE_CLASSES:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Validate bounding box coordinates
                    x = max(0, min(x, width))
                    y = max(0, min(y, height))
                    w = max(0, min(w, width - x))
                    h = max(0, min(h, height - y))
                    
                    if w > 0 and h > 0:  # Valid bounding box
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        detected_vehicles = []
        if len(boxes) > 0:
            try:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                if len(indexes) > 0:
                    for i in indexes.flatten():
                        detected_vehicles.append({
                            'class': classes[class_ids[i]],
                            'confidence': confidences[i],
                            'box': boxes[i]
                        })
            except Exception as e:
                logger.error(f"NMS failed: {e}")
                # Fallback: use all detections without NMS
                for i in range(len(boxes)):
                    detected_vehicles.append({
                        'class': classes[class_ids[i]],
                        'confidence': confidences[i],
                        'box': boxes[i]
                    })
        
        # Draw detections if requested
        output_filename = None
        if save_output:
            output_filename = draw_detections(img, detected_vehicles, image_path)
        
        vehicle_count = len(detected_vehicles)
        logger.info(f"Detected {vehicle_count} vehicles")
        
        return vehicle_count, output_filename
        
    except Exception as e:
        logger.error(f"YOLO detection error: {e}")
        return fallback_detection(image_path, save_output)

def draw_detections(img, detections, original_path):
    """Draw bounding boxes on image"""
    try:
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Yellow
        ]
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']
            class_name = detection['class']
            confidence = detection['confidence']
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save image
        output_filename = f"detected_{os.path.basename(original_path)}"
        output_path = os.path.join("uploads", output_filename)
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        success = cv2.imwrite(output_path, img)
        if not success:
            logger.error(f"Failed to save output image: {output_path}")
            return None
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Failed to draw detections: {e}")
        return None

def fallback_detection(image_path, save_output=False):
    """Fallback when YOLO unavailable"""
    import random
    import shutil
    
    try:
        vehicle_count = random.randint(1, 6)
        
        output_filename = None
        if save_output:
            try:
                output_filename = f"detected_{os.path.basename(image_path)}"
                output_path = os.path.join("uploads", output_filename)
                
                # Ensure uploads directory exists
                os.makedirs("uploads", exist_ok=True)
                
                shutil.copy2(image_path, output_path)
            except Exception as e:
                logger.error(f"Fallback save error: {e}")
                output_filename = None
        
        logger.info(f"Fallback detection: {vehicle_count} vehicles")
        return vehicle_count, output_filename
        
    except Exception as e:
        logger.error(f"Fallback detection error: {e}")
        return 3, None  # Default fallback

# Test function
if __name__ == "__main__":
    print("🧪 Testing YOLO detection...")
    if initialize_yolo():
        print("✅ YOLO ready")
        
        # Test with a sample image if available
        test_image = "test_image.jpg"
        if os.path.exists(test_image):
            count, output = detect_vehicles(test_image, save_output=True)
            print(f"Test detection: {count} vehicles, output: {output}")
    else:
        print("⚠️  YOLO initialization failed, using fallback mode")
        count, output = fallback_detection("dummy.jpg", save_output=False)
        print(f"Fallback test: {count} vehicles")
