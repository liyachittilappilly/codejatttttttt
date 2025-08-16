import os
import cv2
import numpy as np
import math
import os
import cv2
import numpy as np
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from ultralytics import YOLO
from mediapipe import solutions
from collections import deque
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import uuid
from flask import Response


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MODEL_FOLDER'] = 'models'
app.config['DATA_FOLDER'] = 'data'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], '0'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], '3'), exist_ok=True)

# Load models for abs detector
try:
    feature_extractor = tf.keras.models.load_model(os.path.join(app.config['MODEL_FOLDER'], 'feature_extractor.h5'))
    classifier = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'classifier.pkl'))
    abs_model_loaded = True
except:
    abs_model_loaded = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

@app.route('/')
def index():
    return render_template('index.html', abs_model_loaded=abs_model_loaded)

@app.route('/train_abs', methods=['POST'])
def train_abs():
    try:
        # Import and run the training function
        from train_abs import train_model
        train_model(app.config['DATA_FOLDER'], app.config['MODEL_FOLDER'])
        
        # Reload models after training
        global feature_extractor, classifier, abs_model_loaded
        feature_extractor = tf.keras.models.load_model(os.path.join(app.config['MODEL_FOLDER'], 'feature_extractor.h5'))
        classifier = joblib.load(os.path.join(app.config['MODEL_FOLDER'], 'classifier.pkl'))
        abs_model_loaded = True
        
        return jsonify({'success': True, 'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Training failed: {str(e)}'})

@app.route('/upload_abs_data', methods=['POST'])
def upload_abs_data():
    if 'no_abs_files' not in request.files or 'abs_files' not in request.files:
        return jsonify({'success': False, 'message': 'Both file sets are required'})
    
    no_abs_files = request.files.getlist('no_abs_files')
    abs_files = request.files.getlist('abs_files')
    
    # Save no abs images
    for file in no_abs_files:
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['DATA_FOLDER'], '0', filename))
    
    # Save abs images
    for file in abs_files:
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['DATA_FOLDER'], '3', filename))
    
    return jsonify({'success': True, 'message': 'Data uploaded successfully!'})

@app.route('/abs_detector', methods=['POST'])
def abs_detector():
    if not abs_model_loaded:
        return jsonify({'error': 'Abs detector model not loaded. Please train the model first.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Process image
    img = load_img(file_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Extract features
    features = feature_extractor.predict(img_array)
    features = features.reshape(1, -1)
    
    # Predict class
    prediction = classifier.predict(features)[0]
    probability = classifier.predict_proba(features)[0]
    
    # Map prediction to label
    label = "No Abs" if prediction == 0 else "Abs"
    confidence = probability.max()
    
    # Save the prediction result
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {label}\nConfidence: {confidence:.2%}", fontsize=14)
    plt.axis('off')
    
    result_filename = f"abs_result_{uuid.uuid4().hex[:8]}.png"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    plt.savefig(result_path)
    plt.close()
    
    return jsonify({
        'result': label,
        'confidence': f"{confidence:.2%}",
        'image_path': f"/results/{result_filename}"
    })

@app.route('/barcode_detector', methods=['POST'])  # Changed from barcode_counter
def barcode_counter():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process image using the working standalone function
        num_lines = count_barcode_lines(file_path)
        
        # For visualization, we need to process the image again to get the peaks
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'})
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        kernel = np.ones((5, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        height, width = cleaned.shape
        projection = np.sum(cleaned, axis=0) / 255
        smoothed = np.convolve(projection, np.ones(5)/5, mode='same')
        
        min_bar_width = width // 30
        peaks = []
        in_peak = False
        peak_start = 0
        
        for i in range(width):
            if smoothed[i] > 0.5:
                if not in_peak:
                    peak_start = i
                    in_peak = True
            else:
                if in_peak and (i - peak_start) >= min_bar_width:
                    peaks.append((peak_start + i) // 2)
                in_peak = False
        
        if in_peak and (width - peak_start) >= min_bar_width:
            peaks.append((peak_start + width) // 2)
        
        # Save visualization
        result_img = img.copy()
        for peak in peaks:
            cv2.line(result_img, (peak, 0), (peak, height), (0, 255, 0), 2)
        
        result_filename = f"barcode_result_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        return jsonify({
            'result': f"Barcode lines: {num_lines}",
            'image_path': f"/results/{result_filename}"
        })
    
    except Exception as e:
        app.logger.error(f"Error in barcode_counter: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'})

# Add the standalone function to your app.py
def count_barcode_lines(image_path):
    # Step 1: Convert image to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Noise reduction with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Adaptive thresholding with inverted binary
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Inverted to make bars white
        11, 2
    )
    
    # Step 4: Morphological operations to clean up
    kernel = np.ones((5, 1), np.uint8)  # Vertical kernel
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 5: Vertical projection profile
    height, width = cleaned.shape
    projection = np.sum(cleaned, axis=0) / 255  # Normalize to 0-1
    
    # Step 6: Smooth the projection profile
    smoothed = np.convolve(projection, np.ones(5)/5, mode='same')
    
    # Step 7: Find peaks in the projection profile
    min_bar_width = width // 30  # Minimum expected bar width
    peaks = []
    in_peak = False
    peak_start = 0
    
    for i in range(width):
        if smoothed[i] > 0.5:  # Threshold for bar detection
            if not in_peak:
                peak_start = i
                in_peak = True
        else:
            if in_peak and (i - peak_start) >= min_bar_width:
                peaks.append((peak_start + i) // 2)  # Center of the bar
            in_peak = False
    
    # Handle case where image ends with a bar
    if in_peak and (width - peak_start) >= min_bar_width:
        peaks.append((peak_start + width) // 2)
    
    return len(peaks)

@app.route('/eye_detector', methods=['POST'])
def eye_counter():
    # This function will now handle both image upload and real-time camera detection
    # based on the request type
    
    if 'file' in request.files:
        # Handle image upload (existing functionality)
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Create a new MediaPipe Face Mesh instance for image processing
        mp_face_mesh = solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            eye_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Left eye landmarks (indices from MediaPipe documentation)
                    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                    # Right eye landmarks
                    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                    
                    # Check if eye landmarks are visible
                    left_visible = any(face_landmarks.landmark[i].visibility > 0.5 for i in left_eye)
                    right_visible = any(face_landmarks.landmark[i].visibility > 0.5 for i in right_eye)
                    
                    if left_visible:
                        eye_count += 1
                    if right_visible:
                        eye_count += 1
            
            # Save visualization
            result_filename = f"eye_result_{uuid.uuid4().hex[:8]}.png"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, image)
            
            return jsonify({
                'result': f"Eyes detected: {eye_count}",
                'image_path': f"/results/{result_filename}"
            })
    
    elif request.json and request.json.get('action') == 'start_camera':
        # Handle real-time camera detection
        return jsonify({
            'status': 'camera_started',
            'message': 'Camera started successfully'
        })
    
    else:
        return jsonify({'error': 'Invalid request'})

# Add a new route for real-time camera feed
@app.route('/eye_camera_feed')
def eye_camera_feed():
    def generate_frames():
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = face_mesh.process(rgb_frame)
                
                # Simple approach: count 2 eyes per detected face
                face_count = 0
                if results.multi_face_landmarks:
                    face_count = len(results.multi_face_landmarks)
                
                eye_count = face_count * 2
                
                # Draw landmarks for visualization
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Left eye landmarks
                        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                        # Right eye landmarks
                        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                        
                        # Draw eye landmarks
                        for idx in left_eye:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        
                        for idx in right_eye:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Display debug information
                cv2.putText(frame, f"Faces: {face_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Eyes: {eye_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            
            cap.release()
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/cat_detector', methods=['POST'])
def cat_head_counter():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Load YOLOv8 model pre-trained on COCO
        model = YOLO('yolov8n.pt')
        
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Detect cats (class 15 in COCO)
        results = model(image, classes=[15])
        
        if len(results[0].boxes) == 0:
            return jsonify({'result': "No cat detected"})
        
        # Get the first detected cat
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cat_crop = image[y1:y2, x1:x2].copy()
        
        # Preprocess the cat crop
        gray = cv2.cvtColor(cat_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({'result': "No contour found in the cat region"})
        
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Compute centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return jsonify({'result': "Invalid contour moments"})
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = (cx, cy)
        
        # Find the top-most point (likely head for cats)
        min_y = float('inf')
        head_point = None
        for point in contour:
            point = point[0]
            if point[1] < min_y:
                min_y = point[1]
                head_point = point
        
        # Find convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        # Count significant defects near the head
        head_defects = 0
        threshold_depth = 40  # Increased threshold for cats
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                depth = d / 256.0
                
                # Check if defect is near the head and significant
                if depth > threshold_depth:
                    # Check if defect is in the upper half of the cat
                    if far[1] < cy:
                        # Calculate distance to head point
                        dist_to_head = np.linalg.norm(far - head_point)
                        if dist_to_head < 30:  # Reduced distance threshold
                            head_defects += 1
        
        # Count heads based on defects - be more conservative
        num_heads = 1 + min(head_defects, 1)  # Cap at 2 heads maximum
        
        # Additional verification using ear detection
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cat_crop, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color (adjust as needed)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours in skin mask
        skin_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count potential head regions (skin-colored areas in upper part)
        head_regions = 0
        for cnt in skin_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Check if region is in upper part and has reasonable size
            if y < cy and w > 15 and h > 15:  # Increased size threshold
                head_regions += 1
        
        # Take maximum of both methods but cap at 2
        num_heads = max(num_heads, min(head_regions, 2))
        
        # For normal cats, we expect only one head, so we can cap it at 1
        # If you want to detect multi-headed cats, remove this line
        num_heads = min(num_heads, 1)
        
        # Draw results for visualization
        cv2.drawContours(cat_crop, [contour], -1, (0, 255, 0), 2)
        cv2.circle(cat_crop, centroid, 5, (255, 0, 0), -1)
        cv2.circle(cat_crop, tuple(head_point), 5, (0, 0, 255), -1)
        
        # Save visualization
        result_filename = f"cat_result_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, cat_crop)
        
        return jsonify({
            'result': f"Cat heads: {num_heads}",
            'image_path': f"/results/{result_filename}"
        })
    
    except Exception as e:
        app.logger.error(f"Error in cat_head_counter: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'})
    
@app.route('/bubble_detector', methods=['POST'])
def bubble_counter():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try different thresholding methods
        # Method 1: Otsu's thresholding
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Method 3: Inverted Otsu's (for dark bubbles on light background)
        _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try all three methods and choose the one with the most reasonable bubble count
        methods = [
            ("Otsu", binary_otsu),
            ("Adaptive", binary_adaptive),
            ("Inverted Otsu", binary_inv)
        ]
        
        best_method = None
        best_count = 0
        best_contours = []
        
        for method_name, binary in methods:
            # Morphological opening to remove noise
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area (remove small artifacts)
            min_contour_area = 10  # Reduced threshold to detect smaller bubbles
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            # Also filter by circularity to remove non-bubble shapes
            circular_contours = []
            for cnt in valid_contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Threshold for circularity
                        circular_contours.append(cnt)
            
            count = len(circular_contours)
            
            # Choose the method with the most bubbles but not too many (which would indicate noise)
            if count > best_count and count < 100:  # Upper limit to avoid counting noise
                best_count = count
                best_method = method_name
                best_contours = circular_contours
        
        # If no method found a reasonable number of bubbles, use the first method
        if best_count == 0:
            best_contours = valid_contours
            best_count = len(best_contours)
            best_method = "Default"
        
        # Create visualization
        output_img = img.copy()
        cv2.drawContours(output_img, best_contours, -1, (0, 255, 0), 2)
        cv2.putText(output_img, f'Bubble Count: {best_count} (Method: {best_method})', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save visualization
        result_filename = f"bubble_result_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, output_img)
        
        return jsonify({
            'result': f"Bubbles detected: {best_count}",
            'image_path': f"/results/{result_filename}"
        })
    
    except Exception as e:
        app.logger.error(f"Error in bubble_counter: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'})
@app.route('/bristle_detector', methods=['POST'])
def bristle_counter():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Step 1: Read the image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Invalid image file'})
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 4: Edge detection using Canny
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Step 5: Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)
        
        # If no lines detected, return 0
        if lines is None:
            return jsonify({'result': "No bristles detected"})
        
        # Step 6: Filter vertical lines (bristles)
        vertical_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Check if line is near-vertical (80° to 100°)
            if 80 <= abs(angle) <= 100:
                vertical_lines += 1
        
        # Draw detected lines for visualization
        output = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            color = (0, 255, 0) if 80 <= abs(angle) <= 100 else (0, 0, 255)
            cv2.line(output, (x1, y1), (x2, y2), color, 2)
        
        # Add text with bristle count
        cv2.putText(output, f'Bristles detected: {vertical_lines}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save visualization
        result_filename = f"bristle_result_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, output)
        
        return jsonify({
            'result': f"Bristles detected: {vertical_lines}",
            'image_path': f"/results/{result_filename}"
        })
    
    except Exception as e:
        app.logger.error(f"Error in bristle_counter: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'})
    
@app.route('/rain_detector', methods=['POST'])
def rain_detector():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Initialize rain detector
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16,
            detectShadows=False
        )
        
        # Feature thresholds
        edge_density_threshold = 0.008
        fg_density_low = 0.01
        fg_density_high = 0.15
        intensity_threshold = 100
        streak_ratio_threshold = 0.3
        
        # Temporal smoothing
        smoothing_window = 5
        rain_prob_history = deque(maxlen=smoothing_window)
        
        # Initialize video capture
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video file'})
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frames = int(fps * 1.0)  # Minimum 1 second for rain event
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Morphological kernel for noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        rain_events = 0
        consecutive_rainy_frames = 0
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Skip first 30 frames for background model initialization
        for _ in range(30):
            ret, _ = cap.read()
            if not ret:
                break
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Preprocess frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
            # Background subtraction
            fg_mask = bg_subtractor.apply(blurred)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Feature 1: Foreground density
            fg_pixels = cv2.countNonZero(fg_mask)
            fg_density = fg_pixels / (frame_width * frame_height)
            
            # Feature 2: Edge detection in foreground regions
            edges = cv2.Canny(blurred, 50, 150)
            edges_in_fg = cv2.bitwise_and(edges, edges, mask=fg_mask)
            edge_pixels = cv2.countNonZero(edges_in_fg)
            edge_density = edge_pixels / (frame_width * frame_height)
            
            # Feature 3: Average intensity of foreground
            avg_intensity = cv2.mean(blurred, mask=fg_mask)[0]
            
            # Feature 4: Streak analysis
            lines = cv2.HoughLinesP(edges_in_fg, 1, np.pi/180, 30, 
                                   minLineLength=10, maxLineGap=5)
            streak_count = 0
            diagonal_count = 0
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 30 < angle < 60 or 120 < angle < 150:
                        diagonal_count += 1
                    streak_count += 1
            
            streak_ratio = diagonal_count / max(streak_count, 1)
            
            # Rule-based classification
            conditions = [
                fg_density > fg_density_low,
                fg_density < fg_density_high,
                edge_density > edge_density_threshold,
                avg_intensity > intensity_threshold,
                streak_ratio > streak_ratio_threshold
            ]
            
            rain_prob = sum(conditions) / len(conditions)
            rain_prob_history.append(rain_prob)
            smoothed_prob = np.mean(rain_prob_history)
            
            is_rainy = smoothed_prob > 0.6
            
            # Update rain event counter
            if is_rainy:
                consecutive_rainy_frames += 1
            else:
                if consecutive_rainy_frames >= min_frames:
                    rain_events += 1
                consecutive_rainy_frames = 0
            
            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% | Frame {frame_count}/{total_frames} | Rain events: {rain_events}")
        
        # Check for rain event at end of video
        if consecutive_rainy_frames >= min_frames:
            rain_events += 1
        
        cap.release()
        
        return jsonify({
            'result': f"Rain events detected: {rain_events}"
        })
    
    except Exception as e:
        app.logger.error(f"Error in rain_detector: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)