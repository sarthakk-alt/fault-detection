import os
import uuid
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = "best.pt"
if os.path.exists(MODEL_PATH):
    print(f"Loading custom trained YOLOv8 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
else:
    fallback_model = "yolov8n.pt"
    print(f"Warning: Custom weights not found at {MODEL_PATH}. Loading fallback model {fallback_model}...")
    model = YOLO(fallback_model)

def get_recommendations_and_caption(counts):
    caption_parts = []
    recommendations = []
    
    # Check for custom classes and standard coco classes if fallback
    corrosion_cnt = counts.get('corrosion', 0)
    crack_cnt = counts.get('crack', 0)
    dent_cnt = counts.get('dent', 0)
    
    total_damage = corrosion_cnt + crack_cnt + dent_cnt
    
    # Handle standard model detections if we fall back to COCO yolov8n
    other_classes = {k: v for k, v in counts.items() if k not in ['corrosion', 'crack', 'dent'] and v > 0}
    
    if total_damage == 0 and not other_classes:
        caption = "Inspection completed. No surface or structural damage (corrosion, cracks, or dents) was identified on the component."
        recommendations = [
            "Component is cleared for continued operational service.",
            "Schedule next routine visual inspection in accordance with maintenance manuals."
        ]
        return caption, recommendations

    # Build counts text
    counts_text = []
    if corrosion_cnt > 0:
        counts_text.append(f"{corrosion_cnt} corrosion patch{'es' if corrosion_cnt > 1 else ''}")
    if crack_cnt > 0:
        counts_text.append(f"{crack_cnt} crack{'s' if crack_cnt > 1 else ''}")
    if dent_cnt > 0:
        counts_text.append(f"{dent_cnt} dent{'s' if dent_cnt > 1 else ''}")
        
    for cls_name, cnt in other_classes.items():
        counts_text.append(f"{cnt} {cls_name}{'s' if cnt > 1 else ''}")
            
    caption_parts.append(f"Visual anomalies identified. Detected: {', '.join(counts_text)}.")
    
    # Specific warnings and recommendations
    if crack_cnt > 0:
        caption_parts.append("Surface cracks indicate structural stress propagation, requiring immediate NDT depth measurement.")
        recommendations.extend([
            "CRITICAL: Perform immediate Non-Destructive Testing (NDT) (Eddy Current, Ultrasonic, or Dye Penetrant) to determine crack depth and propagation path.",
            "Cross-reference findings with the Structural Repair Manual (SRM) to evaluate if patch reinforcement or skin panel replacement is required."
        ])
        
    if dent_cnt > 0:
        caption_parts.append("Dents suggest localized external impact, which could affect local aerodynamic flow and skin integrity.")
        recommendations.extend([
            "Measure maximum dent depth and width using a digital dial indicator.",
            "Verify compliance against the SRM allowable damage limits. Monitor for sub-surface delamination if composite structure."
        ])
        
    if corrosion_cnt > 0:
        caption_parts.append("Surface corrosion detected on sheet metal surface, indicating coating degradation.")
        recommendations.extend([
            "Perform mechanical blending to remove surface corrosion down to bare sound metal.",
            "Verify remaining skin thickness after blending to ensure structural margin.",
            "Apply protective wash-primer and polyurethane topcoat to restore surface protection."
        ])

    if other_classes:
        caption_parts.append("General objects detected (model running in fallback mode).")
        recommendations.append("Verify model weights and ensure training on the aircraft dataset is complete.")

    caption = " ".join(caption_parts)
    return caption, recommendations

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/inspect')
def inspect():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
        
    try:
        # Secure filename and save
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Run YOLOv8 prediction (conf threshold 0.25)
        results = model(filepath, conf=0.25)
        result = results[0]
        
        # Draw bounding boxes and save annotated image
        annotated_filename = f"annotated_{unique_filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        
        annotated_img = result.plot()
        cv2.imwrite(annotated_path, annotated_img)
        
        # Process detections
        damages = []
        counts = {'corrosion': 0, 'crack': 0, 'dent': 0}
        
        # YOLOv8 class mapping
        names = result.names
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id)).lower()
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                
                # Check if it fits our classes
                if cls_name in counts:
                    counts[cls_name] += 1
                else:
                    if cls_name not in counts:
                        counts[cls_name] = 0
                    counts[cls_name] += 1
                
                damages.append({
                    'class': cls_name,
                    'confidence': round(conf * 100, 1),
                    'box': xyxy
                })
                
        # Generate captions and recommendations
        caption, recommendations = get_recommendations_and_caption(counts)
        
        return jsonify({
            'success': True,
            'original_url': f'/static/uploads/{unique_filename}',
            'annotated_url': f'/static/uploads/{annotated_filename}',
            'damages': damages,
            'counts': counts,
            'caption': caption,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
