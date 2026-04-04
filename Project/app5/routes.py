import base64
import os
from functools import wraps
from collections import Counter

import cv2
from flask import Blueprint, current_app, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename

from . import firmo

bp = Blueprint('app5', __name__, template_folder='templates')


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('app2.login'))
        return f(*args, **kwargs)

    return decorated_function


@bp.route('/')
@login_required
def page():
    return render_template('home4.html')


@bp.route('/predict', methods=['POST'])
@login_required
def predict():
    image_files = request.files.getlist('image')
    if not image_files:
        return render_template('home4.html', error='No image uploaded.')

    if len(image_files) > 1:
        return render_template('home4.html', error='Please upload only one image at a time.')

    image_file = image_files[0]
    if not image_file or image_file.filename == '':
        return render_template('home4.html', error='Please choose an image file.')

    upload_dir = current_app.config.get('UPLOAD_FOLDER')
    if not upload_dir:
        upload_dir = os.path.join(current_app.root_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(image_file.filename)
    save_path = os.path.join(upload_dir, filename)
    image_file.save(save_path)

    try:
        prediction = firmo.detect_from_image_path(save_path, conf=0.25)
        input_img = cv2.imread(save_path, cv2.IMREAD_UNCHANGED)
        img_h, img_w = (0, 0)
        img_channels = 0
        if input_img is not None:
            img_h, img_w = input_img.shape[:2]
            img_channels = 1 if len(input_img.shape) == 2 else int(input_img.shape[2])

        boxes = prediction['boxes']
        confidences = [float(item.get('confidence', 0.0)) for item in boxes]
        class_counter = Counter([item.get('label', 'obj') for item in boxes])
        sorted_counts = class_counter.most_common()
        max_count = sorted_counts[0][1] if sorted_counts else 0
        class_bars = []
        for label, count in sorted_counts:
            share = round((count / len(boxes)) * 100, 2) if boxes else 0.0
            width_pct = round((count / max_count) * 100, 2) if max_count else 0.0
            class_bars.append(
                {
                    'label': label,
                    'count': count,
                    'share_pct': share,
                    'width_pct': width_pct,
                }
            )

        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        max_conf = round(max(confidences), 4) if confidences else 0.0
        file_size_kb = round(os.path.getsize(save_path) / 1024.0, 2)

        metrics = {
            'detections': len(boxes),
            'unique_classes': len(class_counter),
            'avg_conf': avg_conf,
            'max_conf': max_conf,
            'top_class': sorted_counts[0][0] if sorted_counts else 'None',
        }

        image_meta = {
            'filename': filename,
            'content_type': image_file.mimetype or 'image/*',
            'width': img_w,
            'height': img_h,
            'channels': img_channels,
            'file_size_kb': file_size_kb,
        }

        annotated_ok, annotated_buf = cv2.imencode('.jpg', prediction['annotated_bgr'])
        cam_ok, cam_buf = cv2.imencode('.jpg', prediction['cam_overlay_bgr'])
        if not annotated_ok or not cam_ok:
            raise ValueError('Failed to encode prediction images.')

        annotated_b64 = base64.b64encode(annotated_buf.tobytes()).decode('utf-8')
        cam_b64 = base64.b64encode(cam_buf.tobytes()).decode('utf-8')

        return render_template(
            'home4.html',
            boxes=boxes,
            has_cam=prediction['has_cam'],
            annotated_image=annotated_b64,
            cam_image=cam_b64,
            uploaded_name=filename,
            metrics=metrics,
            class_bars=class_bars,
            image_meta=image_meta,
        )
    except Exception as exc:
        return render_template('home4.html', error=f'Prediction failed: {exc}')