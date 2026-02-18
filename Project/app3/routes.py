import os
import time
import cv2
from flask import Blueprint, render_template, session, redirect, url_for, request, current_app, jsonify, Response
from werkzeug.utils import secure_filename
from functools import wraps
from . import thermo

bp = Blueprint('app3', __name__, template_folder='templates')

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
    return render_template('home2.html')


@bp.route('/upload', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'ok': False, 'error': 'No video file provided.'}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'ok': False, 'error': 'Empty file name.'}), 400
    upload_dir = current_app.config.get('UPLOAD_FOLDER')
    if not upload_dir:
        upload_dir = os.path.join(current_app.root_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)
    thermo.start_processing(save_path, show_windows=False)
    return jsonify({'ok': True, 'path': save_path})


def _mjpeg_stream(kind):
    try:
        last_id = -1
        while True:
            frame_id, frame = thermo.get_latest_frame_with_id(kind)
            if frame is None or frame_id == last_id:
                time.sleep(0.01)
                continue
            last_id = frame_id
            h, w = frame.shape[:2]
            if w > 560:
                target_w = 560
                scale = target_w / float(w)
                frame = cv2.resize(frame, (target_w, int(h * scale)))
            ok, buffer = cv2.imencode(
                '.jpg',
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            )
            if not ok:
                time.sleep(0.01)
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
            time.sleep(0.01)
    except GeneratorExit:
        return


@bp.route('/stream/<kind>')
@login_required
def stream(kind):
    if kind not in {'color', 'explain'}:
        return jsonify({'ok': False, 'error': 'Invalid stream kind.'}), 400
    response = Response(
        _mjpeg_stream(kind),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.route('/logs')
@login_required
def logs():
    after = request.args.get('after', type=int)
    lines, latest_seq, reset = thermo.read_log_since(after, max_lines=200)
    return jsonify({'ok': True, 'lines': lines, 'latest_seq': latest_seq, 'reset': reset})


@bp.route('/events')
@login_required
def events():
    items = thermo.read_structured_tail(200)
    summary = thermo.summarize_recent_people(items, limit=4)
    latest_ts = summary[0]["timestamp"] if summary else None
    agent_summary = thermo.read_latest_agent_summary()
    return jsonify({
        'ok': True,
        'latest_ts': latest_ts,
        'summary': summary,
        'agent_summary': agent_summary,
        'agent_state': thermo.get_agent_state()
    })


@bp.route('/status')
@login_required
def status():
    return jsonify({'ok': True, 'state': thermo.processing_state})


@bp.route('/stop', methods=['POST'])
@login_required
def stop():
    thermo.stop_processing()
    return jsonify({'ok': True})












