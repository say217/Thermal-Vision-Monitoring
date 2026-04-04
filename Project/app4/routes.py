import os
from flask import Blueprint, render_template, session, redirect, url_for, send_from_directory, abort, jsonify
from functools import wraps

_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app3', 'Results')
)

bp = Blueprint('app4', __name__, template_folder='templates')

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
    try:
        images = sorted(
            f for f in os.listdir(_RESULTS_DIR) if f.lower().endswith('.jpg')
        )
    except OSError:
        images = []
    return render_template('home3.html', images=images)

@bp.route('/result-image/<path:filename>')
@login_required
def result_image(filename):
    # Prevent path traversal — only allow simple filenames
    safe = os.path.basename(filename)
    if safe != filename or not safe.lower().endswith('.jpg'):
        abort(404)
    return send_from_directory(_RESULTS_DIR, safe)

@bp.route('/frame-meta/<path:filename>')
@login_required
def frame_meta(filename):
    """Return parsed metadata for a frame image as JSON."""
    safe = os.path.basename(filename)
    if safe != filename or not safe.lower().endswith('.jpg'):
        abort(404)
    # Derive txt filename: frame_X.jpg -> person_X.txt
    if safe.lower().startswith('frame_'):
        txt_name = 'person_' + safe[len('frame_'):-4] + '.txt'
    else:
        abort(404)
    txt_path = os.path.join(_RESULTS_DIR, txt_name)
    if not os.path.isfile(txt_path):
        return jsonify({'error': 'No metadata file found.'}), 404
    meta = {}
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, _, val = line.partition(':')
                    meta[key.strip()] = val.strip()
    except OSError:
        return jsonify({'error': 'Could not read metadata.'}), 500
    return jsonify(meta)







