import os
from urllib.parse import quote_plus

from app1 import create_app as create_app1
from app2 import create_app as create_app2
from app3 import create_app as create_app3
from app4 import create_app as create_app4
from app5 import create_app as create_app5
from app6 import create_app as create_app6
from app7 import create_app as create_app7
from flask import Flask, redirect, url_for, session

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

app = Flask(__name__)
if load_dotenv:
    load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration settings
def _get_env_bool(key, default=False):
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def _get_env_int(key, default):
    value = os.getenv(key)
    try:
        return int(value) if value else default
    except ValueError:
        return default

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['MYSQL_CONFIG'] = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', '09062005Sayak@'),
    'database': os.getenv('MYSQL_DATABASE', 'thermoai_user_db'),
    'port': _get_env_int('MYSQL_PORT', 3306),
}
app.config['SMTP_CONFIG'] = {
    'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
    'port': _get_env_int('SMTP_PORT', 587),
    'user': os.getenv('SMTP_USER', ''),
    'password': os.getenv('SMTP_PASSWORD', ''),
    'sender': os.getenv('SMTP_SENDER', ''),
    'use_tls': _get_env_bool('SMTP_USE_TLS', True),
    'use_ssl': _get_env_bool('SMTP_USE_SSL', False),
}
app.config['VERIFY_CODE_EXP_MINUTES'] = _get_env_int('VERIFY_CODE_EXP_MINUTES', 10)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

mysql_cfg = app.config['MYSQL_CONFIG']
mysql_password = quote_plus(mysql_cfg['password'])
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+mysqlconnector://{mysql_cfg['user']}:{mysql_password}"
    f"@{mysql_cfg['host']}:{mysql_cfg['port']}/{mysql_cfg['database']}"
)


# Register blueprints
app.register_blueprint(create_app1(), url_prefix='/app1')
app.register_blueprint(create_app2(), url_prefix='/app2')
app.register_blueprint(create_app3(), url_prefix='/app3')
app.register_blueprint(create_app4(), url_prefix='/app4')
app.register_blueprint(create_app5(), url_prefix='/app5')
app.register_blueprint(create_app6(), url_prefix='/app6')
app.register_blueprint(create_app7(), url_prefix='/app7')
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('app1.home'))
    return redirect(url_for('app2.login'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    
    
    
    
    
    
    
    
    
    