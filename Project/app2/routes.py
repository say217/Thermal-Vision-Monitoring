from flask import Blueprint, current_app, render_template, request, redirect, url_for, session, flash
from .models import db, User
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from sqlalchemy import text
from datetime import datetime, timedelta
import secrets
import smtplib
import ssl
from email.message import EmailMessage

bp = Blueprint('app2', __name__, template_folder='templates')

@bp.record
def record_params(setup_state):
    app = setup_state.app
    mysql_cfg = app.config.get('MYSQL_CONFIG', {})
    if mysql_cfg:
        _ensure_database_exists(mysql_cfg)
    db.init_app(app)
    with app.app_context():
        db.create_all()
        _ensure_user_columns()
        _seed_default_user()

def _ensure_database_exists(mysql_cfg):
    try:
        conn = mysql.connector.connect(
            host=mysql_cfg.get('host'),
            user=mysql_cfg.get('user'),
            password=mysql_cfg.get('password'),
            port=mysql_cfg.get('port', 3306),
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_cfg.get('database')}")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to create database: {exc}")

def _seed_default_user():
    if User.query.first():
        return
    default_user = User(
        name='Default User',
        email='default@example.com',
        password=generate_password_hash('ChangeMe123!'),
        is_verified=True
    )
    db.session.add(default_user)
    db.session.commit()

def _ensure_user_columns():
    db_name = current_app.config.get('MYSQL_CONFIG', {}).get('database')
    if not db_name:
        return
    with db.engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                """
            ),
            {'schema': db_name, 'table': User.__tablename__}
        )
        existing = {row[0] for row in rows}

        alters = []
        if 'is_verified' not in existing:
            alters.append("ALTER TABLE `user` ADD COLUMN `is_verified` TINYINT(1) NOT NULL DEFAULT 0")
        if 'verification_code_hash' not in existing:
            alters.append("ALTER TABLE `user` ADD COLUMN `verification_code_hash` VARCHAR(255) NULL")
        if 'verification_expires_at' not in existing:
            alters.append("ALTER TABLE `user` ADD COLUMN `verification_expires_at` DATETIME NULL")

        for stmt in alters:
            conn.execute(text(stmt))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            if not user.is_verified:
                code = _set_verification_code(user)
                db.session.commit()
                _send_verification_email(user.email, code)
                session['pending_verification_email'] = user.email
                flash('Verification required. We sent a code to your email.')
                return redirect(url_for('app2.verify'))
            session['user_id'] = user.id
            return redirect(url_for('app1.home'))
        flash('Invalid credentials')
    
    return render_template('login.html')

@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            if not existing_user.is_verified:
                code = _set_verification_code(existing_user)
                db.session.commit()
                _send_verification_email(existing_user.email, code)
                session['pending_verification_email'] = existing_user.email
                flash('Verification required. We sent a code to your email.')
                return redirect(url_for('app2.verify'))
            flash('Email already registered')
            return render_template('signup.html')
        
        user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            is_verified=False
        )
        code = _set_verification_code(user)
        db.session.add(user)
        db.session.commit()

        _send_verification_email(user.email, code)
        session['pending_verification_email'] = user.email
        flash('Verification code sent. Please check your email.')
        return redirect(url_for('app2.verify'))
    
    return render_template('signup.html')

@bp.route('/verify', methods=['GET', 'POST'])
def verify():
    email = session.get('pending_verification_email') or request.args.get('email')
    if not email:
        flash('Please sign up or log in first.')
        return redirect(url_for('app2.signup'))

    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Account not found.')
            return redirect(url_for('app2.signup'))

        if user.is_verified:
            session['user_id'] = user.id
            session.pop('pending_verification_email', None)
            return redirect(url_for('app1.home'))

        if not user.verification_expires_at or user.verification_expires_at < datetime.utcnow():
            flash('Verification code expired. We sent a new code.')
            code = _set_verification_code(user)
            db.session.commit()
            _send_verification_email(user.email, code)
            return redirect(url_for('app2.verify'))

        if user.verification_code_hash and check_password_hash(user.verification_code_hash, code):
            user.is_verified = True
            user.verification_code_hash = None
            user.verification_expires_at = None
            db.session.commit()
            session['user_id'] = user.id
            session.pop('pending_verification_email', None)
            return redirect(url_for('app1.home'))

        flash('Invalid verification code.')

    return render_template('verify.html', email=email)

def _set_verification_code(user):
    code = f"{secrets.randbelow(1000000):06d}"
    expiry_minutes = current_app.config.get('VERIFY_CODE_EXP_MINUTES', 10)
    user.verification_code_hash = generate_password_hash(code)
    user.verification_expires_at = datetime.utcnow() + timedelta(minutes=expiry_minutes)
    return code

def _send_verification_email(recipient, code):
    smtp_cfg = current_app.config.get('SMTP_CONFIG', {})
    host = smtp_cfg.get('host')
    port = smtp_cfg.get('port')
    user = smtp_cfg.get('user')
    password = smtp_cfg.get('password')
    sender = smtp_cfg.get('sender') or user
    use_tls = smtp_cfg.get('use_tls', True)
    use_ssl = smtp_cfg.get('use_ssl', False)

    if not host or not port or not user or not password or not sender:
        raise RuntimeError('SMTP is not configured. Please set SMTP_* in .env.')

    message = EmailMessage()
    message['Subject'] = 'Your verification code'
    message['From'] = sender
    message['To'] = recipient
    message.set_content(
        f"Your verification code is: {code}\n\n"
        "This code will expire soon. If you did not request this, ignore this email."
    )

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(user, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port) as server:
        if use_tls:
            context = ssl.create_default_context()
            server.starttls(context=context)
        server.login(user, password)
        server.send_message(message)

@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('app2.login'))