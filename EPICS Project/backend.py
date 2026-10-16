import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from flask import (Flask, abort, flash, redirect, render_template, request,
                   session, url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
app = Flask(__name__)

SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    # Keeps local development working without a .env, but sessions will not
    # survive a restart. Always set SECRET_KEY in deployed environments.
    SECRET_KEY = secrets.token_hex(32)
    app.logger.warning(
        'SECRET_KEY is not set. Generated a temporary key: sessions will be '
        'invalidated on every restart. Set SECRET_KEY in your .env file.'
    )

app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL') or (
    'sqlite:///' + os.path.join(app.instance_path, 'app.db')
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# A dead database host should fail loudly in seconds, not hang the boot forever.
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres'):
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'connect_args': {'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 5))},
    }

# Upload / request limits. Reject oversized bodies before they reach the model.
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_BYTES', 5 * 1024 * 1024))
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER') or os.path.join(BASE_DIR, 'uploads')

# Session hardening. SESSION_COOKIE_SECURE should be enabled behind HTTPS.
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0') == '1'
app.config['CSRF_ENABLED'] = os.getenv('CSRF_ENABLED', '1') != '0'

# The trained weights are kept out of version control; point MODEL_PATH at them.
MODEL_PATH = os.getenv('MODEL_PATH') or os.path.join(BASE_DIR, 'imagenet_model.h5')
MODEL_METRICS_PATH = os.getenv('MODEL_METRICS_PATH') or os.path.join(
    BASE_DIR, 'model_metrics.json'
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
IMAGE_SIZE = (224, 224)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to continue.'
login_manager.session_protection = 'basic'


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    scans = db.relationship('Scan', backref='user', lazy=True,
                            cascade='all, delete-orphan')


class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(100), nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    risk_class = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def init_db():
    """Create any missing tables. Safe to call repeatedly."""
    with app.app_context():
        db.create_all()


# --------------------------------------------------------------------------
# CSRF protection (session-token based, no extra dependency)
# --------------------------------------------------------------------------
def csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


# Make {{ csrf_token() }} available to every template.
app.jinja_env.globals['csrf_token'] = csrf_token


@app.before_request
def csrf_protect():
    if not app.config.get('CSRF_ENABLED', True):
        return
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return
    sent = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    expected = session.get('_csrf_token')
    if not expected or not sent or not hmac.compare_digest(str(sent), str(expected)):
        abort(400, description='CSRF token missing or invalid. Please reload the page.')


# --------------------------------------------------------------------------
# Model handling (loaded lazily so importing the app stays cheap)
# --------------------------------------------------------------------------
_MODEL = None


def build_model():
    """Rebuild the transfer-learning architecture used during training."""
    base = EfficientNetB3(weights='imagenet', include_top=False,
                          input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False

    model = models.Sequential([
        layers.Input((*IMAGE_SIZE, 3)),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.4),
        layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    return model


def get_model():
    """Return the cached model, loading the weights on first use."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'Model weights not found at {MODEL_PATH!r}. '
                'Set the MODEL_PATH environment variable to the location of '
                'imagenet_model.h5.'
            )
        model = build_model()
        model.load_weights(MODEL_PATH)
        _MODEL = model
    return _MODEL


def load_model_metrics():
    """Read the recorded evaluation metrics, if the file is present."""
    try:
        with open(MODEL_METRICS_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def allowed_file(filename):
    return ('.' in filename
            and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_image(filepath):
    """Load an image from disk into the (1, 224, 224, 3) array the model wants."""
    img = load_img(filepath, target_size=IMAGE_SIZE)
    return np.expand_dims(img_to_array(img), axis=0)


RECOMMENDATIONS = {
    'low-risk': (
        '<p><strong>Immediate Action:</strong> Continue with your current wellness '
        'routine. No immediate clinical intervention is suggested based on this '
        'scan.</p><p><strong>Diagnostic Steps:</strong> Maintain your schedule for '
        'annual screening mammograms as recommended by your doctor. Annual '
        'screenings are the best way to catch changes over time.</p>'
    ),
    'mod-risk': (
        '<p><strong>Immediate Action:</strong> Schedule a consultation with your '
        'primary physician or radiologist within the next 2 weeks to discuss these '
        'findings.</p><p><strong>Diagnostic Steps:</strong> Your doctor may request '
        "'Diagnostic Imaging,' which is more detailed than a screening. This "
        'typically includes a targeted Breast Ultrasound or a Diagnostic Mammogram '
        '(magnification views) to clarify the area of concern.</p>'
    ),
    'high-risk': (
        '<p><strong>Immediate Action:</strong> Contact a Breast Specialist or an '
        'Oncology clinic as soon as possible for a priority consultation. This '
        'result indicates a high clinical suspicion that requires investigation.</p>'
        "<p><strong>Diagnostic Steps:</strong> Be prepared for a 'Biopsy' "
        'recommendation. A core needle biopsy is the standard procedure to get a '
        'definitive diagnosis. Your specialist may also order a Breast MRI to '
        'evaluate the extent of the tissue involvement.</p>'
    ),
}


def classify_probability(pred):
    """Map a model probability to (label, risk_class). Thresholds are unchanged."""
    if pred < 0.30:
        return 'LOW PROBABILITY', 'low-risk'
    if pred < 0.50:
        return 'MODERATE CONCERN', 'mod-risk'
    return 'HIGH PROBABILITY', 'high-risk'


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if not username or not email or not password:
            flash('All fields are required.')
            return render_template('signup.html'), 400

        if len(password) < 8:
            flash('Password must be at least 8 characters long.')
            return render_template('signup.html'), 400

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login.')
            return render_template('signup.html'), 409

        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.')
            return render_template('signup.html'), 409

        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256'),
        )
        try:
            db.session.add(new_user)
            db.session.commit()
        except Exception:  # pragma: no cover - only on unexpected DB failure
            db.session.rollback()
            app.logger.exception('Signup failed')
            flash('Something went wrong creating your account. Please try again.')
            return render_template('signup.html'), 500

        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid email or password')
    return render_template('login.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('landing'))


# --------------------------------------------------------------------------
# Content routes
# --------------------------------------------------------------------------
@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/dashboard')
@login_required
def index():
    return render_template('index.html')


@app.route('/prevention')
def prevention():
    return render_template('prevention.html')


@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


@app.route('/history')
@login_required
def history():
    user_scans = (Scan.query
                  .filter_by(user_id=current_user.id)
                  .order_by(Scan.created_at.desc())
                  .all())
    return render_template('history.html', scans=user_scans)


@app.route('/report/<int:scan_id>')
@login_required
def view_report(scan_id):
    scan = db.session.get(Scan, scan_id)
    if scan is None:
        abort(404)
    if scan.user_id != current_user.id:
        abort(403)
    return render_template('result.html',
                           prediction=scan.prediction,
                           recommendation=scan.recommendation,
                           risk_class=scan.risk_class,
                           created_at=scan.created_at,
                           metrics=load_model_metrics())


# --------------------------------------------------------------------------
# Prediction route
# --------------------------------------------------------------------------
@app.route('/analyzer', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method != 'POST':
        return render_template('analyzer.html')

    file = request.files.get('imagefile')
    if file is None:
        flash('No file uploaded.')
        return render_template('analyzer.html'), 400
    if not file.filename:
        flash('No file selected.')
        return render_template('analyzer.html'), 400
    if not allowed_file(file.filename):
        flash('Unsupported file type. Please upload a PNG, JPG, BMP or WEBP image.')
        return render_template('analyzer.html'), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    safe_name = secure_filename(file.filename) or 'scan'
    ext = safe_name.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{uuid.uuid4().hex}.{ext}')

    try:
        file.save(filepath)
        pred = float(get_model().predict(preprocess_image(filepath))[0][0])
    except RequestEntityTooLarge:
        raise
    except Exception:
        app.logger.exception('Prediction failed for upload %s', safe_name)
        flash('We could not analyze that image. Please try a different file.')
        return render_template('analyzer.html'), 500
    finally:
        # Never keep patient images around after scoring them.
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:  # pragma: no cover
                app.logger.warning('Could not delete temp upload %s', filepath)

    label, risk_class = classify_probability(pred)
    prediction = f'{label} ({pred * 100:.2f}%)'
    recommendation = RECOMMENDATIONS[risk_class]

    new_scan = Scan(
        prediction=prediction,
        recommendation=recommendation,
        risk_class=risk_class,
        user_id=current_user.id,
    )
    db.session.add(new_scan)
    db.session.commit()

    return render_template('result.html',
                           prediction=prediction,
                           recommendation=recommendation,
                           risk_class=risk_class,
                           scan_id=new_scan.id,
                           created_at=new_scan.created_at,
                           metrics=load_model_metrics())


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    limit_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return render_template('error.html', code=413,
                           message=f'That image is too large. '
                                   f'Maximum upload size is {limit_mb:.0f} MB.'), 413


@app.errorhandler(400)
def handle_bad_request(error):
    return render_template('error.html', code=400,
                           message=getattr(error, 'description', 'Bad request.')), 400


@app.errorhandler(403)
def handle_forbidden(_error):
    return render_template('error.html', code=403,
                           message='You do not have access to that report.'), 403


@app.errorhandler(404)
def handle_not_found(_error):
    return render_template('error.html', code=404,
                           message='We could not find the page you were looking for.'), 404


@app.errorhandler(500)
def handle_server_error(_error):  # pragma: no cover
    db.session.rollback()
    return render_template('error.html', code=500,
                           message='Something went wrong on our side.'), 500


def resolve_port(default=8000):
    """PORT from the environment, falling back to `default` when unset or 0.

    A blank or zero PORT would otherwise make the OS pick a random port, which
    is confusing during local development (and breaks hard-coded preview URLs).
    """
    try:
        port = int(os.getenv('PORT') or default)
    except ValueError:
        return default
    return port if port > 0 else default


if __name__ == '__main__':
    init_db()
    app.run(
        host=os.getenv('HOST') or '127.0.0.1',
        port=resolve_port(),
        debug=os.getenv('FLASK_DEBUG', '0') == '1',
        use_reloader=False,
    )
