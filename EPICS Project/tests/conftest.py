"""pytest fixtures.

Environment variables must be set *before* `backend` is imported, because the
Flask app reads its configuration (database URL, upload folder, secret key) at
import time.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix='bca-tests-')
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['DATABASE_URL'] = 'sqlite:///' + os.path.join(_TMP, 'test.db')
os.environ['UPLOAD_FOLDER'] = os.path.join(_TMP, 'uploads')
os.environ['CSRF_ENABLED'] = '1'

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import backend  # noqa: E402
from backend import Scan, User, app as flask_app, db  # noqa: E402


@pytest.fixture
def app():
    """A Flask app bound to an empty, throwaway SQLite database."""
    flask_app.config['TESTING'] = True
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
        yield flask_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


class FakeModel:
    """Stands in for the Keras model so tests never touch TensorFlow weights."""

    def __init__(self, score):
        self.score = score

    def predict(self, x, verbose=0):
        return np.array([[self.score]], dtype='float32')


@pytest.fixture
def set_model(monkeypatch):
    def _set(score):
        monkeypatch.setattr(backend, '_MODEL', FakeModel(score))
        return backend._MODEL
    return _set


def get_csrf(client):
    """Read the CSRF token the session issued for the current client."""
    with client.session_transaction() as session:
        return session.get('_csrf_token')


def prime_csrf(client, path='/login'):
    """GET a page containing a form so the session receives a CSRF token."""
    client.get(path)
    return get_csrf(client)


def register(client, username='alice', email='alice@example.com', password='secret123'):
    token = prime_csrf(client, '/signup')
    return client.post('/signup', data={
        'username': username,
        'email': email,
        'password': password,
        'csrf_token': token,
    })


def png_bytes(size=(32, 32), color=(200, 80, 120)):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def scan_for(app):
    """Create a scan owned by a specific user inside the test database."""
    def _create(user_id, prediction='HIGH PROBABILITY (81.00%)',
                risk_class='high-risk'):
        scan = Scan(prediction=prediction, recommendation='<p>test</p>',
                    risk_class=risk_class, user_id=user_id)
        db.session.add(scan)
        db.session.commit()
        return scan
    return _create


def make_user(username='bob', email='bob@example.com', password='secret123'):
    user = User(username=username, email=email,
                password_hash=backend.generate_password_hash(password,
                                                             method='pbkdf2:sha256'))
    db.session.add(user)
    db.session.commit()
    return user
