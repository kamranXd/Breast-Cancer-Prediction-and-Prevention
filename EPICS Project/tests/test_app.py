"""End-to-end tests for the Flask routes.

The Keras model is replaced with a stub, so these run without the 44 MB weights
file and without loading TensorFlow's model graph.
"""
import io
import os

import pytest

from backend import Scan, User, classify_probability
from conftest import get_csrf, make_user, png_bytes, prime_csrf, register


# --------------------------------------------------------------------------
# Public pages and auth enforcement
# --------------------------------------------------------------------------
@pytest.mark.parametrize('path', ['/', '/login', '/signup', '/prevention', '/symptoms'])
def test_public_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200


@pytest.mark.parametrize('path', ['/dashboard', '/history', '/analyzer'])
def test_protected_pages_redirect_to_login(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert '/login' in response.headers['Location']


def test_unknown_route_renders_error_page(client):
    response = client.get('/definitely-not-a-page')
    assert response.status_code == 404
    assert b'404' in response.data


def test_nav_uses_active_state(client):
    response = client.get('/prevention')
    assert b'class="active"' in response.data


def test_mobile_nav_markup_present(client):
    response = client.get('/prevention')
    assert b'nav-toggle' in response.data
    assert b'id="nav-links"' in response.data


# --------------------------------------------------------------------------
# Signup / login / logout
# --------------------------------------------------------------------------
def test_signup_creates_user_and_logs_in(client, app):
    response = register(client)
    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard')

    with app.app_context():
        assert User.query.count() == 1
        assert User.query.first().email == 'alice@example.com'
        assert User.query.first().password_hash != 'secret123'

    # The new session is authenticated.
    assert client.get('/dashboard').status_code == 200


def test_signup_rejects_duplicate_email(client):
    assert register(client).status_code == 302
    client.get('/logout')
    response = register(client, username='alice2', email='alice@example.com')
    assert response.status_code == 409
    assert b'Email already registered' in response.data


def test_signup_rejects_duplicate_username(client):
    assert register(client).status_code == 302
    client.get('/logout')
    response = register(client, username='alice', email='other@example.com')
    assert response.status_code == 409
    assert b'Username already taken' in response.data


def test_signup_rejects_short_password(client):
    response = register(client, password='short')
    assert response.status_code == 400
    assert b'at least 8 characters' in response.data


def test_signup_rejects_missing_fields(client):
    token = prime_csrf(client, '/signup')
    response = client.post('/signup', data={'username': '', 'email': '',
                                            'password': '', 'csrf_token': token})
    assert response.status_code == 400


def test_login_success_and_failure(client, app):
    register(client)
    client.get('/logout')

    token = prime_csrf(client, '/login')
    bad = client.post('/login', data={'email': 'alice@example.com',
                                      'password': 'wrong-password',
                                      'csrf_token': token})
    assert bad.status_code == 200
    assert b'Invalid email or password' in bad.data

    good = client.post('/login', data={'email': 'ALICE@example.com',
                                       'password': 'secret123',
                                       'csrf_token': token})
    assert good.status_code == 302
    assert good.headers['Location'].endswith('/dashboard')


def test_logout_clears_session(client):
    register(client)
    assert client.get('/dashboard').status_code == 200
    client.get('/logout')
    response = client.get('/dashboard')
    assert response.status_code == 302
    assert '/login' in response.headers['Location']


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------
def test_post_without_csrf_token_is_rejected(client):
    response = client.post('/login', data={'email': 'a@b.com', 'password': 'x'})
    assert response.status_code == 400
    assert b'CSRF' in response.data


def test_post_with_wrong_csrf_token_is_rejected(client):
    prime_csrf(client, '/login')
    response = client.post('/login', data={'email': 'a@b.com', 'password': 'x',
                                           'csrf_token': 'not-the-real-token'})
    assert response.status_code == 400


def test_forms_embed_csrf_token(client):
    assert b'name="csrf_token"' in client.get('/login').data
    assert b'name="csrf_token"' in client.get('/signup').data


# --------------------------------------------------------------------------
# Risk tier mapping (pure function, no model needed)
# --------------------------------------------------------------------------
@pytest.mark.parametrize('score,expected', [
    (0.0, ('LOW PROBABILITY', 'low-risk')),
    (0.2999, ('LOW PROBABILITY', 'low-risk')),
    (0.30, ('MODERATE CONCERN', 'mod-risk')),
    (0.4999, ('MODERATE CONCERN', 'mod-risk')),
    (0.50, ('HIGH PROBABILITY', 'high-risk')),
    (1.0, ('HIGH PROBABILITY', 'high-risk')),
])
def test_classify_probability_tiers(score, expected):
    assert classify_probability(score) == expected


# --------------------------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------------------------
@pytest.mark.parametrize('score,label,risk_class', [
    (0.10, 'LOW PROBABILITY', 'low-risk'),
    (0.40, 'MODERATE CONCERN', 'mod-risk'),
    (0.90, 'HIGH PROBABILITY', 'high-risk'),
])
def test_prediction_persists_scan_and_renders_result(client, app, set_model,
                                                     score, label, risk_class):
    register(client)
    set_model(score)
    token = prime_csrf(client, '/analyzer')

    response = client.post('/analyzer',
                           data={'imagefile': (png_bytes(), 'scan.png'),
                                 'csrf_token': token},
                           content_type='multipart/form-data')

    assert response.status_code == 200
    body = response.data.decode()
    assert label in body
    assert risk_class in body
    assert 'Model Prediction' in body

    with app.app_context():
        scans = Scan.query.all()
    assert len(scans) == 1
    assert scans[0].risk_class == risk_class
    assert scans[0].prediction.startswith(label)


def test_prediction_stores_no_uploaded_file(client, app, set_model, tmp_path):
    register(client)
    set_model(0.9)
    token = prime_csrf(client, '/analyzer')
    client.post('/analyzer',
                data={'imagefile': (png_bytes(), 'scan.png'), 'csrf_token': token},
                content_type='multipart/form-data')

    upload_dir = app.config['UPLOAD_FOLDER']
    leftovers = os.listdir(upload_dir) if os.path.isdir(upload_dir) else []
    assert leftovers == []


def test_analyzer_rejects_missing_file(client, set_model):
    register(client)
    set_model(0.9)
    token = prime_csrf(client, '/analyzer')
    response = client.post('/analyzer', data={'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'No file uploaded' in response.data


def test_analyzer_rejects_empty_filename(client, set_model):
    register(client)
    set_model(0.9)
    token = prime_csrf(client, '/analyzer')
    response = client.post('/analyzer',
                           data={'imagefile': (io.BytesIO(b''), ''),
                                 'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 400


def test_analyzer_rejects_unsupported_extension(client, set_model):
    register(client)
    set_model(0.9)
    token = prime_csrf(client, '/analyzer')
    response = client.post('/analyzer',
                           data={'imagefile': (io.BytesIO(b'not an image'),
                                               'notes.txt'),
                                 'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Unsupported file type' in response.data


def test_analyzer_rejects_oversized_upload(client, set_model):
    register(client)
    set_model(0.9)
    token = prime_csrf(client, '/analyzer')
    oversized = io.BytesIO(b'\x00' * (app_max_bytes() + 1024))
    response = client.post('/analyzer',
                           data={'imagefile': (oversized, 'big.png'),
                                 'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 413
    assert b'too large' in response.data


def app_max_bytes():
    from backend import app
    return app.config['MAX_CONTENT_LENGTH']


def test_analyzer_post_requires_login(client, set_model):
    """A valid CSRF token does not bypass login; the POST redirects to /login."""
    set_model(0.9)
    token = prime_csrf(client, '/login')
    response = client.post('/analyzer',
                           data={'imagefile': (png_bytes(), 'scan.png'),
                                 'csrf_token': token},
                           content_type='multipart/form-data')
    assert response.status_code == 302
    assert '/login' in response.headers['Location']


def test_anonymous_post_without_csrf_fails_before_auth(client, set_model):
    """CSRF is checked first, so a token-less POST never reaches the view."""
    set_model(0.9)
    response = client.post('/analyzer', data={},
                           content_type='multipart/form-data')
    assert response.status_code == 400


# --------------------------------------------------------------------------
# History and report access control
# --------------------------------------------------------------------------
def test_history_lists_only_own_scans(client, app, scan_for):
    register(client)
    with app.app_context():
        scan_for(user_id=1, prediction='HIGH PROBABILITY (91.00%)',
                 risk_class='high-risk')
        make_user(username='bob')
        scan_for(user_id=2, prediction='LOW PROBABILITY (5.00%)',
                 risk_class='low-risk')

    response = client.get('/history')
    body = response.data.decode()
    assert response.status_code == 200
    assert 'HIGH PROBABILITY (91.00%)' in body
    assert 'LOW PROBABILITY' not in body


def test_report_owner_can_view(client, app, scan_for):
    register(client)
    with app.app_context():
        scan = scan_for(user_id=1)
        scan_id = scan.id

    response = client.get(f'/report/{scan_id}')
    assert response.status_code == 200
    assert b'Model Prediction' in response.data


def test_report_of_another_user_is_forbidden(client, app, scan_for):
    register(client)
    client.get('/logout')
    with app.app_context():
        register_id = 1
        scan = scan_for(user_id=register_id)
        scan_id = scan.id
        make_user(username='mallory', email='mallory@example.com')

    token = prime_csrf(client, '/login')
    client.post('/login', data={'email': 'mallory@example.com',
                                'password': 'secret123',
                                'csrf_token': token})
    response = client.get(f'/report/{scan_id}')
    assert response.status_code == 403
    assert b'access' in response.data.lower()


def test_missing_report_returns_404(client):
    register(client)
    assert client.get('/report/9999').status_code == 404


# --------------------------------------------------------------------------
# Model metrics surfaced on the result page
# --------------------------------------------------------------------------
def test_result_page_shows_model_metrics(client, app, scan_for):
    from backend import load_model_metrics
    register(client)
    with app.app_context():
        scan = scan_for(user_id=1)
        scan_id = scan.id
        assert load_model_metrics() is not None

    body = client.get(f'/report/{scan_id}').data.decode()
    assert 'Model performance' in body
    assert 'Cancer recall' in body


def test_model_path_defaults_to_h5_file():
    import backend
    assert backend.MODEL_PATH.endswith('.h5')


def test_get_model_raises_when_weights_missing(monkeypatch):
    import backend
    monkeypatch.setattr(backend, '_MODEL', None)
    monkeypatch.setattr(backend, 'MODEL_PATH', '/nonexistent/model.h5')
    with pytest.raises(FileNotFoundError):
        backend.get_model()
