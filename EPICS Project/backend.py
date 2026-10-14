import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
from tensorflow.keras import layers,models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect, url_for, flash
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
app=Flask(__name__)

app.config['SECRET_KEY']=os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(100), nullable=False)
    recommendation = db.Column(db.Text, nullable=False)
    risk_class = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method=='POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            return "Email already registered. Please login."

        hashed_pw = generate_password_hash(password,method='pbkdf2:sha256')
        new_user = User(username=username,email=email,password_hash=hashed_pw)
        
        try:
            db.session.add(new_user)
            db.session.commit()

            login_user(new_user)

            return redirect(url_for('index'))
            
        except Exception as e:
            db.session.rollback()
            return f"An error occurred: {str(e)}"

    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('landing'))

def load_my_model():
    base=EfficientNetB3(weights='imagenet',include_top=False,input_shape=(224,224,3))
    base.trainable=False

    model=models.Sequential([
        layers.Input((224,224,3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.4),
        layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('imagenet_model.h5')
    return model
model=load_my_model()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def index():
    return render_template('index.html')

@app.route('/analyzer',methods=['GET','POST'])
@login_required
def predict():
    prediction=None
    recommendation=" "
    risk_class=" "
    if request.method=='POST':
        if 'imagefile' not in request.files:
            return "No file uploaded"
        file=request.files['imagefile']
        if file.filename=='':
            return "No file selected"
        
        filename=secure_filename(file.filename)
        filepath=os.path.join("./images",filename)
        if not os.path.exists("./images"): os.makedirs("./images")
        file.save(filepath)

        img=load_img(filepath,target_size=(224,224))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=model.predict(x)[0][0]
        if pred<0.30:
            label="LOW PROBABILITY"
            risk_class="low-risk"
            recommendation="""<p><strong>Immediate Action:</strong> Continue with your current wellness routine. No immediate clinical intervention is suggested based on this scan.</p>
                           <p><strong>Diagnostic Steps:</strong> Maintain your schedule for annual screening mammograms as recommended by your doctor. Annual screenings are the best way to catch changes over time.</p>
                           """
        elif 0.30<=pred<0.50:
            label="MODERATE CONCERN"
            risk_class="mod-risk"
            recommendation="""<p><strong>Immediate Action:</strong> Schedule a consultation with your primary physician or radiologist within the next 2 weeks to discuss these findings.</p>
                           <p><strong>Diagnostic Steps:</strong> Your doctor may request 'Diagnostic Imaging,' which is more detailed than a screening. This typically includes a targetted Breast Ultrasound or a Diagnostic Mammogram (magnification views) to clarify the area of concern.</p>
                           """
        else:
            label="HIGH PROBABILITY"
            risk_class="high-risk"
            recommendation="""<p><strong>Immediate Action:</strong> Contact a Breast Specialist or an Oncology clinic as soon as possible for a priority consultation. This result indicates a high clinical suspicion that requires investigation.</p>
                           <p><strong>Diagnostic Steps:</strong> Be prepared for a 'Biopsy' recommendation. A core needle biopsy is the standard procedure to get a definitive diagnosis. Your specialist may also order a Breast MRI to evaluate the extent of the tissue involvement.</p>
                           """
        
        prediction=f"{label} ({pred*100:.2f}%)"
        new_scan = Scan(
            prediction=prediction,
            recommendation=recommendation,
            risk_class=risk_class,
            user_id=current_user.id 
        )
        db.session.add(new_scan)
        db.session.commit()
        return render_template('result.html',prediction=prediction,recommendation=recommendation,risk_class=risk_class)
    return render_template('analyzer.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/report/<int:scan_id>') 
@login_required
def view_report(scan_id):

    scan = Scan.query.get_or_404(scan_id)

    if scan.user_id != current_user.id:
        return "Access Denied", 403

    return render_template('result.html', 
                           prediction=scan.prediction, 
                           recommendation=scan.recommendation, 
                           risk_class=scan.risk_class)

@app.route('/history')
@login_required
def history():
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.created_at.desc()).all()
    return render_template('history.html', scans=user_scans)

if __name__ == '__main__':
    app.run(port=8000,debug=True,use_reloader=False)
