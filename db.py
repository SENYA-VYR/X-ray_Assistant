from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class Doctor(UserMixin, db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    institution = db.Column(db.String(200), nullable=False)  # Медучреждение
    studies = db.relationship('Study', backref='doctor', lazy=True)

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)
    studies = db.relationship('Study', backref='patient', lazy=True)

class Study(db.Model):
    __tablename__ = 'studies'
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    study_date = db.Column(db.Date, nullable=False, default=datetime.utcnow)
    original_image = db.Column(db.String(256), nullable=False)
    result_image = db.Column(db.String(256), nullable=True)
    findings = db.Column(db.Text, nullable=True)
    doctor_feedback = db.Column(db.Text, nullable=True)
    feedback_given = db.Column(db.Boolean, default=False)

class TrainingData(db.Model):
    __tablename__ = 'training_data'
    id = db.Column(db.Integer, primary_key=True)
    study_id = db.Column(db.Integer, db.ForeignKey('studies.id'), nullable=False)
    image_path = db.Column(db.String(256), nullable=False)
    bbox_x = db.Column(db.Integer, nullable=False)
    bbox_y = db.Column(db.Integer, nullable=False)
    bbox_w = db.Column(db.Integer, nullable=False)
    bbox_h = db.Column(db.Integer, nullable=False)
    is_crack = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.Date, default=datetime.utcnow)