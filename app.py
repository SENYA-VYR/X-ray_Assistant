import os, json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from db import db, Doctor, Patient, Study, TrainingData
from model import load_crack_model, analyze_xray, train_model_from_feedback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ortho.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

MODEL = load_crack_model(weights_path='crack_model.pth' if os.path.exists('crack_model.pth') else None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@login_manager.user_loader
def load_user(user_id):
    return Doctor.query.get(int(user_id))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        institution = request.form['institution']

        if Doctor.query.filter_by(username=username).first():
            flash('Врач с таким логином уже существует')
            return redirect(url_for('register'))

        hashed = generate_password_hash(password)
        doctor = Doctor(
            username=username,
            password_hash=hashed,
            full_name=full_name,
            institution=institution
        )
        db.session.add(doctor)
        db.session.commit()
        flash('Регистрация прошла успешно, войдите')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        doctor = Doctor.query.filter_by(username=username).first()
        if doctor and check_password_hash(doctor.password_hash, password):
            login_user(doctor)
            return redirect(url_for('dashboard'))
        flash('Неверный логин или пароль')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    studies = Study.query.filter_by(doctor_id=current_user.id).order_by(Study.study_date.desc()).all()
    return render_template('dashboard.html', studies=studies)

@app.route('/new_study', methods=['GET', 'POST'])
@login_required
def new_study():
    patients = Patient.query.all()
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')

        if patient_id:
            patient_id = int(patient_id)
            patient = Patient.query.get(patient_id)
            if not patient:
                flash('Пациент не найден')
                return redirect(url_for('new_study'))
        else:
            full_name = request.form.get('new_patient_name', '').strip()
            birth_date = request.form.get('new_patient_birth', '').strip()

            if not full_name or not birth_date:
                flash('Заполните ФИО и дату рождения нового пациента')
                return redirect(url_for('new_study'))

            existing_patient = Patient.query.filter_by(
                full_name=full_name,
                birth_date=datetime.strptime(birth_date, '%Y-%m-%d').date()
            ).first()

            if existing_patient:
                patient_id = existing_patient.id
                flash(f'Пациент {full_name} уже существует в базе. Использована существующая запись.')
            else:
                patient = Patient(
                    full_name=full_name,
                    birth_date=datetime.strptime(birth_date, '%Y-%m-%d').date()
                )
                db.session.add(patient)
                db.session.commit()
                patient_id = patient.id

        file = request.files.get('xray_image')
        if not file or not allowed_file(file.filename):
            flash('Выберите изображение в формате PNG или JPG')
            return redirect(url_for('new_study'))
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result_img, findings = analyze_xray(filepath)
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            result_img.save(result_path)
        except Exception as e:
            flash(f'Ошибка анализа: {str(e)}')
            return redirect(url_for('new_study'))

        study = Study(
            doctor_id=current_user.id,
            patient_id=patient_id,
            study_date=datetime.utcnow().date(),
            original_image=filename,
            result_image=result_filename,
            findings=json.dumps(findings),
            feedback_given=False
        )
        db.session.add(study)
        db.session.commit()
        return redirect(url_for('result', study_id=study.id))

    return render_template('new_study.html', patients=patients)

@app.route('/result/<int:study_id>')
@login_required
def result(study_id):
    study = Study.query.get_or_404(study_id)
    if study.doctor_id != current_user.id:
        flash('Доступ запрещён')
        return redirect(url_for('dashboard'))
    findings = json.loads(study.findings) if study.findings else []
    return render_template('result.html', study=study, findings=findings)

@app.route('/feedback/<int:study_id>', methods=['GET', 'POST'])
@login_required
def feedback(study_id):
    study = Study.query.get_or_404(study_id)
    if study.doctor_id != current_user.id:
        flash('Доступ запрещён')
        return redirect(url_for('dashboard'))

    findings = json.loads(study.findings) if study.findings else []

    if request.method == 'POST':
        feedback_data = []

        for i, finding in enumerate(findings):
            zone_confirmed = request.form.get(f'zone_{i}') == 'yes'
            feedback_data.append({
                'zone_index': i,
                'bbox': finding['bbox'],
                'ai_probability': finding['mean_probability'],
                'doctor_confirmed': zone_confirmed
            })

            training_entry = TrainingData(
                study_id=study.id,
                image_path=study.original_image,
                bbox_x=finding['bbox'][0],
                bbox_y=finding['bbox'][1],
                bbox_w=finding['bbox'][2],
                bbox_h=finding['bbox'][3],
                is_crack=zone_confirmed
            )
            db.session.add(training_entry)

        study.doctor_feedback = json.dumps({
            'zones_feedback': feedback_data
        })
        study.feedback_given = True
        db.session.commit()

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], study.original_image)
        try:
            loss = train_model_from_feedback(image_path, feedback_data)

            total_zones = len(feedback_data)
            confirmed_cracks = sum(1 for f in feedback_data if f['doctor_confirmed'])
            false_positives = total_zones - confirmed_cracks

            if loss is not None:
                flash(f'Нейросеть обучена! Потеря: {loss:.4f}. '
                      f'Подтверждено трещин: {confirmed_cracks}, '
                      f'ложных срабатываний: {false_positives}.')
            else:
                flash(f'Данные сохранены. Подтверждено трещин: {confirmed_cracks}, '
                      f'ложных срабатываний: {false_positives}.')
        except Exception as e:
            flash(f'Данные сохранены, но обучение не выполнено: {str(e)}')

        return redirect(url_for('dashboard'))

    return render_template('feedback.html', study=study, findings=findings)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

with app.app_context():
    db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)