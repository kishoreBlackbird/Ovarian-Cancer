from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import random

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('Wbccls.h5', compile=False)

# Define the disease class
disease_class = ['Cancer Detected', 'Healthy Cells']

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define sample data lists
medical_history_samples = ["No significant medical history.", "Family history of cancer.", "Previous treatment for another condition.", "Chronic illness.", "History of smoking."]
symptoms_positive_samples = ["Persistent cough.", "Unexplained weight loss.", "Changes in bowel or bladder habits.", "Fatigue.", "Difficulty swallowing."]
symptoms_negative_samples = ["No symptoms reported.", "General well-being.", "Normal appetite.", "Healthy lifestyle.", "Regular exercise."]
blood_tests_samples = ["Complete blood count (CBC).", "Liver function tests (LFTs).", "Kidney function tests.", "Tumor marker tests.", "Coagulation tests."]
imaging_tests_samples = ["X-ray.", "Computed tomography (CT) scan.", "Magnetic resonance imaging (MRI).", "Ultrasound.", "Positron emission tomography (PET) scan."]
genetic_testing_samples = ["BRCA1 gene mutation test.", "BRCA2 gene mutation test.", "HER2 gene testing.", "EGFR mutation testing.", "BRAF gene mutation testing."]
tumor_size_location_samples = ["3 cm tumor located in the left breast.", "5 cm tumor in the colon.", "2 cm mass in the lung.", "Tumor detected in the liver.", "Localized tumor in the prostate."]
staging_samples = ["Stage I - localized cancer.", "Stage II - locally advanced cancer.", "Stage III - regional spread.", "Stage IV - metastatic cancer.", "Unknown stage."]
response_to_treatment_samples = ["Complete remission.", "Partial response.", "Stable disease.", "Disease progression.", "No response to treatment."]
follow_up_testing_samples = ["Regular CT scans.", "MRI scans every six months.", "Blood tests every three months.", "Colonoscopy every year.", "PET scans as needed."]

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds[0] if preds is not None else None

def calculate_percentages(preds):
    percentages = {
        'Cancer': round(preds[0] * 100, 2),  # Round off to two decimal places
        'Healthy': round(preds[1] * 100, 2)   # Round off to two decimal places
    }
    return percentages

def predict_cancer_stage(percentages):
    if percentages['Cancer'] > 50:
        return 'Advanced Stage'
    else:
        return 'Early Stage'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from the request
        f = request.files['file']
        if f:
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)
            if preds is not None:
                percentages = calculate_percentages(preds)
                result_index = np.argmax(preds)
                result = disease_class[result_index]

                # Generate random data based on the predicted result
                if result == 'Healthy Cells':
                    symptoms = random.choice(symptoms_positive_samples)
                else:
                    symptoms = random.choice(symptoms_negative_samples)

                medical_history = random.choice(medical_history_samples)
                blood_tests = random.choice(blood_tests_samples)
                imaging_tests = random.choice(imaging_tests_samples)
                genetic_info = random.choice(genetic_testing_samples)
                tumor_info = random.choice(tumor_size_location_samples)
                staging = random.choice(staging_samples)
                response_to_treatment = random.choice(response_to_treatment_samples)
                follow_up_testing = random.choice(follow_up_testing_samples)

                # Predict cancer stage
                cancer_stage = predict_cancer_stage(percentages)

                # Redirect to the result page
                return redirect(url_for('result', result=result, percentages=percentages, cancer_stage=cancer_stage,
                                        medical_history=medical_history, symptoms=symptoms, blood_tests=blood_tests,
                                        imaging_tests=imaging_tests, genetic_info=genetic_info, tumor_info=tumor_info,
                                        staging=staging, response_to_treatment=response_to_treatment,
                                        follow_up_testing=follow_up_testing))

    return render_template('index.html')



@app.route('/result')
def result():
    result = request.args.get('result')
    percentages = eval(request.args.get('percentages'))
    cancer_stage = request.args.get('cancer_stage')
    medical_history = request.args.get('medical_history')
    symptoms = request.args.get('symptoms')
    blood_tests = request.args.get('blood_tests')
    imaging_tests = request.args.get('imaging_tests')
    genetic_info = request.args.get('genetic_info')
    tumor_info = request.args.get('tumor_info')
    staging = request.args.get('staging')
    response_to_treatment = request.args.get('response_to_treatment')
    follow_up_testing = request.args.get('follow_up_testing')
    return render_template('result.html', result=result, percentages=percentages, cancer_stage=cancer_stage,
                           medical_history=medical_history, symptoms=symptoms, blood_tests=blood_tests,
                           imaging_tests=imaging_tests, genetic_info=genetic_info, tumor_info=tumor_info,
                           staging=staging, response_to_treatment=response_to_treatment,
                           follow_up_testing=follow_up_testing)

if __name__ == '__main__':
    app.run(debug=True)
