from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from data_extraction import extract_ckd_data_from_image
from prediction import preprocess_input_data,predict_explain
from lung_disease import *
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import shap
import io
import matplotlib.pyplot as plt
import base64
from pdf_generator import * 
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODELS']  = 'static/models'


def format_missing_assets_message(title, assets):
    details = "; ".join([f"{label}: {path}" for label, path in assets])
    return f"{title}. Missing assets -> {details}"


def get_workflow_status():
    kidney_dataset_path = os.path.join(app.config['MODELS'], "chronic_kidney_disease", "data", "processed_kidney_disease.csv")
    kidney_model_path = os.path.join(app.config['MODELS'], "chronic_kidney_disease", "Random_Forest_model.pkl")
    lung_model_path = os.path.join(app.config['MODELS'], "lung_disease", "model.h5")
    gemini_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    kidney_missing = []
    if not gemini_ready:
        kidney_missing.append("GEMINI_API_KEY")
    if not os.path.exists(kidney_dataset_path):
        kidney_missing.append(kidney_dataset_path)
    if not os.path.exists(kidney_model_path):
        kidney_missing.append(kidney_model_path)

    lung_missing = []
    if not os.path.exists(lung_model_path):
        lung_missing.append(lung_model_path)

    return {
        "kidney": {
            "ready": len(kidney_missing) == 0,
            "missing": kidney_missing,
        },
        "lung": {
            "ready": len(lung_missing) == 0,
            "missing": lung_missing,
        }
    }


@app.context_processor
def inject_workflow_status():
    return {"workflow_status": get_workflow_status()}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/services')
def service():
    return render_template('services.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload_report', methods=['GET', 'POST'])
def upload_report():
    if request.method == 'POST':
        file = request.files.get('file')
        disease_type = request.form.get('disease_type', '').strip()
        workflow_status = get_workflow_status()

        if not disease_type:
            return render_template('upload_report.html', error="Please select an analysis type before uploading.", selected_disease="")

        if disease_type in workflow_status and not workflow_status[disease_type]["ready"]:
            missing_summary = ", ".join(workflow_status[disease_type]["missing"])
            return render_template(
                'upload_report.html',
                error=f"The {disease_type} workflow is not available yet. Missing: {missing_summary}",
                selected_disease=disease_type
            )

        if file and file.filename:
            filename = secure_filename(file.filename)
            if not filename:
                return render_template('upload_report.html', error="Please choose a valid file name.", selected_disease=disease_type)

            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], disease_type)
            os.makedirs(upload_dir, exist_ok=True)
            file.save(os.path.join(upload_dir, filename))
            # Redirect to the view_report page with both the disease type and filename
            return redirect(url_for('view_report', disease_type=disease_type, report_image=filename))
        else:
            return render_template('upload_report.html',error="Please upload a report image to continue.", selected_disease=disease_type)
    
    return render_template('upload_report.html',error="", selected_disease="")

@app.route('/view_report', methods=['GET', 'POST'])
def view_report():
    disease_type = request.args.get('disease_type', '')  # Get disease type from URL args
    report_image = request.args.get('report_image', '')  # Get report image filename from URL args
    report_preview_path = os.path.join('uploads', disease_type, report_image).replace("\\", "/") if disease_type and report_image else ''

    # Debugging: Print out the parameters
    print(f"disease_type: {disease_type}, report_image: {report_image}")

    # Check if report_image exists
    if report_image:
        # Get the full file path for the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],disease_type, report_image)

        # Normalize the file path to use forward slashes
        file_path = os.path.normpath(file_path)

        print(f"Normalized file path: {file_path}")  # Debugging file path

        if not os.path.exists(file_path):
            return render_template(
                'error.html',
                message=f"The uploaded file could not be found at {file_path}. Please upload the image again."
            )

        if(disease_type == "kidney"):
            try:
                # print(f"Calling extract_ckd_data_from_image with file: {file_path}")
                ckd_data = extract_ckd_data_from_image(file_path)  # Pass the correct file path here
                # print(f"CKD data extracted: {ckd_data}")  # See what data is returned
                
                if(ckd_data == "the image is not correct"):
                    return render_template('upload_report.html',error="Please provide correct Report!", selected_disease=disease_type)
                
            except Exception as e:
                return render_template('upload_report.html',error=f"Error extracting CKD data: {e}", selected_disease=disease_type)
                ckd_data = {}
            return render_template('view_report.html', disease_type=disease_type, 
                                report_image=report_image, report_preview_path=report_preview_path, ckd_data=ckd_data)
            
        elif (disease_type == "lung"):
            model_path = os.path.join(app.config['MODELS'],"lung_disease","model.h5")
            if not os.path.exists(model_path):
                return render_template(
                    'error.html',
                    message=format_missing_assets_message(
                        "Lung analysis is not ready on this machine",
                        [("model", model_path)]
                    )
                )


            from tensorflow.keras.layers import Dense
            class CustomDense(Dense):
                @classmethod
                def from_config(cls, config):
                    config.pop('quantization_config', None)
                    return super().from_config(config)

            model = load_model(model_path, custom_objects={'Dense': CustomDense}, compile=False)
            image_path = file_path
            predicted_class,predicted_probability, gradcam_image = predict_and_visualize(image_path,model,"conv_pw_13_relu")

            output_units = model.output_shape[-1] if isinstance(model.output_shape, tuple) else model.output_shape[0][-1]
            if output_units == 1:
                if predicted_class == 0:
                    predicted_class = "Normal"
                    text_explanation = "The X-ray appears clear with no significant opacities or structural abnormalities detected. The lung volume looks normal. If the patient is experiencing symptoms like a mild cough or fatigue, they may be arising from a non-pulmonary source or a very early-stage presentation not visible on standard radiography."
                else:
                    predicted_class = "Abnormality Detected"
                    text_explanation = "The X-ray shows regions that deviate from a normal pulmonary baseline. This could include cloudiness (opacities), irregular interstitial patterns, or possible fluid buildup. Patients with this finding may experience shortness of breath, a persistent or productive cough, or chest tightness. A clinical review is strongly recommended to identify the exact cause (e.g. infection, effusion, or other pathology)."
            else:
                if(predicted_class==0):
                    predicted_class = "Lung Opacity"
                    text_explanation = "The X-ray shows regions of cloudiness or 'opacity' which shouldn't normally be there. Patients with this finding often experience shortness of breath, a persistent, sometimes productive cough, and possible chest tightness. Opacities can point to bacterial infections, fluid buildup, or other localized cellular abnormalities requiring further medical review."
                elif(predicted_class==1):
                    predicted_class = "Normal"
                    text_explanation = "The X-ray appears clear with no significant opacities or viral patterns detected. The lung volume looks normal. If the patient is experiencing symptoms like a mild cough or fatigue, they may be arising from a non-pulmonary source or a very early-stage presentation not visible on standard radiography."
                else:
                    predicted_class = "Viral Pneumonia"
                    text_explanation = "The X-ray indicates diffuse, often bilateral interstitial patterns that are standard markers for viral infections (like influenza, RSV, or COVID-19). Patients actively dealing with this often experience fever, a dry and irritating cough, significant fatigue, muscle aches, and shortness of breath that worsens with exertion."
                
            # Convert gradcam_image to base64 for HTML rendering
            _, buffer = cv2.imencode('.jpg', gradcam_image)
            gradcam_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
            original_image = cv2.imread(image_path)
            _, buffer = cv2.imencode('.jpg', original_image)
            original_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('image_result.html', disease_type=disease_type, original_image=original_image_base64,
                                    gradcam_image=gradcam_image_base64, predicted_class=predicted_class,
                                    predicted_probability=predicted_probability, text_explanation=text_explanation) 
            
    return render_template('view_report.html', disease_type=disease_type, report_image=report_image, report_preview_path=report_preview_path, ckd_data={})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


import numpy as np
import pandas as pd
import shap


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    features = [
        "Age", "Blood Pressure (BP)", "Specific Gravity (SG)", "Albumin (AL)", 
        "Sugar (SU)", "Red Blood Cells (RBC)", "Pus Cells (PC)", "Pus Cell Clumps (PCC)",
        "Bacteria (BA)", "Blood Glucose Random (BGR)", "Blood Urea (BU)", "Serum Creatinine (SC)", 
        "Sodium (SOD)", "Potassium (POT)", "Hemoglobin (HEMO)", "Packed Cell Volume (PCV)", 
        "White Blood Cell Count (WC)", "Red Blood Cell Count (RC)", "Hypertension (HTN)", 
        "Diabetes Mellitus (DM)", "Coronary Artery Disease (CAD)", "Appetite (APPET)", 
        "Pedal Edema (PE)", "Anemia (ANE)"
    ]

    # Ensure the upload folder exis

    field_names = {
        'age': 'Age',
        'gender': 'Gender',
        'bp': 'Blood Pressure',
        'sg': 'Specific Gravity',
        'al': 'Albumin',
        'su': 'Sugar',
        'rbc': 'Red Blood Cells',
        'pc': 'Pus Cells',
        'pcc': 'Pus Cell Clumps',
        'ba': 'Bacteria',
        'bgr': 'Blood Glucose Random',
        'bu': 'Blood Urea',
        'sc': 'Serum Creatinine',
        'sod': 'Sodium',
        'pot': 'Potassium',
        'hemo': 'Hemoglobin',
        'pcv': 'Packed Cell Volume',
        'wc': 'White Blood Cell Count',
        'rc': 'Red Blood Cell Count',
        'htn': 'Hypertension',
        'dm': 'Diabetes Mellitus',
        'cad': 'Coronary Artery Disease',
        'appet': 'Appetite',
        'pe': 'Pedal Edema',
        'ane': 'Anemia',
        'heart_rate': 'Heart Rate',
        'respiratory_rate': 'Respiratory Rate',
        'temperature': 'Temperature'
    }
    disease_type = request.form.get('disease_type', '').strip()
    form_data = {key: request.form[key] for key in request.form if key != 'disease_type'}

    # Convert numerical fields
    for key, value in form_data.items():
        if value.replace('.', '', 1).isdigit():  # Check if it's a number
            form_data[key] = float(value)

    if disease_type == "kidney":
        data_path = "static/models/chronic_kidney_disease/data/processed_kidney_disease.csv"
        model_path = "static/models/chronic_kidney_disease/Random_Forest_model.pkl"
        force_plot_output = "static/images/force_plot.html"
        missing_assets = []

        for label, path in [
            ("processed dataset", data_path),
            ("random forest model", model_path),
        ]:
            if not os.path.exists(path):
                missing_assets.append((label, path))

        if missing_assets:
            return render_template(
                'error.html',
                message=format_missing_assets_message(
                    "Kidney prediction is not ready on this machine",
                    missing_assets
                )
            )

        scaled_instance = preprocess_input_data(form_data)

        # Check if preprocessing failed
        if scaled_instance is None:
            print("Error: Preprocessing failed!")
            return render_template('error.html', message="Invalid input data or missing preprocessing assets for kidney prediction.")

        # Read data correctly
        df = pd.read_csv(data_path)
        df.drop('classification', axis=1, inplace=True)  # Fixed `axis=1`

        # Load the model correctly
        import joblib
        rf_model = joblib.load(model_path)
        explainer = shap.TreeExplainer(rf_model)

        # Ensure `predict_explain` is available
        try:
            predicted_class, predicted_probs, shap_values_for_class, explainer = predict_explain(scaled_instance, rf_model, explainer, df.columns)
            print(predicted_class)
            print(predicted_probs)
            print(shap_values_for_class)
        except NameError:
            print("Error: `predict_explain` is not defined.")
            return None

        # Convert shap_values_for_class to list if necessary
        shap_values_for_class = shap_values_for_class.values.tolist() if hasattr(shap_values_for_class, 'values') else shap_values_for_class

        # Calculate the absolute values of SHAP values for the predicted class (for feature importance)
        feature_importance = np.abs(shap_values_for_class)

        # Sort and get top 7 important features
        top_7_importance_indices = feature_importance.argsort()[-7:][::-1]  # Get indices of the top 7 features
        top_7_importance_values = feature_importance[top_7_importance_indices]
        top_7_features = [df.columns[i] for i in top_7_importance_indices]

        # Ensure data integrity
        top_7_importance_values = top_7_importance_values.tolist()
        top_7_features = [str(feature) for feature in top_7_features]
        
        top_7_features_mapped = [field_names[key] for key in top_7_features] 
        
        # Handle single and multiple output cases correctly
        if isinstance(explainer.expected_value, np.ndarray):
            base_value = explainer.expected_value[0]  # Use first expected value if it's an array
        else:
            base_value = explainer.expected_value  # Use it directly if it's a scalar

        # Ensure scaled_instance is a DataFrame with feature names
        if isinstance(scaled_instance, np.ndarray):
            scaled_instance = pd.DataFrame(scaled_instance, columns=df.columns)

        # Generate force plot with feature names
        force_plot = shap.force_plot(base_value, shap_values_for_class, scaled_instance.iloc[0])

        # Save as HTML
        os.makedirs(os.path.dirname(force_plot_output), exist_ok=True)
        shap.save_html(force_plot_output, force_plot)

        # Pass the top 7 features and their importance to the result template
        feature_names = df.columns.tolist()
        
        result = "Chronic Kidney Disease" if predicted_probs >= 0.62 else "Not Chronic Kidney Disease"
        
        # Patient-friendly text explanation
        patient_explanation = []
        
        for feature in top_7_features:
            full_name = field_names.get(feature, feature)  # Get full name or fallback to the short name
            shap_value = shap_values_for_class[scaled_instance.columns.get_loc(feature)]
            impact_direction = "increased risk" if shap_value > 0 else "decreased risk"
            
            # Tailor explanation based on feature
            if feature == "age":
                explanation = f"<li>Your <b>{full_name}</b> had a significant impact, contributing to an {impact_direction} of Chronic Kidney Disease."
            elif feature == "bp":
                explanation = f"</li><li>Elevated <b>blood pressure</b> can strain your kidneys, contributing to disease progression. Your <b>{full_name}</b> influenced your risk by {impact_direction}."
            elif feature == "sg":
                explanation = f"</li><li><b>Specific gravity</b> measures the kidney’s ability to concentrate urine. Your <b>{full_name}</b> had an {impact_direction} of CKD."
            elif feature == "al":
                explanation = f"</li><li>The presence of <b>albumin</b> in urine indicates kidney damage. Your <b>{full_name}</b> increased or decreased your CKD risk."
            elif feature == "su":
                explanation = f"</li><li>High <b>sugar levels</b> in urine may indicate diabetes, a major risk factor for CKD. Your <b>{full_name}</b> contributed to an {impact_direction}."
            elif feature == "rbc":
                explanation = f"</li><li><b>Red blood cells</b> in urine may indicate kidney damage. Your <b>{full_name}</b> influenced the predicted risk."
            elif feature == "pc":
                explanation = f"</li><li><b>Pus cells</b> in the urine can signal infection or inflammation in the kidneys. Your <b>{full_name}</b> influenced your CKD risk."
            elif feature == "pcc":
                explanation = f"</li><li><b>Pus cell clumps</b> in urine are a sign of serious kidney issues. In your case, the <b>{full_name}</b> impacted the prediction by {impact_direction}."
            elif feature == "ba":
                explanation = f"The presence of <b>bacteria</b> in urine can suggest infection. Here, your <b>{full_name}</b> influenced the CKD risk."
            elif feature == "bgr":
                explanation = f"</li><li><b>High blood glucose levels</b>, associated with diabetes, can severely affect kidney function. Your <b>{full_name}</b> contributed to an {impact_direction} of CKD."
            elif feature == "bu":
                explanation = f"</li><li><b>Urea</b> is a waste product that kidneys usually filter. Elevated <b>{full_name}</b> suggests reduced kidney function, impacting your CKD risk."
            elif feature == "sc":
                explanation = f"</li><li><b>{full_name}</b> is a key marker of kidney health. Higher levels usually indicate decreased kidney function. Here, your <b>{full_name}</b> influenced the CKD prediction."
            elif feature == "sod":
                explanation = f"</li><li><b>{full_name}</b> levels are important for blood pressure regulation, which affects kidney health. Your <b>{full_name}</b> contributed to your predicted risk of CKD."
            elif feature == "pot":
                explanation = f"</li><li><b>{full_name}</b> balance is crucial for kidney function. Abnormal levels influenced your CKD risk."
            elif feature == "hemo":
                explanation = f"</li><li><b>{full_name}</b> reflects red blood cell count. Lower hemoglobin may suggest anemia, commonly linked with CKD. Your <b>{full_name}</b> impacted the prediction."
            elif feature == "pcv":
                explanation = f"</li><li>Low <b>{full_name}</b> may indicate anemia, a complication of CKD. Your <b>{full_name}</b> influenced the predicted risk."
            elif feature == "wc":
                explanation = f"</li><li>High <b>{full_name}</b> may indicate an infection affecting the kidneys. This contributed to an {impact_direction}."
            elif feature == "rc":
                explanation = f"</li><li>Abnormal <b>{full_name}</b> can indicate kidney disease. In your case, it influenced the predicted risk."
            elif feature == "htn":
                explanation = f"</li><li><b>High blood pressure</b> is a leading cause of CKD. Your <b>{full_name}</b> status had a significant impact on the risk prediction."
            elif feature == "dm":
                explanation = f"</li><li><b>Diabetes</b> is a major risk factor for CKD. Your <b>{full_name}</b> status influenced the predicted risk."
            elif feature == "cad":
                explanation = f"</li><li><b>Cardiovascular diseases</b> like <b>{full_name}</b> are often linked with CKD. This influenced your risk prediction."
            elif feature == "appet":
                explanation = f"</li><li>Loss of <b>appetite</b> can be a sign of kidney disease. Here, your <b>{full_name}</b> contributed to the prediction."
            elif feature == "pe":
                explanation = f"</li><li><b>Swelling</b> in the legs (<b>pedal edema<b>) is often associated with kidney disease. Your <b>{full_name}</b> status impacted the prediction."
            elif feature == "ane":
                explanation = f"</li><li><b>Anemia</b> is common in CKD patients. Your <b>{full_name}</b> status influenced the predicted risk."
            
            patient_explanation.append(explanation)
            
            
        # Combine explanation into a single paragraph with bold attributes
        text_explanation = f"The prediction is <b>{result}</b>. This prediction is influenced by the following attributes: {', '.join([f'<b>{field_names[feat]}</b>' for feat in top_7_features])}." + "\n ".join(patient_explanation)
        return render_template('result.html', disease=predicted_class, confidence=predicted_probs, top_7_features=top_7_features_mapped, top_7_importance_values=top_7_importance_values, shap_values=shap_values_for_class, feature_names=feature_names,text_explanation=text_explanation)

    # Handle other disease types if needed (heart_disease, etc.)
    else:
        print("Unknown Disease Type")
        return render_template('error.html', message="Unknown disease type")
    
    
@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    pass
    
if __name__ == '__main__':
    app.run(debug=True)
