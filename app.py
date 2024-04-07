from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
from statistics import median, mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from reportlab.lib import colors
from werkzeug.utils import secure_filename
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import requests

# Create Flask app instance
app = Flask(__name__)

# Define static path for serving static files (e.g., images, CSS)
static_path = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_path):
    os.makedirs(static_path)

# Load data and models
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encode categorical target variable
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Split data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Initialize machine learning models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_knn_model = KNeighborsClassifier(n_neighbors=5)

# Train machine learning models
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_knn_model.fit(X, y)

# Extract symptom names
symptoms = X.columns.values

# Create a dictionary to map symptom names to their indices and prediction classes
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Function to predict disease based on symptoms
# Function to predict disease based on symptoms
def predict_disease(symptoms):
    input_data = [0] * len(data_dict["symptom_index"])

    for symptom in symptoms:
        standardized_symptom = " ".join([i.capitalize() for i in symptom.split("_")])
        if standardized_symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][standardized_symptom]
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    knn_prediction = data_dict["predictions_classes"][final_knn_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = median([knn_prediction, nb_prediction, svm_prediction])

    predictions = {
        "knn_model_prediction": knn_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

    return predictions

# Define route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle form submission and display result
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        user_input = request.form.getlist('symptoms') 
        print("Selected Symptoms:", user_input) 
        result = predict_disease(user_input)

        # Extract form data for passing to view_report page
        form_data = {
            'title': request.form.get('title', ''),
            'first_name': request.form.get('first_name', ''),
            'last_name': request.form.get('last_name', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'symptoms': user_input,
        }

        # Render the result page with the result and a link to the view_report page
        return render_template('result.html', result=result, **form_data)

    # If the request method is not POST, render an error message
    return render_template('result.html', result="Invalid request method.")

# Function to generate medical report
def generate_medical_report(patient_details, predictions):
    title = patient_details.get('title', '')
    full_name = f"{title} {patient_details['first_name']} {patient_details['last_name']}\n\n"
    
    report_content = f"Full Name: {full_name}"
    report_content += f"Age: {patient_details['age']}\n\n"
    report_content += f"Gender: {patient_details['gender']}\n\n"
    
    # Add symptoms to the report
    symptoms = patient_details.get('symptoms', '')
    
    report_content += f"Symptoms: {symptoms}\n\n"

    # Add a line break before the "Prediction Details" section
    report_content += "\nPrediction Details:\n"
    report_content += f" \n"
    report_content += f"KNN Model Prediction: {predictions['knn_model_prediction']}\n\n"
    report_content += f"Naive Bayes Prediction: {predictions['naive_bayes_prediction']}\n\n"
    report_content += f"SVM Model Prediction: {predictions['svm_model_prediction']}\n\n\n"
    report_content += f"Final Prediction: {predictions['final_prediction']}\n"

    # For debugging, print the report content
    print("Generated Medical Report Content:")
    print(report_content)

    return report_content

# Define route to handle form submission and generate medical report
# Define route to handle form submission and generate medical report
@app.route('/view_report', methods=['POST'])
def view_report():
    if request.method == 'POST':
        # Use the form data to generate the medical report
        title = request.form['title']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        gender = request.form['gender']
        symptoms = request.form['symptoms']

        # Perform health prediction or other processing here
        predictions = predict_disease(symptoms)

        report_content = generate_medical_report({
            'title': title,
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'symptoms': symptoms,
        }, predictions)
        hospital_logo_path = 'static/images/Hospital.png'  # Provide the actual path to the hospital logo
        hospital_details = {
            'name': 'Sample Hospital',
            'address': '123 Medical Street, City, Country',
            'phone': '123-456-7890',
            'email': 'info@samplehospital.com'
    }

        # Create a temporary PDF file
        temp_pdf_path = os.path.join(static_path, 'temp_report.pdf')
        create_pdf(temp_pdf_path, report_content, hospital_logo_path, hospital_details)

        # Render the view_report page with the medical report content and the PDF file path
        return render_template('view_report.html', report_content=report_content, pdf_path=temp_pdf_path,
                               first_name=first_name, last_name=last_name)

    # If the request method is not POST, render an error message
    return render_template('view_report.html', report_content="Invalid request method.")


# Define route to handle downloading the PDF file
# Define route to handle downloading the PDF file
@app.route('/download_report', methods=['POST'])
def download_report():
    if request.method == 'POST':
        # Get the PDF file path and patient details from the form data
        pdf_path = request.form['pdf_path']
        first_name = request.form.get('first_name', '')
        last_name = request.form.get('last_name', '')

        # Customize the file name with the patient's first name and last name
        file_name = f"{first_name}_{last_name}_Medical_Report.pdf"

        # Serve the PDF file for download with the customized file name
        response = send_file(pdf_path, as_attachment=True)

        # Set the Content-Disposition header to specify the filename
        response.headers['Content-Disposition'] = f'attachment; filename="{file_name}"'

        return response

    # If the request method is not POST, render an error message
    return "Invalid request method."



# Function to create a PDF file
def create_pdf(file_path, report_content, hospital_logo_path, hospital_details):
    # Use reportlab to create a PDF file
    buffer = BytesIO()  # Use a BytesIO buffer to store binary data
    c = canvas.Canvas(buffer)

    # Set font and size
    c.setFont("Helvetica", 12)

    # Add hospital details on the left end
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, hospital_details['name'])
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, hospital_details['address'])
    c.drawString(50, 765, f"Phone: {hospital_details['phone']}")
    c.drawString(50, 750, f"Email: {hospital_details['email']}")

    # Add hospital logo on the right end
    logo = ImageReader(hospital_logo_path)
    c.drawImage(logo, 450, 750, width=100, height=100)

    # Add a line separator
    c.setStrokeColor(colors.black)
    c.line(50, 740, 550, 740)

    # Add space below the line
    line_spacing_below = 20  # Adjust this value based on your preference
    c.drawString(50, 740 - line_spacing_below, "")  # Empty string for spacing

    # Write the report content to the PDF
    lines = report_content.split('\n')
    y_position = 730 - line_spacing_below  # Starting y position for report content
    for line in lines:
        c.drawString(50, y_position, line)
        y_position -= 15  # Adjust this value based on your preference for line spacing

    # Save the PDF to the buffer
    c.save()

    # Write the buffer content to the file in binary mode
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(buffer.getvalue())

    return file_path

# Define route to download the medical report as a PDF
# Function to generate confusion matrix and route to display report
@app.route('/report')
def report():
    # Assuming y_test and preds are defined somewhere in your code
    #KNN predictions
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds_knn = nb_model.predict(X_test)
    #KNN predictions
    knn_model = KNeighborsClassifier(n_neighbors=5)  
    knn_model.fit(X_train, y_train)
    preds_nb = knn_model.predict(X_test)
    #SVM predictions
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds_svm = svm_model.predict(X_test) 
    #combined model
    final_preds = [mode([i, j, k]) for i, j, k in zip(preds_knn, preds_nb, preds_svm)]
    # Generate confusion matrices
    generate_confusion_matrix(y_test, preds_knn, 'KNN')
    generate_confusion_matrix(y_test, preds_nb, 'NB')
    generate_confusion_matrix(y_test, preds_svm, 'SVM')
    generate_confusion_matrix(y_test, final_preds, 'Combined_Model')
    return render_template('report.html')

# Function to generate confusion matrix plot
def generate_confusion_matrix(y_true, y_pred, classifier_name):
    cf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig(f'static/confusion_matrix_{classifier_name.lower()}.png')  # Save the plot as an image
    plt.close()

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
