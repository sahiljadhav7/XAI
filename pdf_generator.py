from flask import Flask, render_template, request, send_file
import os
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Rect

app = Flask(__name__)

# Function to convert HTML force plot to PNG
def convert_html_to_png(html_path, output_png):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=800x600")  

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.get(f"file://{os.path.abspath(html_path)}")
    
    time.sleep(2)  # Wait for rendering
    driver.save_screenshot(output_png)
    driver.quit()

    img = Image.open(output_png)
    img = img.crop(img.getbbox())  # Crop white space
    img.save(output_png)

# Function to draw a progress bar
def draw_progress_bar(confidence, width=400, height=15):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor="lightgrey"))
    d.add(Rect(0, 0, width * confidence, height, fillColor="black"))
    return d

# Function to generate the PDF report
def generate_pdf(patient_name, confidence, shap_html):
    output_pdf = "static/reports/ckd_report.pdf"
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>Chronic Kidney Disease Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    # Prediction Confidence
    elements.append(Paragraph(f"<b>Prediction Confidence:</b> {confidence:.1f}%", styles["Normal"]))
    elements.append(draw_progress_bar(confidence / 100, width=400, height=15))
    elements.append(Spacer(1, 10))
    
    # Convert SHAP force plot to PNG
    shap_plot_path = "static/images/force_plot.png"
    convert_html_to_png(shap_html, shap_plot_path)
    
    # Add SHAP Force Plot
    elements.append(Paragraph("<b>Detailed Factor Analysis</b>", styles["Heading2"]))
    elements.append(RLImage(shap_plot_path, width=400, height=200))

    # Build PDF
    doc.build(elements)
    return output_pdf