# pdf_generator.py
# PDF generation via Selenium is currently disabled in production.
# Selenium requires a headless Chrome installation which is not available
# in the Render deployment environment.
# This feature will be re-enabled with a reportlab-only implementation.

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Rect


def draw_progress_bar(confidence, width=400, height=15):
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor="lightgrey"))
    d.add(Rect(0, 0, width * confidence, height, fillColor="black"))
    return d


def generate_pdf(patient_name, confidence, shap_html=None):
    """Generate a basic PDF report without Selenium."""
    output_pdf = "static/reports/ckd_report.pdf"
    import os
    os.makedirs("static/reports", exist_ok=True)

    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Chronic Kidney Disease Detection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Patient:</b> {patient_name}", styles["Normal"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"<b>Prediction Confidence:</b> {confidence:.1f}%", styles["Normal"]))
    elements.append(draw_progress_bar(confidence / 100, width=400, height=15))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "Note: Detailed SHAP force plot rendering requires a Chrome environment "
        "and is not available in the current deployment.",
        styles["Normal"]
    ))

    doc.build(elements)
    return output_pdf