from PIL import Image
import google.genai as genai
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

def clean_value(value):
    """
    Cleans extracted values by removing units and keeping only numbers or meaningful labels.

    Parameters:
        value (str | int | float): The extracted value.

    Returns:
        str | int | float: Cleaned value without units.
    """
    if isinstance(value, str):
        # Extract numeric values or keep categorical labels
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value)
        if len(numbers) > 1:  # Case like "120/80"
            return [float(num) if "." in num else int(num) for num in numbers]
        elif numbers:
            return float(numbers[0]) if "." in numbers[0] else int(numbers[0])
        else:
            return value.strip()  # Keep categorical labels like "Normal", "Yes", "No"
    return value  # Return as is if it's already a number


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your environment or .env file."
        )

    return genai.Client(api_key=api_key)

def extract_ckd_data_from_image(img_path):
    """
    Extracts CKD-related attributes from a medical report image using Gemini AI and returns cleaned JSON data.
    If the image does not appear to be a valid medical report, returns a message indicating the image is incorrect.

    Parameters:
        api_key (str): Google Generative AI API key.
        img_path (str): Path to the image file.

    Returns:
        dict or str: Extracted CKD attributes in JSON format, or a string if the image is not correct.
    """

    client = get_gemini_client()

    # Load the image
    image = Image.open(img_path)

    

    # Define the AI model
    #model = genai.GenerativeModel("gemini-pro")
    #model = genai.GenerativeModel("gemini-3-flash")
    # Prompt for extraction
    prompt = (
        "Extract the following data from the medical report image and return it in JSON format: "
        "age, bp (blood pressure), sg (specific gravity), al (albumin), su (sugar), rbc (red blood cells), "
        "pc (pus cell), pcc (pus cell clumps), ba (bacteria), bgr (blood glucose random), bu (blood urea), "
        "sc (serum creatinine), sod (sodium), pot (potassium), hemo (hemoglobin), pcv (packed cell volume), "
        "wc (white cell count), rc (red cell count), htn (hypertension), dm (diabetes mellitus), "
        "cad (coronary artery disease), appet (appetite), pe (pedal edema), ane (anemia). "
        "Return only numeric values and categorical labels, without any units."
    )

    # Generate response
    # response = model.generate_content([prompt, image])
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[image, prompt]
        )
    except Exception as exc:
        message = str(exc)
        if "PERMISSION_DENIED" in message or "API key" in message:
            raise RuntimeError(
                "Gemini API request failed. Your API key is invalid, blocked, or leaked. "
                "Create a new key and set GEMINI_API_KEY in your .env file."
            ) from exc
        raise RuntimeError(f"Gemini extraction failed: {message}") from exc
    # Extract text and clean it
    json_text = response.text.strip()

    # Ensure only valid JSON is extracted (remove anything before '{')
    json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
    if json_match:
        json_text = json_match.group(0)  # Extract only the JSON part

    # Convert to JSON format and clean values
    try:
        extracted_data = json.loads(json_text)
    except json.JSONDecodeError:
        return "the image is not correct"

    # Define expected keys for a valid medical report
    expected_keys = [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
        "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm",
        "cad", "appet", "pe", "ane"
    ]

    # Check if any of the expected attributes are present in the extracted data
    if not any(key in extracted_data for key in expected_keys):
        return "the image is not correct"

    # Clean values (remove any units or extraneous text)
    cleaned_data = {key: clean_value(value) for key, value in extracted_data.items()}

    return cleaned_data

# def extract_ckd_data_from_image(img_path):
#     """
#     Extracts CKD-related attributes from a medical report image using Gemini AI and returns cleaned JSON data.
#     If the image does not appear to be a valid medical report, returns a message indicating the image is incorrect.

#     Parameters:
#         api_key (str): Google Generative AI API key.
#         img_path (str): Path to the image file.

#     Returns:
#         dict or str: Extracted CKD attributes in JSON format, or a string if the image is not correct.
#     """
    
#   

#     # Load the image
#     image = Image.open(img_path)

#     # Define the AI model
#     # model = genai.GenerativeModel("gemini-2.0-flash-exp")
#     model = genai.GenerativeModel("gemini-3-flash")
#     # Prompt for extraction
#     prompt = (
#         "Extract the following data from the medical report image and return it in JSON format: "
#         "age, bp (blood pressure), sg (specific gravity), al (albumin), su (sugar), rbc (red blood cells), "
#         "pc (pus cell), pcc (pus cell clumps), ba (bacteria), bgr (blood glucose random), bu (blood urea), "
#         "sc (serum creatinine), sod (sodium), pot (potassium), hemo (hemoglobin), pcv (packed cell volume), "
#         "wc (white cell count), rc (red cell count), htn (hypertension), dm (diabetes mellitus), "
#         "cad (coronary artery disease), appet (appetite), pe (pedal edema), ane (anemia). "
#         "Return only numeric values and categorical labels, without any units."
#     )

#     # Generate response
#     response = model.generate_content([prompt, image])

#     # Extract text and clean it
#     json_text = response.text.strip()

#     # Ensure only valid JSON is extracted (remove anything before '{')
#     json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
#     if json_match:
#         json_text = json_match.group(0)  # Extract only the JSON part

#     # Convert to JSON format and clean values
#     try:
#         extracted_data = json.loads(json_text)
#     except json.JSONDecodeError:
#         return "the image is not correct"

#     # Define expected keys for a valid medical report
#     expected_keys = [
#         "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr",
#         "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm",
#         "cad", "appet", "pe", "ane"
#     ]

#     # Check if any of the expected attributes are present in the extracted data
#     if not any(key in extracted_data for key in expected_keys):
#         return "the image is not correct"

#     # Clean values (remove any units or extraneous text)
#     cleaned_data = {key: clean_value(value) for key, value in extracted_data.items()}

#     return cleaned_data
