import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
from ultralytics import YOLO
import ast
import numpy as np


# Page Config
st.set_page_config(
    page_title="Smart Fridge Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio("Go to", ["App", "Camera + Manual Input", "Image Upload + Manual Input", "Manual Input Only", "Action History"])

# Load files from fixed paths (assumes they are always present)
csv_path = os.path.join("data", "recipes.csv")
json_path = os.path.join("data", "alternatives.json")

recipes_df = pd.read_csv(csv_path)
recipes_df = recipes_df.rename(columns={'name': 'title', 'ingredient ': 'ingredients', 'steps ': 'steps'}, errors="ignore")
recipes_df = recipes_df.rename(columns={'Title': 'title', 'Ingredients': 'ingredients', 'Steps': 'steps'}, errors="ignore")
recipes_df = recipes_df[['title', 'ingredients', 'steps']]

with open(json_path, "r", encoding='utf-8') as f:
    substitutions = json.load(f)

model = YOLO("yolov8m.pt")

if "history" not in st.session_state:
    st.session_state.history = []

# Helper to parse fields
def parse_list_field(field):
    try:
        if isinstance(field, str) and field.strip().startswith("["):
            items = ast.literal_eval(field)
            return ", ".join(items) if isinstance(items, list) else str(items)
    except:
        pass
    return str(field)

# Recipe matching

def match_recipes(inputs):
    def get_match_score(ingredients_text):
        ingredients = [i.strip().lower() for i in str(ingredients_text).split(',')]
        found = [i for i in ingredients if any(i in x or x in i for x in inputs)]
        missing = [i for i in ingredients if i not in found]
        return len(found), len(ingredients), found, missing

    matches = []
    for _, row in recipes_df.iterrows():
        score, total, found, missing = get_match_score(row['ingredients'])
        if score > 0:
            ingredients = parse_list_field(row['ingredients'])
            steps = parse_list_field(row['steps'])
            matches.append((score / total, row['title'], ingredients, steps, found, missing))

    matches.sort(reverse=True)
    return matches

# Core detection + match

def detect_ingredients(image):
    img = Image.fromarray(image)
    img.save("input.jpg")

    results = model("input.jpg")
    names = model.names
    classes = results[0].boxes.cls
    detected_raw = [names[int(cls)] for cls in classes]

    label_map = {
        "bottle": "milk", "jar": "honey", "bowl": "fruit", "vase": "vegetable",
        "potted plant": "vegetable", "cup": "yogurt", "orange": "orange", "apple": "apple",
        "banana": "banana", "broccoli": "broccoli", "carrot": "carrot",
        "sandwich": "sandwich", "pizza": "pizza", "donut": "pastry", "refrigerator": "fridge"
    }
    labels_lower = [label_map.get(x.lower(), x.lower()) for x in detected_raw]

    ocr_text = pytesseract.image_to_string(cv2.imread("input.jpg")).lower()
    ocr_words = [word.strip() for word in ocr_text.split() if word.strip().isalpha()]

    smart_detected = set(labels_lower + ocr_words)
    if any(word in smart_detected for word in ['egg', 'eggs', 'بيض', 'بيضه', 'بيضة']):
        smart_detected.add('egg')

    return list(smart_detected)

# Render result

def display_matches(all_inputs):
    matches = match_recipes(all_inputs)
    if not matches:
        st.warning("⚠️ No matching recipes found.")
        return

    for ratio, title, ingredients, steps, found, missing in matches:
        with st.expander(f"🔹 {title} ({int(ratio * 100)}% match)"):
            st.markdown(f"✅ **Found:** {', '.join(found)}")
            st.markdown(f"❌ **Missing:** {', '.join(missing) if missing else 'None'}")
            subs = [f"🔁 {m} → {', '.join(substitutions[m])}" for m in missing if m in substitutions]
            if subs:
                st.markdown("💡 **Substitutions:**")
                st.markdown("\n".join(subs))
            st.markdown("📝 **Steps:**")
            st.markdown(steps)

            if st.button(f"💾 Save {title}", key=title):
                st.session_state.history.append({
                    "title": title,
                    "found": found,
                    "missing": missing,
                    "steps": steps
                })
                st.success(f"Saved {title} to history!")

# Pages

if page == "App":
    st.title("🥗 AI chefmate")
    st.markdown("A smart assistant that suggests recipes based on what it sees from your fridge📷🧠")
    st.image("Screenshot_4-5-2025_221314_.jpeg", width=200)
    st.markdown("""
    <div style='font-size: 14px; line-height: 1.6;'>
    Welcome to the <b>AI chefmate</b> app!<br><br>
    📸 Use your camera or upload a photo of your fridge contents,<br>
    🔎 Let AI detect ingredients,<br>
    🥘 We'll suggest smart recipes,<br>
    💾 You can even save the recipes you like!<br>
    <hr>
    <h4>🧭 Available pages:</h4>
    <ul>
      <li>📷 <b>Camera</b>: Take a live photo and analyze it</li>
      <li>🖼️ <b>Upload Image</b>: Choose a photo from your device to analyze</li>
      <li>💾 <b>Saved Recipes</b>: Browse recipes you've previously saved</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.info("⬅️ Use the side menu to get started!")
    st.markdown("---")
    st.caption("This project was developed using Streamlit and YOLOv8 | Graduation Project❤️")

elif page == "Camera + Manual Input":
    st.header("📷 Camera + Manual Ingredients")
    image = st.camera_input("Take a picture")
    manual = st.text_input("Add ingredients (comma separated)")
    if image:
        img = Image.open(image)
        detected = detect_ingredients(np.array(img))
        manual_list = [i.strip().lower() for i in manual.split(',') if i.strip()]
        all_inputs = list(set(detected + manual_list))
        st.markdown("### 📦 Detected Ingredients:")
        st.markdown(", ".join(all_inputs))
        display_matches(all_inputs)

elif page == "Image Upload + Manual Input":
    st.header("🖼️ Upload Image + Manual Ingredients")
    image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    manual = st.text_input("Add ingredients (comma separated)")
    if image:
        img = Image.open(image)
        detected = detect_ingredients(np.array(img))
        manual_list = [i.strip().lower() for i in manual.split(',') if i.strip()]
        all_inputs = list(set(detected + manual_list))
        st.markdown("### 📦 Detected Ingredients:")
        st.markdown(", ".join(all_inputs))
        display_matches(all_inputs)

elif page == "Manual Input Only":
    st.header("✍️ Manual Ingredients Only")
    manual = st.text_input("Add ingredients (comma separated)")
    if manual:
        manual_list = [i.strip().lower() for i in manual.split(',') if i.strip()]
        st.markdown("### 📦 Input Ingredients:")
        st.markdown(", ".join(manual_list))
        display_matches(manual_list)

elif page == "Action History":
    st.header("📜 Saved Recipes")
    if not st.session_state.history:
        st.info("No saved recipes yet.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"{i+1}. {item['title']}"):
                st.markdown(f"✅ **Found:** {', '.join(item['found'])}")
                st.markdown(f"❌ **Missing:** {', '.join(item['missing']) if item['missing'] else 'None'}")
                st.markdown("📝 **Steps:**")
                st.markdown(item['steps'])
