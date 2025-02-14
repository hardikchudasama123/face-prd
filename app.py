import streamlit as st
import pickle
import numpy as np
import cv2
import pywt
import joblib
import json
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load trained model and class dictionary
with open("saved_model.pkl", "rb") as file:
    model = joblib.load(file)

with open("class_dictionary.json", "r") as file:
    class_dict = json.load(file)

# Function to detect face and ensure two eyes are present
def get_cropped_img_if_2_eyes(img):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img = np.array(img)  
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    
    return cropped_faces

# Function for Wavelet Transform
def w2d(img, mode='db1', level=5):
    imga = img
    imga = cv2.cvtColor(imga,cv2.COLOR_RGB2GRAY)
    imga = imga / 255 

   
    coeffs = pywt.wavedec2(imga, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    img_H = pywt.waverec2(coeffs_H, mode)
    img_H *= 255
    img_H = np.uint8(img_H)

    raw_img = cv2.resize(img, (32, 32))
    scal_img = cv2.resize(img_H, (32, 32))

    combined_img = np.vstack((raw_img.reshape(32*32*3, 1), scal_img.reshape(32*32, 1)))
    combined_img= combined_img.reshape(1, -1).astype('float32')
    combined_img=np.array(combined_img).reshape(len(combined_img),4096).astype('float32')

      
    return combined_img


st.title("Celebrity Face Recognition")
st.subheader("Upload the photo to the below mentioned persons")


a,b,c,d,e=st.columns(5)
img_size = 150

with a:
    st.image("img/rtik.jpg",caption="Hrithik roshan")   
with b:
     st.image("img/prabhas.jpg",caption="Prabhas")
with c:
    st.image("img/mesi.jpg",caption="Leo messi")
with d:
    st.image("img/smriti.jpg",caption="Smriti mandhana")
with e:
    st.image("img/yami.jpg",caption="Yami gautam")

st.write("""Upload an image to classify the person.""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = np.array(image)

    cropped_faces = get_cropped_img_if_2_eyes(image)
    
    if not cropped_faces:
        st.write("No face with two visible eyes detected.")
    else:
        scaler = StandardScaler()
       
        processed_images=[w2d(face) for face in cropped_faces]
        for i, img in enumerate(processed_images):
            img = img.reshape(1, -1)  # Ensure shape consistency

            if not isinstance(model, Pipeline):
                img = scaler.transform(img)

            prediction = model.predict(img)
            pre = prediction[0]

            probabilities = model.predict_proba(img)[0]


            predicted_probability = np.round(probabilities[prediction] * 100, 3)
            reverse_dict = {v: k for k, v in class_dict.items()}
            
            
            key = reverse_dict.get(pre, "Unknown") 
            if(predicted_probability>45):
                st.write(f"### probability : {predicted_probability}")
                st.write(f"### Prediction: {key}")
            else:
                st.write(f"### Unknown person ")

            for i, prob in enumerate(probabilities):
                st.write(f"{reverse_dict.get(i)}   :{np.round(prob*100,2)}%")
# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            text-align: center;
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
    </style>
    <div class="footer">
        Developed by Hardik Chudasama 🚀
    </div>
""", unsafe_allow_html=True)

