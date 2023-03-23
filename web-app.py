import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

#@st.cache_data()

model = YOLO('forest.pt')

def main():
    st.title("Tree Segmentation")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        inputs = Image.open(uploaded_file)
        st.image(inputs, caption='Uploaded Image.', use_column_width=True)

        results = model.predict(source=inputs, conf=0.1)
        result = results[0]
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        num_trees = len(class_ids)
        text = f'Number of Trees Detected: {num_trees}'
        font = cv2.FONT_HERSHEY_SIMPLEX

        res_plotted = result.plot()
        res_plotted = cv2.putText(res_plotted, text, (390, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


        
        st.subheader("Segmentation Results:")
        st.image(res_plotted, caption='Predicted Image.', use_column_width=True, channels='BGR')
        st.write(f"Number of Instances: {num_trees}")
if __name__ == '__main__':
    main()