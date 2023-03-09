import streamlit as st

from face_detection import detection

st.title("写真から顔を検出する")
image = st.file_uploader("写真アップロード", "jpg")

before, after = st.columns(2)
before.subheader("入力写真")
after.subheader("検出結果")
if image:
    before.image(image)
    after.image(detection(image))
