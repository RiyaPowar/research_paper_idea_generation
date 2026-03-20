from call_llm import generate_ideas
import streamlit as st

st.title("Research Paper Idea Generator")

topic = st.text_input("Enter Research Domain")

if st.button("Generate Ideas"):
    ideas, docs = generate_ideas(topic)

    st.subheader("Retrieved Papers")
    for d in docs:
        st.write(d["text"])
    
    st.subheader("Generated Ideas")
    st.write(ideas)
