import streamlit as st
import joblib
import pandas as pd
import gmail

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")

st.set_page_config(layout="wide")


st.markdown("""
    <h1 style='background-color: powderblue; color: black; padding: 10px; border-radius: 10px; text-align: center;'>
        🎯 LENS eXpert(NLP Suite)
    </h1>
""", unsafe_allow_html=True)
st.title("")
tab1,tab2,tab3,tab4=st.tabs(["🤖 Spam Classifier","🗣️ Language Detection","👍 Food Review Sentiment 👎","📰 News Classification"])
with tab1:
    msg=st.text_input("Enter Msg")
    if st.button("Prediction"):
        pred=spam_model.predict([msg])
        if pred[0]==0:
            st.image("spam.jpg")
        else:
            st.image("not_spam.png")

    uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"])
   
  
    if uploaded_file:
            
        df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Prediction"]=pred
        df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam)

with tab2:
    msg=st.text_input("Enter Text")
    if st.button("Prediction",key="b2"):
        pred=language_model.predict([msg])
        st.success(pred[0])

    uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u2")
   
  
    if uploaded_file:
            
        df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=language_model.predict(df.Msg)
        df.index=range(1,df.shape[0]+1)
        df["Prediction"]=pred
        st.dataframe(df)

with tab3:
    msg=st.text_input("Enter Review")
    if st.button("Prediction",key="b3"):
        pred=review_model.predict([msg])
        if pred[0]==0:
            st.image("dislike.jpg")
        else:
            st.image("like.png")

    uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u3")
   
  
    if uploaded_file:
            
        df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=review_model.predict(df.Msg)
        df.index=range(1,df.shape[0]+1)
        df["Prediction"]=pred
        df["Prediction"]=df["Prediction"].map({0:'Disliked',1:'Liked'})
        st.dataframe(df)
with tab4:
    st.image("under_const.jpg")
st.sidebar.image("f:/images/flag.jpg")

with st.sidebar.expander("🧑‍🤝‍🧑 About us"):
    st.write("We are a group of students trying to understand the concept of NLP")

with st.sidebar.expander("📞 Contact us"):
    st.write("999999999")
    st.write("aaaaa@gamil.com")
    
with st.sidebar.expander("🤝 Help"):
    st.write("we have used sklearn & nltk libs")
    

