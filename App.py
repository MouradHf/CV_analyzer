import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

import pandas as pd
import base64, random, time, datetime, json, re
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import pafy
import plotly.express as px

# ----------------- UTILS -----------------
def fetch_yt_video(link):
    try:
        video = pafy.new(link)
        return video.title
    except:
        return "YouTube Video"

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for i, (c_name, c_link) in enumerate(course_list):
        st.markdown(f"({i+1}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if i+1 == no_of_reco:
            break
    return rec_course

# ----------------- EXTRACT RESUME DATA -----------------
def extract_resume_data(text):
    doc = nlp(text)

    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Non détecté")
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)
    pages = text.count("\f") + 1

    # Skills extraction
    skills_list = [
        "Python","Java","C++","SQL","Machine Learning","Deep Learning",
        "Data Science","Flask","Streamlit","React","Node.js","MongoDB",
        "Tensorflow","Keras","Pytorch","HTML","CSS","JavaScript","Angular","Django","PHP"
    ]
    found_skills = [s for s in skills_list if s.lower() in text.lower()]

    return {
        "name": name,
        "email": email[0] if email else "Non détecté",
        "mobile_number": phone[0] if phone else "Non détecté",
        "no_of_pages": pages,
        "skills": found_skills
    }

# ----------------- DATABASE -----------------
connection = pymysql.connect(host='localhost', user='root', password='root')
cursor = connection.cursor()

cursor.execute("CREATE DATABASE IF NOT EXISTS SRA;")
connection.select_db("SRA")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_data (
    ID INT NOT NULL AUTO_INCREMENT,
    Name VARCHAR(100),
    Email_ID VARCHAR(50),
    resume_score VARCHAR(8),
    Timestamp VARCHAR(50),
    Page_no VARCHAR(5),
    Predicted_Field VARCHAR(25),
    User_level VARCHAR(30),
    Actual_skills TEXT,
    Recommended_skills TEXT,
    Recommended_courses TEXT,
    PRIMARY KEY (ID)
);
""")

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    sql = """
    INSERT INTO user_data
    (ID, Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
    VALUES (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cursor.execute(sql, (name, email, res_score, timestamp, no_of_pages, reco_field, cand_level,
                         json.dumps(skills), json.dumps(recommended_skills), json.dumps(courses)))
    connection.commit()

# ----------------- STREAMLIT APP -----------------
st.set_page_config(page_title="Smart Resume Analyzer", page_icon='./Logo/SRA_Logo.ico')

def run():
    st.title("Smart Resume Analyzer")
    st.sidebar.header("Choose User")
    choice = st.sidebar.selectbox("Choose:", ["Normal User", "Admin"])

    img = Image.open('./Logo/SRA_Logo.jpg')
    st.image(img.resize((250, 250)))

    if choice == "Normal User":
        pdf_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
        if pdf_file is not None:
            save_path = f'./Uploaded_Resumes/{pdf_file.name}'
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_path)

            resume_text = pdf_reader(save_path)
            resume_data = extract_resume_data(resume_text)

            st.header("**Resume Analysis**")
            st.subheader("Basic Info")
            st.text(f"Name: {resume_data['name']}")
            st.text(f"Email: {resume_data['email']}")
            st.text(f"Contact: {resume_data['mobile_number']}")
            st.text(f"Pages: {resume_data['no_of_pages']}")

            # Skills
            st_tags(label='### Skills Detected', value=resume_data['skills'], key='skills_detected')

            # Candidate Level
            pages = resume_data['no_of_pages']
            if pages == 1:
                cand_level = "Fresher"
            elif pages == 2:
                cand_level = "Intermediate"
            else:
                cand_level = "Experienced"
            st.markdown(f"**Candidate Level:** {cand_level}")

            # Recommendations
            reco_field, recommended_skills, rec_course = '', [], []
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep learning','flask','streamlit']
            web_keyword = ['react','django','node js','php','laravel','magento','wordpress','javascript','angular js','c#','flask']
            android_keyword = ['android','flutter','kotlin','xml','kivy']
            ios_keyword = ['ios','swift','xcode','cocoa','cocoa touch']
            uiux_keyword = ['ux','figma','adobe xd','wireframes','prototyping']

            for skill in resume_data['skills']:
                sk = skill.lower()
                if sk in ds_keyword:
                    reco_field = "Data Science"
                    recommended_skills = ['Data Visualization','Predictive Analysis','ML Algorithms','Keras','Tensorflow','Pytorch','Flask','Streamlit']
                    rec_course = course_recommender(ds_course)
                    break
                elif sk in web_keyword:
                    reco_field = "Web Development"
                    recommended_skills = ['React','Node.js','Django','PHP','JavaScript']
                    rec_course = course_recommender(web_course)
                    break
                elif sk in android_keyword:
                    reco_field = "Android Development"
                    recommended_skills = ['Android','Flutter','Kotlin','XML']
                    rec_course = course_recommender(android_course)
                    break
                elif sk in ios_keyword:
                    reco_field = "IOS Development"
                    recommended_skills = ['iOS','Swift','Xcode','Cocoa']
                    rec_course = course_recommender(ios_course)
                    break
                elif sk in uiux_keyword:
                    reco_field = "UI-UX Development"
                    recommended_skills = ['Figma','Adobe XD','Wireframes','Prototyping']
                    rec_course = course_recommender(uiux_course)
                    break

            st_tags(label="### Recommended Skills", value=recommended_skills, key='recommended_skills')

            # Resume Tips & Score
            st.subheader("Resume Tips & Ideas")
            resume_score = 0
            for key in ['Objective','Declaration','Hobbies','Achievements','Projects']:
                if key.lower() in resume_text.lower():
                    resume_score += 20
            st.subheader(f"Resume Score: {resume_score}/100")

            # Insert into DB
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                        str(resume_data['no_of_pages']), reco_field, cand_level, resume_data['skills'],
                        recommended_skills, rec_course)

            # Bonus videos
            st.header("Resume Tips Video")
            vid = random.choice(resume_videos)
            st.subheader(fetch_yt_video(vid))
            st.video(vid)

            st.header("Interview Tips Video")
            vid2 = random.choice(interview_videos)
            st.subheader(fetch_yt_video(vid2))
            st.video(vid2)

    else:
        st.subheader("Admin Panel")
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button("Login"):
            if ad_user == "admin" and ad_password == "admin":
                cursor.execute("SELECT * FROM user_data")
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['ID','Name','Email','Resume Score','Timestamp','Total Page',
                                                 'Predicted Field','User Level','Actual Skills','Recommended Skills','Recommended Courses'])
                # Decode JSON columns
                for col in ['Actual Skills','Recommended Skills','Recommended Courses']:
                    df[col] = df[col].apply(lambda x: ', '.join(json.loads(x)))
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv','Download Report'), unsafe_allow_html=True)

                st.subheader("Predicted Field Distribution")
                fig = px.pie(df, names='Predicted Field', title='Field Distribution')
                st.plotly_chart(fig)
                st.subheader("Experience Levels Distribution")
                fig2 = px.pie(df, names='User Level', title='Experience Levels')
                st.plotly_chart(fig2)
            else:
                st.error("Wrong credentials!")

run()
