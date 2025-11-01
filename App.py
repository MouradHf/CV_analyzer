import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
# Chargement du modèle spaCy (anglais)
nlp = spacy.load('en_core_web_sm')

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # rendre la détection déterministe

import pandas as pd
import base64, random, time, datetime, json, re
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
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

    # Nom (entité PERSON) — si plusieurs, on prend la première
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Non détecté")
    # Email (regex)
    email = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    # Téléphone (regex basique)
    phone = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)
    pages = text.count("\f")

    # Skills extraction (liste basique - tu peux enrichir)
    skills_list = [
    # --- Langages de programmation ---
    "Python", "Java", "C", "C++", "C#", "R", "Go", "Rust", "Scala", "PHP", 
    "JavaScript", "TypeScript", "Swift", "Kotlin", "Ruby", "MATLAB", "Perl",

    # --- Web Development ---
    "HTML", "CSS", "SASS", "Bootstrap", "Tailwind", "React", "Next.js", "Vue.js",
    "Angular", "Django", "Flask", "FastAPI", "Node.js", "Express.js", "Laravel",
    "Spring Boot", "ASP.NET", "WordPress", "REST API", "GraphQL",

    # --- Data Science & Machine Learning ---
    "Machine Learning", "Deep Learning", "Data Science", "Data Analysis",
    "Data Visualization", "Natural Language Processing", "Computer Vision",
    "Predictive Modeling", "Reinforcement Learning", "MLOps", "AutoML",

    # --- Frameworks & Libraries ---
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "Matplotlib", "Seaborn", "OpenCV", "NLTK", "spaCy", "Hugging Face", 
    "Transformers", "XGBoost", "LightGBM", "CatBoost", "Statsmodels",

    # --- Databases ---
    "SQL", "MySQL", "PostgreSQL", "SQLite", "MongoDB", "Firebase", 
    "Oracle", "Redis", "Elasticsearch", "Cassandra", "Neo4j",

    # --- Cloud & DevOps ---
    "AWS", "Azure", "Google Cloud", "GCP", "Docker", "Kubernetes", "CI/CD",
    "Jenkins", "Terraform", "Ansible", "GitLab CI", "GitHub Actions", "Linux",
    "Nginx", "Apache", "DevOps", "Shell Scripting", "Bash", "Linux Administration",

    # --- Data Engineering / Big Data ---
    "Apache Spark", "Hadoop", "Kafka", "Airflow", "ETL", "Data Pipeline",
    "Snowflake", "Databricks", "Hive", "Pig", "Power BI", "Tableau", "Excel",
    "Data Warehouse", "BigQuery", "Data Lake", "NoSQL",

    # --- Mobile Development ---
    "Android", "iOS", "Flutter", "React Native", "SwiftUI", "Xcode",
    "XML", "Jetpack Compose", "Kivy",

    # --- UI/UX & Design ---
    "UI Design", "UX Design", "Figma", "Adobe XD", "Sketch", "InVision",
    "Wireframes", "Prototyping", "User Research", "Design Thinking",

    # --- Cybersecurity ---
    "Cybersecurity", "Network Security", "Penetration Testing", "Ethical Hacking",
    "Kali Linux", "Wireshark", "Metasploit", "OWASP", "Cryptography", "Firewall",
    "Vulnerability Assessment", "Incident Response", "SOC", "SIEM", "IDS", "IPS",

    # --- Project Management / Tools ---
    "Agile", "Scrum", "Kanban", "JIRA", "Trello", "Asana", "Notion",
    "Git", "GitHub", "GitLab", "Bitbucket", "Slack", "Confluence",
    "Continuous Integration", "Continuous Deployment", "Version Control",

    # --- Soft Skills / Autres compétences techniques ---
    "Problem Solving", "Teamwork", "Leadership", "Communication",
    "Time Management", "Critical Thinking", "Analytical Skills",
    "Cloud Computing", "API Development", "Microservices Architecture",
    "Software Testing", "Unit Testing", "Integration Testing", "Automation",
    "Selenium", "Postman", "JMeter", "Cypress", "TestNG", "PyTest",
    "Blockchain", "Smart Contracts", "Solidity", "Web3", "NFT", "Metaverse",

    # --- AI Specialized Areas ---
    "Generative AI", "LLM", "Prompt Engineering", "LangChain", "Vector Database",
    "Pinecone", "ChromaDB", "FAISS", "RAG", "OpenAI API", "ChatGPT API",
    "Speech Recognition", "Text-to-Speech", "Voice Cloning"
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
# Connexion MySQL (XAMPP par défaut: user=root, password vide)
connection = pymysql.connect(
    host='localhost', 
    user='root', 
    password='root',  # Mot de passe vide par défaut pour XAMPP
)
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
st.set_page_config(page_title="Smart Resume Analyzer", page_icon='./Logo/logo.png', layout='centered')

def run():
    st.title("Smart Resume Analyzer")
    st.sidebar.header("Choose User")
    choice = st.sidebar.selectbox("Choose:", ["Normal User", "Admin"])

    # Logo
    try:
        img = Image.open('./Logo/logo.png')
        st.image(img,  use_container_width=True)
    except Exception:
        pass

    if choice == "Normal User":
        pdf_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
        if pdf_file is not None:
            save_path = f'./Uploaded_Resumes/{pdf_file.name}'
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            try:
                show_pdf(save_path)
            except Exception:
                pass

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

            # Candidate Level based on pages
            pages = resume_data['no_of_pages']
            if pages == 1:
                cand_level = "Fresher"
            elif pages == 2:
                cand_level = "Intermediate"
            else:
                cand_level = "Experienced"
            st.markdown(f"**Candidate Level:** {cand_level}")

            # Recommendations by skill
            reco_field, recommended_skills, rec_course = '', [], []
            # --- Définition des catégories de compétences ---
            ds_keyword = ['tensorflow','keras','pytorch','machine learning','deep learning','flask','streamlit','data science','pandas','numpy','scikit-learn','matplotlib','seaborn','sql']
            web_keyword = ['react','django','node js','php','laravel','magento','wordpress','javascript','angular js','angular','c#','html','css','flask','express','typescript']
            android_keyword = ['android','flutter','kotlin','xml','kivy','java']
            ios_keyword = ['ios','swift','xcode','cocoa','cocoa touch']
            uiux_keyword = ['ux','ui','figma','adobe xd','wireframes','prototyping','user experience','user interface']

            # --- Initialisation des compteurs ---
            scores = {
                'Data Science': 0,
                'Web Development': 0,
                'Android Development': 0,
                'IOS Development': 0,
                'UI-UX Development': 0
            }

            # --- Comptage des correspondances ---
            for skill in resume_data['skills']:
                sk = skill.lower()
                if sk in ds_keyword:
                    scores['Data Science'] += 1
                if sk in web_keyword:
                    scores['Web Development'] += 1
                if sk in android_keyword:
                    scores['Android Development'] += 1
                if sk in ios_keyword:
                    scores['IOS Development'] += 1
                if sk in uiux_keyword:
                    scores['UI-UX Development'] += 1

            # --- Filtrer les domaines avec au moins une compétence détectée ---
            detected_domains = {domain: count for domain, count in scores.items() if count > 0}

            # --- Détermination du domaine dominant ---
            if detected_domains:
                reco_field = max(detected_domains, key=detected_domains.get)
                max_score = detected_domains[reco_field]
            else:
                reco_field = "Unknown"
                max_score = 0

            # --- Affichage du détail du comptage ---
            st.write("### Détails des correspondances détectées :")
            if detected_domains:
                st.json(detected_domains)
            else:
                st.info("Aucun domaine détecté à partir des compétences du CV.")

            st.write(f"✅ Domaine détecté : **{reco_field}** ({max_score} compétences détectées)")

            # --- Recommandations associées ---
            recommended_skills, rec_course = [], []

            if reco_field == "Data Science":
                recommended_skills = ['Data Visualization','Predictive Analysis','ML Algorithms','Keras','Tensorflow','Pytorch','Flask','Streamlit']
                rec_course = course_recommender(ds_course)

            elif reco_field == "Web Development":
                recommended_skills = ['React','Node.js','Django','PHP','JavaScript','MongoDB','Express','HTML','CSS']
                rec_course = course_recommender(web_course)

            elif reco_field == "Android Development":
                recommended_skills = ['Android','Flutter','Kotlin','XML','Java']
                rec_course = course_recommender(android_course)

            elif reco_field == "IOS Development":
                recommended_skills = ['iOS','Swift','Xcode','Cocoa']
                rec_course = course_recommender(ios_course)

            elif reco_field == "UI-UX Development":
                recommended_skills = ['Figma','Adobe XD','Wireframes','Prototyping']
                rec_course = course_recommender(uiux_course)

            # --- Affichage final ---
            st_tags(label="### Recommended Skills", value=recommended_skills, key='recommended_skills')



            # ----------------- Sections à vérifier (détection FR/EN) -----------------
            # Dictionnaires de sections en anglais et français (poids inchangés)
            sections_en = {
                "Summary": 10,
                "Education": 20,
                "Professional Experience": 20,
                "Skills": 20,
                "Projects": 20,
                "Languages": 5,
                "Certifications": 5
            }

            sections_fr = {
                "Profil": 10,
                "Formation": 20,
                "Expérience Professionnelle": 20,
                "Compétences": 20,
                "Projets": 20,
                "Langues": 5,
                "Certifications": 5
            }
            # === Synonymes supplémentaires (pour plus de flexibilité) ===
            synonyms = {
                "Professional Experience": ["experience", "work experience", "professional experience"],
                "Expérience Professionnelle": ["expérience", "experience professionnelle"],
                "Skills": ["skills", "technical skills", "competencies"],
                "Compétences": ["compétences", "compétences techniques"],
                "Summary": ["summary", "profile"],
                "Profil": ["profil", "résumé"]
            }

            # Détection de la langue (français si detect renvoie 'fr*' sinon anglais)
            try:
                lang = detect(resume_text)
            except Exception:
                lang = 'en'

            if str(lang).startswith('fr'):
                sections = sections_fr
                lang_label = "Français"
            else:
                sections = sections_en
                lang_label = "Anglais"

            resume_text_lower = resume_text.lower()
            found_sections = []
            missing_sections = []
            resume_score = 0

            # Évaluer la présence de chaque section (recherche simple du titre)
            for section, weight in sections.items():
                section_found = False

                # Vérifie si le titre exact est dans le texte
                if section.lower() in resume_text_lower:
                    resume_score += weight
                    section_found = True
                    found_sections.append(section)
                else:
                    # Vérifie les synonymes possibles
                    if section in synonyms:
                        for syn in synonyms[section]:
                            if syn.lower() in resume_text_lower:
                                resume_score += weight
                                section_found = True
                                found_sections.append(section)
                                break

                # Si la section n’a pas été trouvée du tout
                if not section_found:
                    missing_sections.append(section)

            # Limiter le score à 100 max
            if resume_score > 100:
                resume_score = 100


            # --- AFFICHAGE ---
            st.subheader("📋 Resume Tips & Evaluation")

            # Barre de progression
            st.progress(resume_score / 100)
            st.markdown(f"**Resume Score:** 🎯 {resume_score}/100")
            st.write(f"**Language Detected:** {lang_label}")

            # Section Analysis
            col1, col2 = st.columns(2)
            with col1:
                st.success("✅ **Identified Sections:**")
                if found_sections:
                    for sec in found_sections:
                        st.write(f"• {sec}")
                else:
                    st.write("No key sections identified.")
                    
            with col2:
                st.warning("⚠️ **Recommended Improvements:**")
                if missing_sections:
                    for sec in missing_sections:
                        st.write(f"• {sec}")
                else:
                    st.write("All essential sections are present.")

            # Improvement suggestions
            if missing_sections:
                st.info("💡 **Recommendations for Resume Enhancement**")
                tips_en = {
                    "Objective": "Add a short professional objective summarizing your career goals.",
                    "Summary": "Include a summary that highlights your strengths and experience.",
                    "Education": "Mention your degrees, institutions, and graduation dates.",
                    "Experience": "Detail your work experience with measurable results.",
                    "Skills": "List both technical and soft skills relevant to your field.",
                    "Projects": "Include a few key projects demonstrating your practical skills.",
                    "Achievements": "Mention awards, certifications, or special recognitions.",
                    "Hobbies": "Add hobbies if they reflect creativity, teamwork, or leadership."
                }
                tips_fr = {
                    "Objectif": "Ajoute un objectif professionnel court résumant tes objectifs de carrière.",
                    "Profil": "Inclure un résumé qui met en avant tes forces et ton expérience.",
                    "Formation": "Indique tes diplômes, établissements et dates.",
                    "Experience": "Détaille tes expériences avec des résultats mesurables.",
                    "Compétences": "Liste les compétences techniques et comportementales pertinentes.",
                    "Projets": "Ajoute quelques projets clés montrant ton travail concret.",
                    "Réalisations": "Mentionne distinctions, certifications ou résultats remarquables.",
                    "Loisirs": "Ajoute des loisirs qui montrent créativité, esprit d'équipe ou leadership."
                }
                for sec in missing_sections:
                    if str(lang).startswith('fr'):
                        tip = tips_fr.get(sec, "")
                    else:
                        tip = tips_en.get(sec, "")
                    if tip:
                        st.write(f"- **{sec}:** {tip}")

            # ----------------- Insert into DB -----------------
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                        str(resume_data['no_of_pages']), reco_field, cand_level, resume_data['skills'],
                        recommended_skills, rec_course)

            # Bonus videos
            st.header("Resume Tips Video")
            try:
                vid = random.choice(resume_videos)
                st.subheader(fetch_yt_video(vid))
                st.video(vid)
            except Exception:
                pass

            st.header("Interview Tips Video")
            try:
                vid2 = random.choice(interview_videos)
                st.subheader(fetch_yt_video(vid2))
                st.video(vid2)
            except Exception:
                pass

    else:
        st.subheader("Admin Panel")
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button("Login"):
            if ad_user == "root" and ad_password == "root":
                cursor.execute("SELECT * FROM user_data")
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=['ID','Name','Email','Resume Score','Timestamp','Total Page',
                                                 'Predicted Field','User Level','Actual Skills','Recommended Skills','Recommended Courses'])
                # Decode JSON columns
                for col in ['Actual Skills','Recommended Skills','Recommended Courses']:
                    try:
                        df[col] = df[col].apply(lambda x: ', '.join(json.loads(x)) if x else '')
                    except Exception:
                        pass
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv','Download Report'), unsafe_allow_html=True)

                st.subheader("Predicted Field Distribution")
                try:
                    fig = px.pie(df, names='Predicted Field', title='Field Distribution')
                    st.plotly_chart(fig)
                except Exception:
                    pass
                st.subheader("Experience Levels Distribution")
                try:
                    fig2 = px.pie(df, names='User Level', title='Experience Levels')
                    st.plotly_chart(fig2)
                except Exception:
                    pass
            else:
                st.error("Wrong credentials!")

run()
