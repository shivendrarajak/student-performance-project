import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# LOAD MODELS

cgpa_model = pickle.load(open("models/cgpa_model.pkl", "rb"))
sgpa_model = pickle.load(open("models/sgpa_model.pkl", "rb"))
grade_model = pickle.load(open("models/grade_model.pkl", "rb"))
perf_model = pickle.load(open("models/perf_model.pkl", "rb"))

scaler = pickle.load(open("models/scaler.pkl", "rb"))

subject_encoder = pickle.load(open("models/subject_encoder.pkl", "rb"))
perf_encoder = pickle.load(open("models/perf_encoder.pkl", "rb"))

# Grade reverse mapping
grade_map_rev = {
    10:"A+", 9:"A", 8:"B+", 7:"B",
    6:"C+", 5:"C", 4:"D+", 3:"D"
}


# UI CONFIG + CSS

st.set_page_config(page_title="Student Performance", layout="wide")

st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1f77b4, #4CAF50);
    color: white;
    text-align: center;
    font-size: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
.small-card {
    padding: 15px;
    border-radius: 10px;
    background: #1c1f26;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Dashboard</h1>", unsafe_allow_html=True)


# SIDEBAR INPUT

st.sidebar.header("📌 Student Details")
st.sidebar.info(
    "Enter academic details to predict CGPA, SGPA, "
    "Performance and Subject Grades.")

student_type = st.sidebar.radio(
    "Select Student Type",
    ["Regular", "Lateral Entry"]
)

if student_type == "Regular":
    tenth = st.sidebar.number_input("10th %", 0.0, 100.0)
    twelfth = 0.0
else:
    twelfth = st.sidebar.number_input("12th %", 0.0, 100.0)
    tenth = 0.0


# SEM INPUT

st.subheader("📊 Semester SGPA")

col1, col2, col3 = st.columns(3)

with col1:
    sem1 = st.number_input("Sem 1",0.0,10.0)
    sem4 = st.number_input("Sem 4",0.0,10.0)

with col2:
    sem2 = st.number_input("Sem 2",0.0,10.0)
    sem5 = st.number_input("Sem 5",0.0,10.0)

with col3:
    sem3 = st.number_input("Sem 3",0.0,10.0)



# SEMESTER WISE SUBJECTS


semester_subjects = {

    "Semester-I": [
        "Mathematics-I [T]",
        "Applied Physics-I [T]",
        "Applied Chemistry [T]",
        "Communication Skills [T]",
        "Applied Physics-I [P]",
        "Applied Chemistry [P]",
        "Communication Skills [P]",
        "Engineering Graphics [P]",
        "Engineering Workshop Practice [P]",
        "Sports and Yoga [P]"
    ],

    "Semester-II": [
        "Mathematics-II [T]",
        "Applied Physics-II [T]",
        "Introduction to IT Systems [T]",
        "Fundamentals of Electrical and Electronics Engineering [T]",
        "Engineering Mechanics [T]",
        "Applied Physics-II [P]",
        "Introduction to IT Systems [P]",
        "Fundamentals of Electrical and Electronics Engineering [P]",
        "Engineering Mechanics [P]"
    ],

    "Semester-III": [
        "Computer Programming [T]",
        "Scripting Languages [T]",
        "Data Structures [T]",
        "Computer System Organisation [T]",
        "Algorithms [T]",
        "Computer Programming [P]",
        "Scripting Languages [P]",
        "Data Structures [P]",
        "Summer Internship-I"
    ],

    "Semester-IV": [
        "Operating Systems [T]",
        "Introduction to DBMS [T]",
        "Computer Network [T]",
        "Software Engineering [T]",
        "Web Technologies [T]",
        "Operating Systems [P]",
        "Introduction to DBMS [P]",
        "Computer Networks [P]",
        "Web Technologies [P]",
        "Minor Project [P]"
    ],

    "Semester-V": [
        "E-Governance [T]",
        "IoT [T]",
        "Information Security [T]",
        "Information Security [P]",
        "Advance Computer Network [T]",
        "Data Sciences [T]",
        "Renewable Energy [T]",
        "Summer Internship-II"
    ],

    "Semester-VI": [
        "Entrepreneurship and Startup [T]",
        "Mobile Computing [T]",
        "Software Testing [T]",
        "Project Management [T]",
        "Artificial Intelligence [T]",
        "Mobile Computing [P]",
        "Software Testing [P]",
        "Major Project [P]",
        ]
}


# SELECT SEMESTER


selected_sem = st.selectbox(
    "📚 Select Semester",
    list(semester_subjects.keys())
)


# SUBJECT SELECTION

subjects = st.multiselect(
    "📘 Select Subjects",
    semester_subjects[selected_sem]
)

# PREDICTION

if st.button("🚀 Predict"):

    base_input = [[
        tenth, twelfth,
        sem1, sem2, sem3, sem4, sem5
    ]]

    scaled = scaler.transform(base_input)

    cgpa = cgpa_model.predict(scaled)[0]
    next_sgpa = sgpa_model.predict(scaled)[0]
    perf_pred = perf_model.predict(scaled)

    performance = perf_encoder.inverse_transform(
        perf_model.predict(scaled)
    )[0]
    avg_sgpa = np.mean([sem1, sem2, sem3, sem4, sem5])
    # CARDS 
    st.subheader("📈 Prediction Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        f"""
        <div class='card'>
        🎯 Final CGPA <br><br>
        <h2>{round(cgpa,2)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class='card'>
        📊 Next SGPA <br><br>
        <h2>{round(next_sgpa,2)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class='card'>
        🏆 Performance <br><br>
        <h2>{performance}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    c4.markdown(
        f"""
        <div class='card'>
        📚 Avg SGPA <br><br>
        <h2>{round(avg_sgpa,2)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


    # PERFORMANCE MESSAGE
    if cgpa >= 8:
        st.success("🔥 Excellent Performance")
    elif cgpa >= 6:
        st.warning("👍 Good Performance")
    else:
        st.error("⚠️ Needs Improvement")

    st.divider()

    # GRAPH 
    
    st.subheader("📉 SGPA Trend")

    sgpa_list = [sem1, sem2, sem3, sem4, sem5]

    fig, ax = plt.subplots()
    ax.plot(sgpa_list, marker='o', linewidth=3)
    ax.fill_between(range(len(sgpa_list)), sgpa_list, alpha=0.2)

    ax.set_xticks(range(len(sgpa_list)))
    ax.set_xticklabels(['Sem1','Sem2','Sem3','Sem4','Sem5'])

    ax.set_title("Performance Trend")
    st.pyplot(fig)

    st.divider()

    #  SUBJECT GRADE 
    st.subheader("📘 Subject-wise Grades")

    results = []

    for sub in subjects:
        sub_enc = subject_encoder.transform([sub])[0]

        grade_input = [[
            tenth, twelfth,
                sem1,
                sem2,
                sem3,
                sem4,
                sem5,
                sub_enc
        ]  ]

    grade_pred = grade_model.predict(grade_input)[0]

    grade = grade_map_rev.get(int(grade_pred), "N/A")

    results.append([sub, grade])

    cols = st.columns(3)

    for i, (sub, grade) in enumerate(results):
        cols[i % 3].markdown(
            f"<div class='small-card'>{sub}<br><b>{grade}</b></div>",
            unsafe_allow_html=True
        )
        
# SHOW RESULTS
    # -----------------------------------------------------

    if results:

        df_result = pd.DataFrame(
            results,
            columns=["Subject", "Predicted Grade"]
        )

        st.dataframe(
            df_result,
            use_container_width=True
        )

        # DOWNLOAD CSV
        csv = df_result.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Result CSV",
            csv,
            "student_prediction.csv",
            "text/csv"
        )

    else:

        st.warning("⚠️ Please select subjects")

    st.divider()


# 📊 MODEL COMPARISON DASHBOARD

st.divider()
st.subheader("📊 Model Comparison Dashboard")

try:
    results_df = pd.read_csv("models/model_results.csv")

    st.dataframe(results_df, use_container_width=True)

    # R2 GRAPH
    fig1, ax1 = plt.subplots()
    ax1.bar(results_df["Model"], results_df["R2 Score"])
    ax1.set_title("R2 Score Comparison")
    st.pyplot(fig1)

    # CV GRAPH
    fig2, ax2 = plt.subplots()
    ax2.bar(results_df["Model"], results_df["CV Score"])
    ax2.set_title("Cross Validation Score")
    st.pyplot(fig2)

    # BEST MODEL
    best_model = results_df.loc[results_df["R2 Score"].idxmax()]
    st.success(f"🏆 Best Model: {best_model['Model']} (R2: {round(best_model['R2 Score'],2)})")

except:
    st.warning("⚠️ Model comparison file not found. Please generate model_results.csv")


# FOOTER

st.divider()
st.caption("🎓 Final Year Major Project | ML + Streamlit")