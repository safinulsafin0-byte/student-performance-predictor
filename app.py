
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "student_performance_rf.joblib"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's estimated final grade (G3) "
    "using a trained Random Forest regression model."
)

st.info(
    "This is an educational prediction tool. "
    "Predictions are estimates, not guaranteed results."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place "
        "'student_performance_rf.joblib' beside app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


with st.form("student_form"):
    st.subheader("1. Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.number_input("Age", 15, 22, 17)
        address = st.selectbox("Home address", ["U", "R"])
        famsize = st.selectbox("Family size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parents' cohabitation status", ["T", "A"])

    with col2:
        Medu = st.selectbox("Mother's education (0–4)", [0, 1, 2, 3, 4], index=2)
        Fedu = st.selectbox("Father's education (0–4)", [0, 1, 2, 3, 4], index=2)
        Mjob = st.selectbox(
            "Mother's job",
            ["at_home", "health", "other", "services", "teacher"]
        )
        Fjob = st.selectbox(
            "Father's job",
            ["at_home", "health", "other", "services", "teacher"]
        )
        reason = st.selectbox(
            "School choice reason",
            ["course", "home", "reputation", "other"]
        )
        guardian = st.selectbox(
            "Student's guardian",
            ["mother", "father", "other"]
        )

    with col3:
        traveltime = st.selectbox("Travel time (1–4)", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly study time (1–4)", [1, 2, 3, 4])
        failures = st.selectbox("Past class failures (0–4)", [0, 1, 2, 3, 4])
        schoolsup = st.selectbox("Extra educational support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])

    st.subheader("2. Lifestyle and Support")

    col4, col5, col6 = st.columns(3)

    with col4:
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])

    with col5:
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])
        famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1–5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1–5)", 1, 5, 3)

    with col6:
        Dalc = st.slider("Workday alcohol consumption (1–5)", 1, 5, 1)
        Walc = st.slider("Weekend alcohol consumption (1–5)", 1, 5, 1)
        health = st.slider("Self-reported health (1–5)", 1, 5, 3)
        absences = st.number_input("School absences", 0, 100, 0)

    st.subheader("3. Previous Grades")

    col7, col8 = st.columns(2)

    with col7:
        G1 = st.number_input("First period grade (G1)", 0, 20, 10)

    with col8:
        G2 = st.number_input("Second period grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button(
        "Predict Final Grade",
        type="primary",
        use_container_width=True
    )


if submitted:
    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2
    }])

    try:
        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, min(20.0, prediction))

        st.subheader("Prediction Result")
        st.metric("Estimated Final Grade (G3)", f"{prediction:.2f} / 20")

        if prediction >= 16:
            st.success("High predicted grade range")
        elif prediction >= 10:
            st.info("Middle predicted grade range")
        else:
            st.warning("Lower predicted grade range")

        st.caption(
            "The grade range labels are illustrative, not official "
            "academic classifications. The model may make errors."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")