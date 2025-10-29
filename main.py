# main.py

import os
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Course Recommendation System", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#555;'>Find similar courses from our curated Coursera dataset!</h4>", unsafe_allow_html=True)

# --- Load Models ---
try:
    courses_df = pickle.load(open('models/courses.pkl', 'rb'))
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
except Exception as e:
    st.error("❌ Error loading model files. Please ensure 'courses.pkl' and 'similarity.pkl' exist in the 'models' folder.")
    st.stop()

# --- Load Dataset ---
data_path = os.path.join("Data", "Coursera.csv")
if not os.path.exists(data_path):
    st.error("❌ Coursera.csv not found in the 'Data' folder.")
    st.stop()

data = pd.read_csv(data_path)

# --- Create URL Mapping ---
url_dict = {}
if 'Course Name' in data.columns and 'Course URL' in data.columns:
    for i, row in data.iterrows():
        url_dict[row['Course Name'].strip()] = row['Course URL']

# --- Recommendation Function ---
def recommend(course):
    if course not in courses_df['course_name'].values:
        return []
    index = courses_df[courses_df['course_name'] == course].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    return [courses_df.iloc[i[0]].course_name for i in distances[1:7]]

# --- Sidebar ---
st.sidebar.header("🔍 Find Similar Courses")
selected_course = st.sidebar.selectbox("Search or select a course:", sorted(courses_df['course_name'].values))

if st.sidebar.button("Recommend"):
    st.markdown("<h3 style='color:#1A5276;'>📘 Recommended Courses:</h3>", unsafe_allow_html=True)
    recommended_courses = recommend(selected_course)

    if recommended_courses:
        cols = st.columns(3)
        for i, course in enumerate(recommended_courses):
            course_row = data[data['Course Name'].str.strip() == course.strip()]
            university = course_row['University'].iloc[0] if 'University' in course_row.columns and not course_row.empty else 'N/A'
            difficulty = course_row['Difficulty Level'].iloc[0] if 'Difficulty Level' in course_row.columns and not course_row.empty else 'N/A'
            rating = course_row['Course Rating'].iloc[0] if 'Course Rating' in course_row.columns and not course_row.empty else 'N/A'
            course_url = url_dict.get(course.strip(), None)

            with cols[i % 3]:
                st.markdown(f"""
                <div style="background-color:#F8F9F9; border-radius:15px; padding:15px; margin:10px;
                            box-shadow:2px 2px 8px rgba(0,0,0,0.1); text-align:center;">
                    <h4 style="color:#2E4053;">{course}</h4>
                    <p><b>University:</b> {university}</p>
                    <p><b>Difficulty:</b> {difficulty}</p>
                    <p><b>Rating:</b> ⭐ {rating}</p>
                    {"<a href='"+course_url+"' target='_blank' style='color:#1A5276; text-decoration:none;'>🔗 View Course</a>" if course_url else "<i style='color:gray;'>No link available</i>"}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations found for this course.")

# --- Analytics Section ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#2E4053;'>📊 Course Analytics Dashboard</h2>", unsafe_allow_html=True)

if not data.empty:
    col1, col2 = st.columns(2)

    with col1:
        if 'Difficulty Level' in data.columns:
            st.markdown("**Difficulty Level Distribution**")
            fig1 = px.histogram(data, x='Difficulty Level', color='Difficulty Level', title='Courses by Difficulty Level')
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if 'Course Rating' in data.columns:
            st.markdown("**Course Ratings Distribution**")
            fig2 = px.histogram(data, x='Course Rating', nbins=20, title='Course Ratings Distribution',
                                color_discrete_sequence=['#1F618D'])
            st.plotly_chart(fig2, use_container_width=True)

    if 'University' in data.columns:
        st.markdown("**Top 10 Universities by Number of Courses**")
        uni_counts = data['University'].value_counts().head(10).reset_index()
        uni_counts.columns = ['University', 'Course Count']
        fig3 = px.bar(uni_counts, x='University', y='Course Count', color='University',
                      title='Top 10 Universities Offering Courses')
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("⚠️ Analytics data not found. Please place 'Coursera.csv' inside the 'Data' folder.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Developed by <b>Group 14</b> | Powered by Streamlit & Machine Learning</p>", unsafe_allow_html=True)
