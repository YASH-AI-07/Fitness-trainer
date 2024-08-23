from cProfile import label
from msilib import Table
import time
from turtle import right
import cv2
import mediapipe as mp
import numpy as np
import json
import streamlit as st
import pandas as pd
 
cong_gif = r'C:\Users\z004vnwk\Downloads\A.gif'
thanks_gif = r"C:\Users\z004vnwk\Downloads\ted-cute.gif"
#  camera index at 145
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
style='''
<style>
.metric-container{
background-color:#f0f2f6
border-radius:10px
padding: 10px
</style>
}
'''
 
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def save_rep_counts(person_name, left_count, right_count,result_placeholder1,result_placeholder2,result_placeholder3):
    try:
        with open('rep_counts.json', 'r') as file:
            rep_data = json.load(file)
    except FileNotFoundError:
        rep_data = {}
   
    rep_data[person_name] = {'Left Reps': left_count, 'Right Reps': right_count}
    sumation=left_count+right_count
    st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {font-size: 50px;}
        [data-testid='stMetric']{
              background-color: rgba(28, 131, 225, 0.1);
           border: 1px solid rgba(28, 131, 225, 0.1);
           padding: 5% 5% 5% 10%;
           border-radius: 5px;
           color: rgb(30, 103, 119);
           overflow-wrap: break-word;
              }
        [data-testid='stMetricLable']{ font-size:200px;}
        </style>
        """,
            unsafe_allow_html=True,
        )
    
    result_placeholder2.metric(label=person_name+"- Left hand reps",value=left_count)
    
    result_placeholder1.metric(label=person_name+"- Right hand reps",value=right_count)
    
    result_placeholder3.metric(label=person_name+"- Both hand reps",value=sumation)
    
    with open('rep_counts.json', 'w') as file:
        json.dump(rep_data, file)
    

def get_highest_rep_count(table_placeholder,person_name,col2,end_button):
    try:
        with open('rep_counts.json', 'r') as file:
            rep_data = json.load(file)
    except FileNotFoundError:
        st.info('File')
        return None
    except json.JSONDecodeError:
        with col2:
            st.info('Welcome...You are the first person to start')
 
    if person_name and end_button:   
        # Calculate the sum of reps for each person
        sum_reps = {}
        for person, reps in rep_data.items():
            sum_reps[person] = reps['Left Reps'] + reps['Right Reps']
        sum_reps[person_name] = reps['Left Reps'] + reps['Right Reps']
    
        # Sort the persons based on the sum of reps
        sorted_persons = sorted(sum_reps.items(), key=lambda x: x[1], reverse=True)

        # Get the top 5 persons
        top_5 = sorted_persons[:5]
        df=pd.DataFrame(top_5,columns=['Person','Total Reps'])
        df.index += 1
        df.index.rename="Rank"
        st.markdown(''' 
        <style>
         [data-testid="stTable"]{font-size:x-large;}  
        </style> ''', unsafe_allow_html=True,)
        table_placeholder.table(df.head())
       
        # If the current person is in the top 5 ranks, print a notification
        with col2:
            if person_name in df['Person'].head(5).tolist():
                    st.info(f"Your total reps: {sum_reps[person_name]}",icon='👤')  
                    st.balloons()    
                    st.success(f"Congradulations!! {person_name} you are one among the top 5 ranks!",icon='💐')
                    st.image(cong_gif,use_column_width='always')
                    print(f"Congradulations!! {person} is in the top 5 ranks!")
                    
            else:
                st.info(f"Your total reps: {sum_reps[person_name]}",icon='👤') 
                st.info(f"{person_name},apperciated for your best participation!!Thank you",icon='👏')
                st.image(thanks_gif,use_column_width='always')
        
    
    highest_rep_name = max(rep_data, key=lambda k: rep_data[k]['Left Reps'] + rep_data[k]['Right Reps'])
    return highest_rep_name
\
def main():
    st.set_page_config(page_title="Curls-Challenge",layout='wide')
    col1,col2,col3,col4=st.columns([2,3,1,1])
    
    with col2:     
        frame_placeholder=st.empty()
    with col3:
        result_placeholder1=st.empty()
        result_placeholder3=st.empty()
    with col4:
        result_placeholder2=st.empty()
    with col1:
        st.header('Curls-Challenge')
        table_placeholder=st.empty()
    
    with col1:
        name=st.text_input("YOUR EMAIL_ID")
        person_name=name.upper()
        start_button=st.button('START')
        end_button=st.button('END')
    if person_name and start_button:
        cap = cv2.VideoCapture(0)
    
        counter_left = 0 
        counter_right = 0 
    
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame,channels='RGB')

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Left elbow
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Right elbow
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Rep counting logic for left elbow
                    if left_angle > 90:
                        stage_left = "up"
                    if left_angle < 45 and stage_left =='up':
                        stage_left="down"
                        counter_left += 1
                        print(f"{name} left elbow rep count: {counter_left}")

                    # Rep counting logic for right elbow
                    if right_angle > 90:
                        stage_right = "up"
                    if right_angle < 45 and stage_right =='up':
                        stage_right="down"
                        counter_right += 1
                        print(f"{name} right elbow rep count: {counter_right}")

                except:
                    pass

                # Store rep counts
                save_rep_counts(person_name, counter_left, counter_right,result_placeholder1,result_placeholder2,result_placeholder3)

                cv2.rectangle(image, (0,0), (250,100), (0,0,0), -1)        

                cv2.putText(image, 'LEFT REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_left), 
                            (10,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                cv2.putText(image, 'RIGHT REPS', (200,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter_right), 
                            (200,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                           mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                          )               

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q') and end_button:
                    st.rerun()
                    break
            cap.release()
            cv2.destroyAllWindows()
            

    # Get the name of the person with the highest rep count
    highest_rep_name = get_highest_rep_count(table_placeholder,person_name,col2,end_button)
    if highest_rep_name:
        print(f"The person with the highest rep count is: {highest_rep_name}")
    else:
        print("No rep counts found.")
    
if __name__ == "__main__":
    main()