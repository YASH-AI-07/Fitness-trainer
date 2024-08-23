import cv2
import mediapipe as mp
import numpy as np
import json
import streamlit as st
import pandas as pd

cong_gif = r"C:\Trainer\A.gif"
thanks_gif = r"C:\Trainer\thanks.gif"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def save_rep_counts(person_name, left_count, right_count, result_placeholder3):
    try:
        with open('rep_counts.json', 'r') as file:
            rep_data = json.load(file)
    except FileNotFoundError:
        rep_data = {}
   
    rep_data[person_name] = {'Left Reps': left_count, 'Right Reps': right_count}
    sumation = int((left_count + right_count)//2)
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
        [data-testid='stMetricLabel']{ font-size:200px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # result_placeholder2.metric(label=f"{person_name} - Left reps", value=left_count)
    # result_placeholder1.metric(label=f"{person_name} - Right reps", value=right_count)
    result_placeholder3.metric(label=f"{person_name} - Total reps", value=sumation)
    
    with open('rep_counts.json', 'w') as file:
        json.dump(rep_data, file)

def get_highest_rep_count(table_placeholder, person_name, col2, end_button):
    try:
        with open('rep_counts.json', 'r') as file:
            rep_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        st.info('No previous data found.')
        return None

    if person_name and end_button:
        sum_reps = {person: (reps['Left Reps'] + reps['Right Reps'])//2 for person, reps in rep_data.items()}
        sum_reps[person_name] = sum_reps.get(person_name, 0)

        sorted_persons = sorted(sum_reps.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_persons[:5]
        
        df = pd.DataFrame(top_5, columns=['Person', 'Total Reps'])
        df.index += 1
        df.index.name = "Rank"
        
        st.markdown('''
        <style>
        [data-testid="stTable"]{font-size:x-large;}
        </style>
        ''', unsafe_allow_html=True)
        
        table_placeholder.table(df)
       
        with col2:
            if person_name in df['Person'].head(5).tolist():
                st.info(f"Your total reps: {sum_reps[person_name]}", icon='👤')
                st.balloons()
                st.success(f"Congratulations!! {person_name}, you are in the top 5!", icon='💐')
                st.image(cong_gif, use_column_width='always')
            else:
                st.info(f"Your total reps: {sum_reps[person_name] }", icon='👤')

                st.info(f"Thank you for participating, {person_name}!", icon='👏')
                st.image(thanks_gif, use_column_width='always')

    return max(rep_data, key=lambda k: rep_data[k]['Left Reps'] + rep_data[k]['Right Reps'], default=None)

def main():
    st.set_page_config(page_title="MEN SQUAT CHALLENGE", layout='wide')
    col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
    
    with col2:
        frame_placeholder = st.empty()
    with col3:
        # result_placeholder1 = st.empty()
        result_placeholder3 = st.empty()
    # with col4:
    #     result_placeholder2 = st.empty()
    with col1:
        st.header('MEN SQUAT CHALLENGE')
        table_placeholder = st.empty()
    
    with col1:
        name = st.text_input("Your Email ID")
        person_name = name.upper()
        start_button = st.button('START')
        end_button = st.button('END')
    
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
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels='RGB')

                try:
                    landmarks = results.pose_landmarks.landmark

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
                    left_angle_hip = calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                                     left_hip, left_knee)
                    right_angle_hip = calculate_angle([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                                                      right_hip, right_knee)

                    # Rep counting logic based on knee and hip angles
                    if left_angle_knee > 170 and left_angle_hip > 170:
                        stage_left = "up"
                    if left_angle_knee < 145 and left_angle_hip < 145 and stage_left == 'up':
                        stage_left = "down"
                        counter_left += 1

                    if right_angle_knee > 170 and right_angle_hip > 170:
                        stage_right = "up"
                    if right_angle_knee < 145 and right_angle_hip < 145 and stage_right == 'up':
                        stage_right = "down"
                        counter_right += 1

                except Exception as e:
                    print(f"Error: {e}")
                    pass

                save_rep_counts(person_name, counter_left, counter_right, result_placeholder3)

                cv2.rectangle(image, (0, 0), (250, 100), (0, 0, 0), -1)
                # cv2.putText(image, f'LEFT: {left_angle_knee:.2f}, {left_angle_hip:.2f}', (15, 12), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(image, f'RIGHT: {right_angle_knee:.2f}, {right_angle_hip:.2f}', (200, 12), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q') or end_button:
                    st.rerun()
                    break

            cap.release()
            cv2.destroyAllWindows()

    highest_rep_name = get_highest_rep_count(table_placeholder, person_name, col2, end_button)
    if highest_rep_name:
        print(f"The person with the highest rep count is: {highest_rep_name}")
    else:
        print("No rep counts found.")
    
if __name__ == "__main__":
    main()
