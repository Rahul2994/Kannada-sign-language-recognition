import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
import os
from gtts import gTTS
import pyttsx3
import time



# Initialize the TTS engine
engine = pyttsx3.init()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Ka', 1: 'Kha', 2: 'Ga', 3: 'Gha', 4: 'Na', 5: 'Cha', 6: 'Chha', 7: 'Ja', 8: 'Jha', 9: 'Na', 10:'Ta', 11:'Tta', 12:'Da', 13:'Dda', 14:'Na', 15:'Tha', 16:'Ttha', 17:'Ddha', 18:'Ddha',19:'Na',20:'Pa', 21:'Pha', 22:'Ba',23:'Bha',24:'Ma', 25:'Ya', 26:'Ra', 27:'La',28:'Va',29:'Sha',30:'Shha',31:'Sa',32:'Ha',33:'Ksha',34:'Lla',35:'Thra'}
kannada_texts = {
    0: 'ಕ', 1: 'ಖ', 2: 'ಗ', 3: 'ಘ', 4: 'ಙ', 5: 'ಚ', 6: 'ಛ', 7: 'ಜ', 8: 'ಝ', 9: 'ಞ', 10: 'ಟ',
    11: 'ಠ', 12: 'ಡ', 13: 'ಢ', 14: 'ಣ', 15: 'ತ', 16: 'ಥ', 17: 'ದ', 18: 'ಧ', 19: 'ನ', 20: 'ಪ',
    21: 'ಫ', 22: 'ಬ', 23: 'ಭ', 24: 'ಮ', 25: 'ಯ', 26: 'ರ', 27: 'ಲ', 28: 'ವ', 29: 'ಶ', 30: 'ಷ',
    31: 'ಸ', 32: 'ಹ', 33: 'ಕ್ಷ', 34: 'ಳ', 35: 'ತ್ರ'
}

predicted_character = None  # Initialize predicted_character outside the loop
pre=None
def update_predicted_character():
    global predicted_character, pre

    while True:
        ret, frame = cap.read()

        if ret:  # Check if frame is retrieved successfully
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                pre=prediction
               

                predicted_character = labels_dict[int(prediction[0])]
                

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, predicted_character, (x1, y1 - 10), font, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
                break
        else:
            print("Error: Couldn't retrieve frame from camera")
            break

update_thread = threading.Thread(target=update_predicted_character)
update_thread.start()

def text_to_speech(text, language='kn', file_name='output.mp3'):
    """
    Convert text to speech using gTTS.
    
    Args:
    - text (str): The text to be converted to speech.
    - language (str): The language code. Default is Kannada ('kn').
    - file_name (str): The name of the output audio file. Default is 'output.mp3'.
    """
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(file_name)
    os.system("start " + file_name)

while True:
    if predicted_character is not None:
        print("Predicted Character:", predicted_character)
        # Speak the predicted character using TTS
        # engine.say(predicted_character)
        # engine.runAndWait()
        text_key = 'greeting'  # Key to retrieve text from the dictionary
        text = kannada_texts.get(text_key,kannada_texts[int(pre[0])])  # Get text from the dictionary
        text_to_speech(text)
        time.sleep(2)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
