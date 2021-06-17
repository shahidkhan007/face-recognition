import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from time import sleep
from sql_handler import User, array_to_base64, combine_base64, get_all_users, sql_init
import cv2
from collections import namedtuple
from face_recognition import detect_faces, face_recognition, img_to_encoding
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)



Props = namedtuple("Props", "frame ret feed")
detector = MTCNN()
facenet_model = FaceNet()
sql_session = sql_init()
all_users = get_all_users(sql_session)


def get_choice(choices, prompts=(), modifiers=()):
    print(*prompts, sep='\n')

    choice = input("Choice> ")

    for m in modifiers:
        choice = m(choice)

    if choice in choices:
        return choice
    elif choice == '999':
        exit(0)
    else:
        error("Invalid choice!\n\n\n")
        get_choice(choices, prompts, modifiers)


def first_page():
    choice = get_choice(['1', '2'], ['1. Login', '2. Register'])
    if choice == '1':
        login_page()
    
    elif choice == '2':
        register_page()


def register_page():
    global all_users, sql_session
    name = input("Name: ")
    email = input("Email: ")
    occupation = input("Occupation: ")
    age = input("Age: ")
    username = input("Username: ")
    password = input("Password: ")
    full_encoding = []

    feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    counter = 0
    while True:
        if counter > 9:
            break
        ret, frame = feed.read()

        faces = detect_faces(detector, frame)

        if len(faces) > 0:
            face_array = faces[0]
            encoding = img_to_encoding(facenet_model, face_array)
            full_encoding.append(array_to_base64(encoding))
            counter += 1
        else:
            pass
        
        info(f"Captured {counter} images of 10")
        sleep(.25)
    
    encode_string = combine_base64(full_encoding)

    user = User(
        name=name,
        email=email,
        occupation=occupation,
        age=age,
        username=username,
        password=password,
        faces=encode_string
    )

    sql_session.add(user)
    sql_session.commit()

    success("USER CREATED SUCCESSFULLY")
    
    all_users = get_all_users(sql_session)
    feed.release()
    cv2.destroyAllWindows()
    first_page()


def login_page():
    choice = get_choice(
            choices=['1', '2', '3'],
            prompts=[
                '1. Login with username and password',
                '2. Login with face recognition',
                '3. Back'
                ]
            )
    
    if choice == '1':
        username = input("Username: ")
        password = input("Password: ")


        for user in all_users:
            if user.username == username and user.password == password:
                success("Login successful!")
                main_page(user)
                break
        else:
            error("Invalid username or password!")
            login_page()

    elif choice == '2':
        user = face_recognition(detector, facenet_model, all_users)

        if user == None:
            error("Couldn't confirm Identity, please use username and password Or signup")
            login_page()
        else:
            success("IDENTITY CONFIRMED: ", user.name)
            main_page(user)
    else:
        first_page()
            

def main_page(logged_user):
    choice = get_choice(
        ['1', '2', '3', '4'],
        prompts=[
            '1. Consultation',
            '2. Consultaion + Check-Up',
            '3. X-Ray',
            '4. Back'
        ]
    )

    final_choice = ''
    if choice == '1':
        final_choice = get_choice(
            choices=['y', 'n'],
            prompts=["You are selecting Cosultation are you sure (Y/N)?"],
            modifiers=[lambda x: x.lower()]
            )

    elif choice == '2':
        final_choice = get_choice(
            choices=['y', 'n'],
            prompts=["You are selecting Cosultation + Check-Up are you sure (Y/N)?"],
            modifiers=[lambda x: x.lower()]
            )
    elif choice == '3':
        final_choice = get_choice(
            choices=['y', 'n'],
            prompts=["Your are selecting X-Ray are you sure (Y/N)?"],
            modifiers=[lambda x: x[0].lower()]
            )
    else:
        login_page()
    
    if final_choice == 'y':
        user = face_recognition(detector, facenet_model, all_users)
        if logged_user == user:
            success("Payment Confirmed")
            main_page(logged_user)
        else:
            error("Payment Failed")
            login_page()
    else:
        main_page(logged_user)
    

def success(*msg):
    print(Back.LIGHTGREEN_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


def info(*msg):
    print(Back.LIGHTBLUE_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


def error(*msg):
    print(Back.LIGHTRED_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


def interface():
    success("Welcome to the Store.")
    info("Enter 999 at any choice input to exit.")
    first_page()


if __name__ == '__main__':
    interface()
