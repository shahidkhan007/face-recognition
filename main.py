import sqlalchemy
import os

# These three lines disables the tensorflow warnings and loggings for clean environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True

from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from time import sleep
from sql_handler import User, array_to_base64, combine_base64, get_all_users, sql_init
import cv2
from collections import namedtuple
from face_recognition import detect_single_face, draw_features, face_recognition, img_to_encoding

# Responsible for the color messages in the terminal
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)



Props = namedtuple("Props", "frame ret feed")
detector = MTCNN()
facenet_model = FaceNet()
sql_session = sql_init()
all_users = get_all_users(sql_session)


# A generic choice input function
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


# The fiorst page of the program as per the requirements
def first_page():
    choice = get_choice(['1', '2'], ['1. Login', '2. Register'])
    if choice == '1':
        login_page()
    
    elif choice == '2':
        register_page()


# The user registration page
def register_page(show_scan=True, draw_face_features=True):
    global all_users, sql_session
    name = input("Name: ")
    email = input("Email: ")
    occupation = input("Occupation: ")
    age = input("Age: ")
    username = input("Username: ")
    password = input("Password: ")
    full_encoding = []

    # cv2 video feed
    feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # the number of face captured
    counter = 0

    while True:
        # Capture only 10 images
        if counter > 9:
            break

        # Grab a single frame of video
        ret, frame = feed.read()

        # Get the face in frame if any
        face_array, features, is_face = detect_single_face(detector, frame)

        # If face is detected
        if is_face:

            # Draws the features onto the frame
            if draw_face_features:
                frame = draw_features(frame, features)
            
            # Show the scanning window
            if show_scan:
                cv2.imshow("Register", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    main_page()
        
            encoding = img_to_encoding(facenet_model, face_array)
            full_encoding.append(array_to_base64(encoding))
            counter += 1

        
        info(f"Captured {counter} images of 10")
        sleep(.25)
    
    # Combine all the face encodings in the list to a single string to be saved in the db
    encode_string = combine_base64(full_encoding)

    # Creating the user
    user = User(
        name=name,
        email=email,
        occupation=occupation,
        age=age,
        username=username,
        password=password,
        faces=encode_string
    )

    # Checks if the user already exists in the database, if not, creates it
    try:

        sql_session.add(user)
        sql_session.commit()
        all_users = get_all_users(sql_session)

        success("USER CREATED SUCCESSFULLY")
    except sqlalchemy.exc.IntegrityError:
        # Rollback in case the user exists
        sql_session.rollback()     
        error("Username already exists!")
    
    except Exception as e:
        error("Something went wrong!")

    # all_users = get_all_users(sql_session)

    # Releasing the feed
    feed.release()
    cv2.destroyAllWindows()

    # Redirecting to first page
    first_page()


# The user login page, either with username and passowrd or face recognition
def login_page():
    # Get the choice on how the user wants to login
    choice = get_choice(
            choices=['1', '2', '3'],
            prompts=[
                '1. Login with username and password',
                '2. Login with face recognition',
                '3. Back'
                ]
            )
    
    if choice == '1':
        # Simple authentication if the user chooses to login via username and password
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
        # Face recognition login if the user chooses to login via face recognition
        user = face_recognition(detector, facenet_model, all_users)

        # User wasn't found, error message
        if user == None:
            error("Couldn't confirm Identity, please use username and password Or signup")
            login_page()
        
        # User was found, success message and redirectt to main page
        else:
            success("IDENTITY CONFIRMED: ", user.name)
            main_page(user)
    else:
        # Redirect to first page if the user wants to go back
        first_page()
            

# The main page where you can buy services and stuff
def main_page(logged_user):
    # Choose the service
    choice = get_choice(
        ['1', '2', '3', '4'],
        prompts=[
            '1. Consultation',
            '2. Consultaion + Check-Up',
            '3. X-Ray',
            '4. Back'
        ]
    )

    # If the user chooses to buy a service, reconfirm it's the same user
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
    
    # When the user chooses to buy a service, we need to check if the user is the same user
    if final_choice == 'y':
        user = face_recognition(detector, facenet_model, all_users)

        # User identity confirmed, make the payment and redirect to the main page
        if logged_user == user:

            success("Payment Confirmed")
            main_page(logged_user)

        # User identity not confirmed, error message and redirect to login page as the account is maybe compromised
        else:
            error("Payment Failed")
            login_page()
    else:
        # If the user doesn't want to buy a service, redirect to the main page
        main_page(logged_user)
    

# Basic helper function to log success, info and error messages


def success(*msg):
    print(Back.LIGHTGREEN_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


def info(*msg):
    print(Back.LIGHTBLUE_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


def error(*msg):
    print(Back.LIGHTRED_EX + Fore.WHITE + Style.NORMAL + ' '.join(msg))


# Entry point for the program
def interface():
    success("Welcome to the Store.")
    info("Enter 999 at any choice input to exit.")
    first_page()


if __name__ == '__main__':
    interface()
