# Overview

This is a clinic app that provides the services for the patient required by the project.
**Note**: The Face recognition process docs are in 'Face recognition.md'

## Fuctional requirenments:

1. User **Login** / **Signup** system with persistency through the use of a **MySQL** database.
2. User can signup, his face identity will be captured and stored in the database.
3. User can login via user name and password.
4. User can login via face recognition.
5. User can choose among different services.
6. User must pay for the service via face recognition.
7. User can navigate through the different pages of the app.

## UI and using the app:

### First Page

1. **Login**: User can login via user name and password or face recognition.
2. **Register**: User can signup, his face identity will be captured and stored in the database.

### Registeraion Page

1. User will enter their information (name, email, etc).
2. 10 images of the user will be captured and their encoding will be stored in the database.
3. After registration, user will be redirected to the First page where they can login.

### Login Page

1. **Login via username and password.**
2. **Login via face recognition.**
3. If the login fails, user will be redirected to the Login Page, otherwise, user will be redirected to the Main page.

### Main Page

This is where the services will be listed.
Once the use selects a service, the user will be asked to reconfirm, once reconfirmed, the user's identity will be reconfirmed.
If the user's identity is confirmed the payment will be successfull, otherwise, they'll be redirected to the Login Page as the account maybe compromized.
