from json import load
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import Integer, String
from sqlalchemy.orm import sessionmaker
import base64
import numpy as np


Base = declarative_base()


# The user model, which translates to the table in the mysql database
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=False)
    email = Column(String(64), nullable=False)
    occupation = Column(String(64), nullable=False)
    age = Column(String(64), nullable=False)
    username = Column(String(64), nullable=False, unique=True)
    password = Column(String(128), nullable=False)
    faces = Column(String(1000e3), nullable=False) # face encodings

    def __repr__(self):
        return f"<User Name={self.name} id={self.id} email={self.email} password={self.password}>"


# Converting a face encoding to a user base64 string so that it can be stored in the databse
def array_to_base64(face_array):
    return base64.b64encode(face_array)

# face encoding is saved in the DB in base64 form, this converts it back to a numpy array
def base64_to_array(b64_string, dtype=np.float32):
    b64_decoded = base64.b64decode(b64_string)
    return np.frombuffer(b64_decoded, dtype)


# Combine multiple face encoding arrays to a single string with a delimiter
def combine_base64(base64_strings, delimiter='@'):
    decoded = list(map(lambda x: x.decode(), base64_strings))
    return delimiter.join(decoded)


# Splits the combined base64 string to the multiple face encoding arrays
def split_base64(one_big_base64, delimiter='@'):
    split = one_big_base64.split(delimiter)
    encoded = list(map(lambda x: x.encode(), split))
    return encoded


# Creates a Databse if one doesn't exist
def create_mysql_database(engine, dbname):
    try:
        engine.execute(f"CREATE DATABASE {dbname}")
    except:
        pass
    engine.execute(f"USE {dbname}")


# Get all the users and there info from the database
def get_all_users(sess):
    users = sess.query(User).all()
    return users


# Initializes databse models and session etc
def sql_init():
    with open('credentials.json', 'r') as f:
        credentials = load(f)

    uri = f"mysql://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}"
    engine = sqlalchemy.create_engine(uri, echo=False)
    create_mysql_database(engine, 'users')
    try:
        Base.metadata.create_all(engine)
    except:
        pass
    Session = sessionmaker(bind=engine)
    sql_session = Session()
    return sql_session
