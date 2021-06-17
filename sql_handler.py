from json import load
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import Integer, String
from sqlalchemy.orm import sessionmaker
import base64
import numpy as np


Base = declarative_base()



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


def array_to_base64(face_array):
    return base64.b64encode(face_array)


def base64_to_array(b64_string, dtype=np.float32):
    b64_decoded = base64.b64decode(b64_string)
    return np.frombuffer(b64_decoded, dtype)


def combine_base64(base64_strings, delimiter='@'):
    decoded = list(map(lambda x: x.decode(), base64_strings))
    return delimiter.join(decoded)


def split_base64(one_big_base64, delimiter='@'):
    split = one_big_base64.split(delimiter)
    encoded = list(map(lambda x: x.encode(), split))
    return encoded


def create_mysql_database(engine, dbname):
    try:
        engine.execute(f"CREATE DATABASE {dbname}")
    except:
        pass
    engine.execute(f"USE {dbname}")


def get_all_users(sess):
    users = sess.query(User).all()
    return users


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
