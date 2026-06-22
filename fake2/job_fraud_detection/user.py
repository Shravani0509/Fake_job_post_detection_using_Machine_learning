from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import sqlite3
import os

# Initialize database
def init_db():
    if not os.path.exists('users.db'):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     password_hash TEXT NOT NULL)''')
        conn.commit()
        conn.close()

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        if not user:
            return None
        return User(id=user[0], username=user[1], password_hash=user[2])

    @staticmethod
    def create(username, password):
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                     (username, password_hash))
            conn.commit()
            user_id = c.lastrowid
            return User(id=user_id, username=username, password_hash=password_hash)
        except sqlite3.IntegrityError:
            return None  # Username already exists
        finally:
            conn.close()

    @staticmethod
    def authenticate(username, password):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            return User(id=user[0], username=user[1], password_hash=user[2])
        return None
