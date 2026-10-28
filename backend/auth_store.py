import hashlib
import sqlite3
from pathlib import Path
from werkzeug.security import check_password_hash, generate_password_hash


class AuthStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    questions TEXT NOT NULL,
                    answers TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    @staticmethod
    def normalize_answer(value: str) -> str:
        normalized = ' '.join((value or '').strip().lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def create_user(self, name: str, email: str, password: str, questions, answers):
        email = email.strip().lower()
        if self.find_user(email):
            return False, 'An account with this email already exists.'
        if not name.strip() or not email:
            return False, 'Name and email are required.'
        if len(password) < 8:
            return False, 'Password must contain at least 8 characters.'
        with self._connect() as connection:
            connection.execute(
                'INSERT INTO users (name, email, password_hash, questions, answers, created_at) VALUES (?, ?, ?, ?, ?, datetime("now"))',
                (name.strip(), email, generate_password_hash(password), '||'.join(questions), '||'.join(self.normalize_answer(answer) for answer in answers)),
            )
        return True, ''

    def find_user(self, email: str):
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM users WHERE email = ?', (email.strip().lower(),)).fetchone()
            return dict(row) if row else None

    def authenticate(self, email: str, password: str):
        user = self.find_user(email)
        if not user or not check_password_hash(user['password_hash'], password):
            return None
        return user

    def recovery_questions(self, email: str):
        user = self.find_user(email)
        if not user:
            return []
        return user['questions'].split('||')

    def verify_recovery_answers(self, email: str, answers):
        user = self.find_user(email)
        if not user:
            return None
        saved = user['answers'].split('||')
        submitted = [self.normalize_answer(answer) for answer in answers]
        if submitted != saved:
            return None
        return user

    def update_password(self, email: str, password: str):
        with self._connect() as connection:
            connection.execute('UPDATE users SET password_hash = ? WHERE email = ?', (generate_password_hash(password), email.strip().lower()))
