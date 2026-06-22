import os
import re
from typing import Any, Dict

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from scipy.sparse import hstack

from .inference import predict_job_post


# Ensure Flask can find templates from the top-level templates/ folder.
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"), static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# --- Minimal auth layer (kept local to src/app.py for production readiness) ---
import sqlite3


DB_PATH = os.environ.get("JOB_FRAUD_DB_PATH", "users.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


class User(UserMixin):
    def __init__(self, id: int, username: str, password_hash: str):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id: str):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = c.fetchone()
        conn.close()
        if not user:
            return None
        return User(id=user[0], username=user[1], password_hash=user[2])

    @staticmethod
    def create(username: str, password: str):
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            conn.commit()
            user_id = c.lastrowid
            return User(id=user_id, username=username, password_hash=password_hash)
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    @staticmethod
    def authenticate(username: str, password: str):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if not user:
            return None
        if check_password_hash(user[2], password):
            return User(id=user[0], username=user[1], password_hash=user[2])
        return None


@login_manager.user_loader
def load_user(user_id: str):
    return User.get(user_id)


init_db()


@app.route("/")
def home():
    if current_user.is_authenticated:
        return render_template("home.html")
    return redirect(url_for("register"))


@app.route("/predict-page")
@login_required
def predict_page():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.create(username, password)
        if user:
            login_user(user)
            return redirect(url_for("home"))
        flash("Username already exists")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.authenticate(username, password)
        if user:
            login_user(user)
            return redirect(url_for("home"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    # Static portfolio dashboard; dynamic counters removed for correctness.
    return render_template(
        "dashboard.html",
        total_posts_checked=0,
        fraudulent_posts=0,
        non_fraudulent_posts=0,
        fraud_percentage=0,
    )


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    job_description = request.form.get("job_description", "")
    location = request.form.get("location", "")
    salary = request.form.get("salary", "")
    role = request.form.get("role", "")

    try:
        result = predict_job_post(
            job_description=job_description,
            location=location,
            salary=salary,
            role=role,
        )

        extracted = result.extracted_features
        suspicious = result.suspicious_contributions

        return render_template(
            "index.html",
            prediction=result.prediction,
            confidence=result.confidence,
            fraud_score=result.fraud_score,
            explanation=None,
            location_suggestions=extracted.get("location_suggestions"),
            extracted_features=extracted,
            suspicious_contributions=suspicious,
            model_top_features=result.model_top_features,
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error",
            confidence=0,
            explanation="Model inference failed.",
            error_message=str(e),
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)

