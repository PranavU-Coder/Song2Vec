from pathlib import Path

from flask import Flask, render_template

from web import register_routes

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload


@app.route("/")
def index() -> str:
    """Serve the main UI."""
    return render_template("index.html")


# Register API routes
register_routes(app)


if __name__ == "__main__":
    # Create directories if needed
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

    app.run(debug=True, host="0.0.0.0", port=5000)
