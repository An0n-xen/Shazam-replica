from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
from pathlib import Path
import time
import base64
import threading
import glob
from datetime import datetime, timedelta


from utils.database import FingerprintDatabase
from utils.fingerprint import fingerprint_audio
from utils.matcher import match_query
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "./data/uploads"
app.config["DATABASE_PATH"] = "./data/db/fingerprint_database.pkl"
app.config["NEW_SONGS_FOLDER"] = "./data/db_tracks"

Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)
Path(app.config["NEW_SONGS_FOLDER"]).mkdir(exist_ok=True)

db = None


def load_database():
    global db
    try:
        db = FingerprintDatabase.load(app.config["DATABASE_PATH"])
        logger.info(f"✓ Database loaded: {len(db.song_metadata)} songs")
        return True
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return False


        return False


def cleanup_old_files():
    """Background task to delete audio files older than 30 minutes."""
    while True:
        try:
            folder = app.config["NEW_SONGS_FOLDER"]
            # Look for audio files
            files = []
            for ext in ["*.mp3", "*.wav", "*.m4a"]:
                files.extend(glob.glob(os.path.join(folder, ext)))
            
            logger.info(f"🧹 Running cleanup task. Checking {len(files)} files...")
            
            cutoff_time = time.time() - (30 * 60) # 30 minutes ago
            deleted_count = 0
            
            for file_path in files:
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"Deleted old file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error checking/deleting {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✓ Cleanup complete. Deleted {deleted_count} files.")
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        # Sleep for 30 minutes before next run
        # For testing purposes, you might want to check more frequently, e.g. every minute
        # but the requirement is "every 30 mins the song files get deleted"
        time.sleep(30 * 60) 


def identify_audio_file(filepath):
    try:
        start_time = time.time()
        query_hashes, metadata = fingerprint_audio(filepath, visualize=False)
        matches = match_query(query_hashes, db, return_top_n=5, visualize=False)
        elapsed = time.time() - start_time

        if matches and len(matches) > 0:
            best = matches[0]
            return {
                "success": True,
                "match": {
                    "title": best["song_info"]["title"],
                    "artist": best["song_info"]["artist"],
                    "score": best["score"],
                    "confidence": best["confidence"],
                    "offset": round(best["offset"], 2),
                    "offset": round(best["offset"], 2),
                    "song_id": best["song_id"],
                    "audio_available": os.path.exists(best["song_info"].get("filepath", "")),
                },
                "alternatives": [
                    {
                        "title": m["song_info"]["title"],
                        "artist": m["song_info"]["artist"],
                        "score": m["score"],
                    }
                    for m in matches[1:5]
                ],
                "query_time": round(elapsed * 1000, 1),
                "num_hashes": len(query_hashes),
            }
        else:
            return {
                "success": False,
                "message": "No match found",
                "query_time": round(elapsed * 1000, 1),
            }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"success": False, "message": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    if db is None:
        return jsonify({"status": "error", "message": "Database not loaded"}), 500
    stats = db.get_stats()
    return jsonify(
        {
            "status": "ok",
            "database": {
                "num_songs": stats["num_songs"],
                "songs": [
                    {"id": s["song_id"], "title": s["title"], "artist": s["artist"]}
                    for s in db.get_all_songs()
                ],
            },
        }
    )


@app.route("/api/identify", methods=["POST"])
def api_identify():
    if db is None:
        return jsonify({"success": False, "message": "Database not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file"}), 400

    file = request.files["file"]
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"], f"upload_{time.time()}_{file.filename}"
    )

    try:
        file.save(filepath)
        result = identify_audio_file(filepath)
        os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/record", methods=["POST"])
def api_record():
    if db is None:
        return jsonify({"success": False, "message": "Database not loaded"}), 500

    try:
        data = request.get_json()
        audio_data = (
            data["audio_data"].split(",")[1]
            if "," in data["audio_data"]
            else data["audio_data"]
        )
        audio_bytes = base64.b64decode(audio_data)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"recording_{time.time()}.wav")
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        result = identify_audio_file(filepath)
        os.remove(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/add-song", methods=["POST"])
def api_add_song():
    if db is None:
        return jsonify({"success": False, "message": "Database not loaded"}), 500

    try:
        file = request.files["file"]
        title = request.form.get("title", file.filename)
        artist = request.form.get("artist", "Unknown")

        filepath = os.path.join(app.config["NEW_SONGS_FOLDER"], file.filename)
        file.save(filepath)

        hashes, metadata = fingerprint_audio(filepath, visualize=False)
        song_id = db.add_song(
            hashes,
            filepath,
            {"title": title, "artist": artist, "duration": metadata.get("duration", 0)},
        )
        db.save(app.config["DATABASE_PATH"])

        return jsonify(
            {
                "success": True,
                "song_id": song_id,
                "title": title,
                "artist": artist,
                "num_hashes": len(hashes),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/audio/<int:song_id>")
def serve_audio(song_id):
    """Serve audio file for playback after identification."""
    if db is None:
        return jsonify({"error": "Database not loaded"}), 500
    
    try:
        song_info = db.song_metadata.get(song_id)
        if song_info is None:
            return jsonify({"error": "Song not found"}), 404
        
        file_path = song_info.get("filepath")
        if file_path and os.path.exists(file_path):
            return send_file(file_path, mimetype="audio/mpeg")
        else:
            return jsonify({"error": "Audio file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():

    if not load_database():
        logger.warning("⚠️  Database not loaded - build it first!")

    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    logger.info("🕒 Cleanup background task started (runs every 30 mins)")

    print("\n🌐 Starting web server at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
