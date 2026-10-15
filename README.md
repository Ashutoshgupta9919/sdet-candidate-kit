# Employee Attendance & Analytics API

FastAPI and MongoDB implementation for the HRone Software Engineer Trainee assignment.

## Requirements

- Python 3.11 or newer
- MongoDB 6.0 or newer

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:MONGO_URI = "mongodb://localhost:27017"
$env:MONGO_DB = "attendance_db"
uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. MongoDB must be running, or `MONGO_URI` must point to an available MongoDB instance.

