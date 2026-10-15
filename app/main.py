"""Employee Attendance & Analytics API.

Run from the project root with ``uvicorn app.main:app --port 8000``.
MongoDB configuration comes from ``MONGO_URI`` and ``MONGO_DB``.
"""

import calendar
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Annotated, Literal, NoReturn, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000
PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")
client = MongoClient(
    MONGO_URI,
    tz_aware=True,
    tzinfo=UTC,
    serverSelectionTimeoutMS=3_000,
)
db = client[MONGO_DB]

_indexes_ready = False
_indexes_lock = threading.Lock()


def _ensure_indexes() -> None:
    """Create the indexes used by writes, list filters, and aggregations."""
    global _indexes_ready
    if _indexes_ready:
        return

    with _indexes_lock:
        if _indexes_ready:
            return

        db.employees.create_index(
            [("emp_code", ASCENDING)], unique=True, name="uq_employees_emp_code"
        )
        db.employees.create_index(
            [("department", ASCENDING), ("emp_code", ASCENDING)],
            name="ix_employees_department_emp_code",
        )
        db.employees.create_index(
            [("department", ASCENDING), ("joined_on", ASCENDING), ("emp_code", ASCENDING)],
            name="ix_employees_department_joined_emp_code",
        )
        db.employees.create_index(
            [("joined_on", ASCENDING), ("department", ASCENDING), ("emp_code", ASCENDING)],
            name="ix_employees_joined_department_emp_code",
        )
        db.attendance_logs.create_index(
            [("emp_code", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="uq_attendance_emp_code_date",
        )
        db.attendance_logs.create_index(
            [("date", DESCENDING), ("emp_code", ASCENDING)],
            name="ix_attendance_date_emp_code",
        )
        db.attendance_logs.create_index(
            [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
            name="ix_attendance_status_date_emp_code",
        )
        db.attendance_logs.create_index(
            [("date", ASCENDING), ("late_minutes", ASCENDING), ("emp_code", ASCENDING)],
            name="ix_attendance_date_late_emp_code",
        )
        _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _ensure_indexes()
    except PyMongoError:
        # Keep the process alive so /health can report that MongoDB is unavailable
        # and retry index creation when the database comes back.
        logger.warning("MongoDB indexes are not ready; readiness checks will retry")
    yield
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


class AttendanceStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatus(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


EpochMillis = Annotated[
    StrictInt,
    Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS),
]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class EmployeeIn(RequestModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120)
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @model_validator(mode="after")
    def validate_employee(self) -> "EmployeeIn":
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", self.email):
            raise ValueError("email must be a valid email address")
        _parse_date(self.joined_on)
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInIn(RequestModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None
    status: PresenceStatus = PresenceStatus.PRESENT

    @model_validator(mode="after")
    def reject_null_timestamp(self) -> "PunchInIn":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at must be omitted or an integer epoch-millisecond value")
        return self


class PunchOutIn(RequestModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None

    @model_validator(mode="after")
    def reject_null_timestamp(self) -> "PunchOutIn":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at must be omitted or an integer epoch-millisecond value")
        return self


class RegularizeIn(RequestModel):
    status: Optional[AttendanceStatus] = None
    punch_in: Optional[EpochMillis] = None
    punch_out: Optional[EpochMillis] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def reject_null_changes(self) -> "RegularizeIn":
        has_null_change = any(
            name in self.model_fields_set and getattr(self, name) is None
            for name in ("status", "punch_in", "punch_out")
        )
        if has_null_change:
            raise ValueError("changed fields cannot be null")
        return self


# --------------------------------------------------------------------------- #
# Time and serialization helpers
# --------------------------------------------------------------------------- #
def _parse_date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be a valid calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD")
    return parsed


def _parse_month(value: str) -> tuple[date, date]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError("month must use YYYY-MM")
    year, month = map(int, value.split("-"))
    end_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, end_day)


def _parse_shift_time(value: str) -> time:
    if not isinstance(value, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("shift time must use HH:MM")
    hour, minute = map(int, value.split(":"))
    return time(hour, minute)


def _truncate_instant(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def _epoch_ms_to_datetime(value: int) -> datetime:
    if type(value) is not int or not MIN_EPOCH_MS <= value <= MAX_EPOCH_MS:
        raise ValueError("timestamp must be an integer epoch-millisecond value in range")
    return datetime.fromtimestamp(value // 1000, tz=UTC)


def _datetime_to_epoch_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    normalized = _truncate_instant(value)
    delta = normalized - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> date:
    local = _truncate_instant(punch_in).astimezone(IST)
    local_date = local.date()
    start_time = _parse_shift_time(shift_start)
    end_time = _parse_shift_time(shift_end)
    is_overnight = end_time <= start_time
    if is_overnight and local.time().replace(tzinfo=None) < end_time:
        return local_date - timedelta(days=1)
    return local_date


def compute_late_minutes(punch_in: datetime, shift_start: str, attendance_day: date) -> int:
    punch = _truncate_instant(punch_in).astimezone(IST)
    start = datetime.combine(attendance_day, _parse_shift_time(shift_start), tzinfo=IST)
    elapsed = punch - start
    if elapsed <= timedelta(minutes=10):
        return 0
    return max(0, int(elapsed.total_seconds() // 60))


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    elapsed = _truncate_instant(punch_out) - _truncate_instant(punch_in)
    seconds = elapsed.days * 86_400 + elapsed.seconds
    hours = (Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(hours)


def compute_overtime(punch_out: datetime, shift_end: str, attendance_day: date, shift_start: str) -> int:
    end_day = attendance_day
    if _parse_shift_time(shift_end) <= _parse_shift_time(shift_start):
        end_day += timedelta(days=1)
    shift_end_at = datetime.combine(end_day, _parse_shift_time(shift_end), tzinfo=IST)
    elapsed_seconds = int((_truncate_instant(punch_out).astimezone(IST) - shift_end_at).total_seconds())
    if elapsed_seconds < 30 * 60:
        return 0
    return elapsed_seconds // 60


def _to_stored_datetime(value: Optional[int]) -> Optional[datetime]:
    return _epoch_ms_to_datetime(value) if value is not None else None


def _serialize_employee(document: dict) -> dict:
    result = {key: value for key, value in document.items() if key != "_id"}
    result["created_at"] = _datetime_to_epoch_ms(result.get("created_at"))
    return result


def _serialize_history(history: Optional[list]) -> list:
    serialized = []
    for entry in history or []:
        changes = {}
        for field, values in (entry.get("changes") or {}).items():
            change = dict(values)
            if field in ("punch_in", "punch_out"):
                change["from"] = _datetime_to_epoch_ms(change.get("from"))
                change["to"] = _datetime_to_epoch_ms(change.get("to"))
            changes[field] = change
        serialized.append(
            {
                "at": _datetime_to_epoch_ms(entry.get("at")),
                "by": entry.get("by", ""),
                "reason": entry.get("reason", ""),
                "changes": changes,
            }
        )
    return serialized


def _serialize_attendance(document: dict) -> dict:
    return {
        "emp_code": document["emp_code"],
        "date": document["date"],
        "status": document["status"],
        "punch_in": _datetime_to_epoch_ms(document.get("punch_in")),
        "punch_out": _datetime_to_epoch_ms(document.get("punch_out")),
        "work_hours": document.get("work_hours"),
        "late_minutes": document.get("late_minutes", 0) or 0,
        "overtime_minutes": document.get("overtime_minutes", 0) or 0,
        "half_day": document.get("half_day", False) or False,
        "history": _serialize_history(document.get("history", [])),
    }


def _http_422(message: str) -> NoReturn:
    raise HTTPException(status_code=422, detail=message)


def _validated_date(value: Optional[str], field: str) -> Optional[date]:
    if value is None:
        return None
    try:
        return _parse_date(value)
    except ValueError as exc:
        _http_422(f"{field}: {exc}")


def _validated_month(value: Optional[str]) -> tuple[date, date]:
    if value is None:
        _http_422("month is required")
    try:
        return _parse_month(value)
    except ValueError as exc:
        _http_422(str(exc))


# --------------------------------------------------------------------------- #
# MongoDB aggregation builders
# --------------------------------------------------------------------------- #
def _round_half_up_expression(value, places: int):
    factor = 10**places
    return {
        "$divide": [
            {"$floor": {"$add": [{"$multiply": [value, factor]}, 0.5]}},
            factor,
        ]
    }


def _weekday_expression(date_expression):
    day_number = {"$dayOfWeek": date_expression}
    return {"$and": [{"$gte": [day_number, 2]}, {"$lte": [day_number, 6]}]}


def _working_days_expression(start_date: date, end_date: date, joined_field: str = "$joined_on"):
    start_string = start_date.isoformat()
    end_string = end_date.isoformat()
    effective_start_string = {
        "$cond": [
            {"$gt": [joined_field, start_string]},
            joined_field,
            start_string,
        ]
    }
    effective_start = {"$dateFromString": {"dateString": effective_start_string}}
    effective_end = {"$dateFromString": {"dateString": end_string}}
    number_of_days = {
        "$add": [
            {"$dateDiff": {"startDate": effective_start, "endDate": effective_end, "unit": "day"}},
            1,
        ]
    }
    days = {
        "$map": {
            "input": {"$range": [0, number_of_days]},
            "as": "offset",
            "in": {"$dateAdd": {"startDate": effective_start, "unit": "day", "amount": "$$offset"}},
        }
    }
    return {"$size": {"$filter": {"input": days, "as": "day", "cond": _weekday_expression("$$day")}}}


def _monthly_pipeline(emp_code: str, month: str) -> list[dict]:
    start, end = _validated_month(month)
    start_string, end_string = start.isoformat(), end.isoformat()
    weekday_presence_logs = {
        "$filter": {
            "input": {"$ifNull": ["$logs", []]},
            "as": "log",
            "cond": {
                "$and": [
                    {"$in": ["$$log.status", list(PRESENCE_STATUSES)]},
                    _weekday_expression({"$dateFromString": {"dateString": "$$log.date"}}),
                ]
            },
        }
    }
    return [
        {"$match": {"emp_code": emp_code}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "let": {"month_start": start_string, "month_end": end_string},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$gte": ["$date", "$$month_start"]},
                                    {"$lte": ["$date", "$$month_end"]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "date": 1,
                            "status": 1,
                            "half_day": 1,
                            "late_minutes": 1,
                            "overtime_minutes": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$project": {
                "_id": 0,
                "emp_code": 1,
                "month": {"$literal": month},
                "working_days": _working_days_expression(start, end),
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": weekday_presence_logs,
                            "as": "log",
                            "in": {"$cond": [{"$eq": [{"$ifNull": ["$$log.half_day", False]}, True]}, 0.5, 1.0]},
                        }
                    }
                },
                "leave_days": {
                    "$size": {
                        "$filter": {
                            "input": {"$ifNull": ["$logs", []]},
                            "as": "log",
                            "cond": {"$eq": ["$$log.status", "LEAVE"]},
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": {"$ifNull": ["$logs", []]},
                            "as": "log",
                            "cond": {"$gt": [{"$ifNull": ["$$log.late_minutes", 0]}, 0]},
                        }
                    }
                },
                "total_late_minutes": {
                    "$sum": {
                        "$map": {
                            "input": {"$ifNull": ["$logs", []]},
                            "as": "log",
                            "in": {"$ifNull": ["$$log.late_minutes", 0]},
                        }
                    }
                },
                "total_overtime_minutes": {
                    "$sum": {
                        "$map": {
                            "input": {"$ifNull": ["$logs", []]},
                            "as": "log",
                            "in": {"$ifNull": ["$$log.overtime_minutes", 0]},
                        }
                    }
                },
            }
        },
        {
            "$set": {
                "attendance_pct": {
                    "$cond": [
                        {"$eq": ["$working_days", 0]},
                        None,
                        _round_half_up_expression(
                            {"$multiply": [{"$divide": ["$present_days", "$working_days"]}, 100]},
                            4,
                        ),
                    ]
                }
            }
        },
    ]


def _department_summary_pipeline(month: str, department: Optional[str]) -> list[dict]:
    start, end = _validated_month(month)
    match = {"joined_on": {"$lte": end.isoformat()}}
    if department is not None:
        match["department"] = department

    is_presence = {"$in": [{"$ifNull": ["$logs.status", ""]}, list(PRESENCE_STATUSES)]}
    weekday_log = _weekday_expression({"$dateFromString": {"dateString": "$logs.date"}})
    return [
        {"$match": match},
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "emp_codes": {"$push": "$emp_code"},
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_codes",
                "foreignField": "emp_code",
                "let": {"month_start": start.isoformat(), "month_end": end.isoformat()},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$gte": ["$date", "$$month_start"]},
                                    {"$lte": ["$date", "$$month_end"]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "date": 1,
                            "status": 1,
                            "half_day": 1,
                            "work_hours": 1,
                            "late_minutes": 1,
                            "overtime_minutes": 1,
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$_id",
                "headcount": {"$first": "$headcount"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": [is_presence, weekday_log]},
                            {"$cond": [{"$eq": [{"$ifNull": ["$logs.half_day", False]}, True]}, 0.5, 1.0]},
                            0,
                        ]
                    }
                },
                "avg_work_hours_raw": {
                    "$avg": {
                        "$cond": [
                            {"$and": [is_presence, {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]}]},
                            "$logs.work_hours",
                            None,
                        ]
                    }
                },
                "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
                "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
                "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
            }
        },
        {
            "$set": {
                "department": "$_id",
                "avg_work_hours": {
                    "$cond": [
                        {"$eq": ["$avg_work_hours_raw", None]},
                        None,
                        _round_half_up_expression("$avg_work_hours_raw", 2),
                    ]
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": 1,
                "headcount": 1,
                "present_days": 1,
                "avg_work_hours": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]


def _late_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list[dict]:
    start, end = _validated_month(month)
    pipeline = [
        {
            "$match": {
                "date": {"$gte": start.isoformat(), "$lte": end.isoformat()},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "pipeline": [{"$project": {"_id": 0, "name": 1, "department": 1}}],
                "as": "employee",
            }
        },
        {"$unwind": "$employee"},
    ]
    if department is not None:
        pipeline.append({"$match": {"employee.department": department}})
    pipeline.extend(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$employee.name",
                    "department": "$employee.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def _department_trend_pipeline(department: str, start: date, end: date) -> list[dict]:
    day_count = (end - start).days + 1
    start_datetime = datetime.combine(start, time.min, tzinfo=UTC)
    date_add_expression = {
        "$dateAdd": {"startDate": start_datetime, "unit": "day", "amount": "$$offset"}
    }
    return [
        {"$match": {"department": department}},
        {
            "$group": {
                "_id": "$department",
                "employees": {"$push": {"emp_code": "$emp_code", "joined_on": "$joined_on"}},
            }
        },
        {
            "$set": {
                "days": {
                    "$map": {
                        "input": {"$range": [0, day_count]},
                        "as": "offset",
                        "in": {
                            "date": date_add_expression,
                            "date_str": {"$dateToString": {"format": "%Y-%m-%d", "date": date_add_expression}},
                        },
                    }
                }
            }
        },
        {"$unwind": "$days"},
        {
            "$set": {
                "active_employees": {
                    "$filter": {
                        "input": "$employees",
                        "as": "employee",
                        "cond": {"$lte": ["$$employee.joined_on", "$days.date_str"]},
                    }
                }
            }
        },
        {
            "$set": {
                "days.active_codes": {
                    "$map": {"input": "$active_employees", "as": "employee", "in": "$$employee.emp_code"}
                },
                "days.headcount": {"$size": "$active_employees"},
                "days.is_working_day": _weekday_expression("$days.date"),
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "days.active_codes",
                "foreignField": "emp_code",
                "let": {"day": "$days.date_str"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$date", "$$day"]}}},
                    {"$project": {"_id": 0, "emp_code": 1, "status": 1, "half_day": 1, "late_minutes": 1}},
                ],
                "as": "day_logs",
            }
        },
        {
            "$set": {
                "days.present_count": {
                    "$sum": {
                        "$map": {
                            "input": {"$ifNull": ["$day_logs", []]},
                            "as": "log",
                            "in": {
                                "$cond": [
                                    {"$in": ["$$log.status", list(PRESENCE_STATUSES)]},
                                    {"$cond": [{"$eq": [{"$ifNull": ["$$log.half_day", False]}, True]}, 0.5, 1.0]},
                                    0,
                                ]
                            },
                        }
                    }
                },
                "days.late_count": {
                    "$sum": {
                        "$map": {
                            "input": {"$ifNull": ["$day_logs", []]},
                            "as": "log",
                            "in": {"$cond": [{"$gt": [{"$ifNull": ["$$log.late_minutes", 0]}, 0]}, 1, 0]},
                        }
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date_obj": "$days.date",
                "date": "$days.date_str",
                "is_working_day": "$days.is_working_day",
                "headcount": "$days.headcount",
                "present_count": "$days.present_count",
                "late_count": "$days.late_count",
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$days.is_working_day", {"$gt": ["$days.headcount", 0]}]},
                        _round_half_up_expression(
                            {"$multiply": [{"$divide": ["$days.present_count", "$days.headcount"]}, 100]},
                            4,
                        ),
                        None,
                    ]
                },
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date_obj": 1},
                "output": {
                    "moving_avg_raw": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$set": {
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$moving_avg_raw", None]},
                        None,
                        _round_half_up_expression("$moving_avg_raw", 4),
                    ]
                }
            }
        },
        {"$sort": {"date_obj": 1}},
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1,
            }
        },
    ]


def _attendance_filter(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[AttendanceStatus],
) -> dict:
    start = _validated_date(date_from, "date_from")
    end = _validated_date(date_to, "date_to")
    if start and end and start > end:
        _http_422("date_from must be on or before date_to")
    query = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if start or end:
        date_filter = {}
        if start:
            date_filter["$gte"] = start.isoformat()
        if end:
            date_filter["$lte"] = end.isoformat()
        query["date"] = date_filter
    if status:
        query["status"] = status.value
    return query


def _run_explain(collection_name: str, pipeline: list[dict]) -> dict:
    return db.command(
        {
            "explain": {
                "aggregate": collection_name,
                "pipeline": pipeline,
                "cursor": {},
            },
            "verbosity": "executionStats",
        }
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", responses={503: {"description": "MongoDB unavailable"}})
def health():
    try:
        client.admin.command("ping")
        _ensure_indexes()
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="MongoDB is unavailable") from exc
    return {"status": "ok"}


@app.post("/employees", status_code=201, responses={409: {"description": "Employee code already exists"}})
def create_employee(body: EmployeeIn):
    document = body.model_dump()
    document["created_at"] = _now_utc()
    try:
        db.employees.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="emp_code already exists") from exc
    return _serialize_employee(document)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    query = {"department": department} if department is not None else {}
    total = (
        db.employees.count_documents(query, hint="ix_employees_department_emp_code")
        if department is not None
        else db.employees.count_documents(query, hint="_id_")
    )
    cursor = (
        db.employees.find(query, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [_serialize_employee(document) for document in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post(
    "/attendance/punch-in",
    status_code=201,
    responses={404: {"description": "Employee not found"}, 409: {"description": "Already punched in for this date"}},
)
def punch_in(body: PunchInIn):
    employee = db.employees.find_one(
        {"emp_code": body.emp_code},
        {"_id": 0, "shift_start": 1, "shift_end": 1},
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")

    punched_at = _to_stored_datetime(body.punched_at) or _now_utc()
    attendance_day = _attendance_date(punched_at, employee["shift_start"], employee["shift_end"])
    document = {
        "emp_code": body.emp_code,
        "date": attendance_day.isoformat(),
        "status": body.status.value,
        "punch_in": punched_at,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(punched_at, employee["shift_start"], attendance_day),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="already punched in for this date") from exc
    return _serialize_attendance(document)


@app.post(
    "/attendance/punch-out",
    status_code=200,
    responses={404: {"description": "Employee or punch-in not found"}, 409: {"description": "Already punched out"}},
)
def punch_out(body: PunchOutIn):
    employee = db.employees.find_one(
        {"emp_code": body.emp_code},
        {"_id": 0, "shift_start": 1, "shift_end": 1},
    )
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")

    punched_at = _to_stored_datetime(body.punched_at) or _now_utc()
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$type": "date", "$lte": punched_at}},
        sort=[("punch_in", DESCENDING), ("date", DESCENDING)],
    )
    if record is None:
        raise HTTPException(status_code=404, detail="no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="record is already punched out")

    punch_in_at = _truncate_instant(record["punch_in"])
    elapsed = punched_at - punch_in_at
    if elapsed <= timedelta(0) or elapsed > timedelta(hours=24):
        raise HTTPException(status_code=422, detail="punched_at must be after punch_in and within 24 hours")

    attendance_day = _parse_date(record["date"])
    work_hours = compute_work_hours(punch_in_at, punched_at)
    updated = db.attendance_logs.find_one_and_update(
        {
            "_id": record["_id"],
            "punch_out": None,
            "punch_in": record["punch_in"],
        },
        {
            "$set": {
                "punch_out": punched_at,
                "work_hours": work_hours,
                "overtime_minutes": compute_overtime(
                    punched_at,
                    employee["shift_end"],
                    attendance_day,
                    employee["shift_start"],
                ),
                "half_day": work_hours < 4.5,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="record was updated by another request")
    return _serialize_attendance(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[AttendanceStatus] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    query = _attendance_filter(emp_code, date_from, date_to, status)
    total = (
        db.attendance_logs.count_documents(query)
        if query
        else db.attendance_logs.count_documents(query, hint="_id_")
    )
    cursor = (
        db.attendance_logs.find(query, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [_serialize_attendance(document) for document in cursor],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch(
    "/attendance/{emp_code}/{date}",
    responses={
        404: {"description": "Employee or attendance record not found"},
        409: {"description": "Concurrent update conflict"},
    },
)
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    try:
        attendance_day = _parse_date(date)
    except ValueError as exc:
        _http_422(str(exc))

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": attendance_day.isoformat()})
    if record is None:
        raise HTTPException(status_code=404, detail="employee or attendance record not found")
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee or attendance record not found")

    supplied = body.model_fields_set
    old_status = record["status"]
    old_punch_in = record.get("punch_in")
    old_punch_out = record.get("punch_out")
    new_status = body.status.value if body.status is not None else old_status
    if new_status in ("ABSENT", "LEAVE") and supplied.intersection({"punch_in", "punch_out"}):
        _http_422("punch times cannot be supplied with ABSENT or LEAVE")

    new_punch_in = _to_stored_datetime(body.punch_in) if "punch_in" in supplied else old_punch_in
    new_punch_out = _to_stored_datetime(body.punch_out) if "punch_out" in supplied else old_punch_out

    if new_status in ("ABSENT", "LEAVE"):
        new_punch_in = None
        new_punch_out = None
    else:
        if new_punch_in is None:
            _http_422("a presence status requires punch_in")
        if _attendance_date(new_punch_in, employee["shift_start"], employee["shift_end"]) != attendance_day:
            _http_422("punch_in must belong to the attendance date")
        if new_punch_out is not None:
            elapsed = new_punch_out - new_punch_in
            if elapsed <= timedelta(0) or elapsed > timedelta(hours=24):
                _http_422("punch_out must be after punch_in and within 24 hours")

    if new_status in ("ABSENT", "LEAVE"):
        derived = {
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
        }
    else:
        late_minutes = compute_late_minutes(new_punch_in, employee["shift_start"], attendance_day)
        if new_punch_out is None:
            derived = {
                "work_hours": None,
                "late_minutes": late_minutes,
                "overtime_minutes": 0,
                "half_day": False,
            }
        else:
            work_hours = compute_work_hours(new_punch_in, new_punch_out)
            derived = {
                "work_hours": work_hours,
                "late_minutes": late_minutes,
                "overtime_minutes": compute_overtime(
                    new_punch_out,
                    employee["shift_end"],
                    attendance_day,
                    employee["shift_start"],
                ),
                "half_day": work_hours < 4.5,
            }

    old_values = {
        "status": old_status,
        "punch_in": old_punch_in,
        "punch_out": old_punch_out,
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0) or 0,
        "overtime_minutes": record.get("overtime_minutes", 0) or 0,
        "half_day": record.get("half_day", False) or False,
    }
    new_values = {
        "status": new_status,
        "punch_in": new_punch_in,
        "punch_out": new_punch_out,
        **derived,
    }
    changes = {}
    for field, new_value in new_values.items():
        old_value = old_values[field]
        if field in ("punch_in", "punch_out") and old_value is not None:
            old_value = _truncate_instant(old_value)
        if field in ("punch_in", "punch_out") and new_value is not None:
            new_value = _truncate_instant(new_value)
        if old_value != new_value:
            changes[field] = {"from": old_value, "to": new_value}
    if not changes:
        _http_422("request does not change the record")

    history_entry = {
        "at": _now_utc(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    filter_query = {
        "_id": record["_id"],
        "status": old_status,
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }
    if "history" in record:
        filter_query["history"] = record["history"]
    else:
        filter_query["history"] = {"$exists": False}

    update = {**new_values}
    updated = db.attendance_logs.find_one_and_update(
        filter_query,
        {"$set": update, "$push": {"history": history_entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="record was updated by another request")
    return _serialize_attendance(updated)


@app.get(
    "/analytics/employees/{emp_code}/monthly",
    responses={404: {"description": "Employee not found"}},
)
def employee_monthly(emp_code: str, month: str):
    _validated_month(month)
    result = next(db.employees.aggregate(_monthly_pipeline(emp_code, month)), None)
    if result is None:
        raise HTTPException(status_code=404, detail="employee not found")
    return result


@app.get("/analytics/departments/summary")
def department_summary(month: str, department: Optional[str] = None):
    _validated_month(month)
    items = list(db.employees.aggregate(_department_summary_pipeline(month, department)))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str,
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = None,
):
    _validated_month(month)
    items = list(db.attendance_logs.aggregate(_late_leaderboard_pipeline(month, limit, department)))
    return {"month": month, "items": items}


@app.get(
    "/analytics/departments/{department}/trend",
    responses={404: {"description": "Department not found"}},
)
def department_trend(department: str, from_date: str = Query(alias="from"), to: str = Query(...)):
    start = _validated_date(from_date, "from")
    end = _validated_date(to, "to")
    if end < start:
        _http_422("to must be on or after from")
    if (end - start).days + 1 > 92:
        _http_422("date range cannot exceed 92 days")
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(status_code=404, detail="department not found")
    items = list(db.employees.aggregate(_department_trend_pipeline(department, start, end)))
    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal[
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    ],
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[AttendanceStatus] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    from_date: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = None,
):
    if endpoint == "attendance_list":
        query = _attendance_filter(emp_code, date_from, date_to, status)
        explain = db.command(
            {
                "explain": {
                    "find": "attendance_logs",
                    "filter": query,
                    "sort": {"date": DESCENDING, "emp_code": ASCENDING},
                    "skip": (page - 1) * page_size,
                    "limit": page_size,
                },
                "verbosity": "executionStats",
            }
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "employee_monthly":
        if not emp_code:
            _http_422("emp_code is required for employee_monthly")
        _validated_month(month)
        pipeline = _monthly_pipeline(emp_code, month)
        collection = "employees"
    elif endpoint == "department_summary":
        _validated_month(month)
        pipeline = _department_summary_pipeline(month, department)
        collection = "employees"
    elif endpoint == "late_leaderboard":
        _validated_month(month)
        pipeline = _late_leaderboard_pipeline(month, limit, department)
        collection = "attendance_logs"
    else:
        if not department or from_date is None or to is None:
            _http_422("department, from, and to are required for department_trend")
        start = _validated_date(from_date, "from")
        end = _validated_date(to, "to")
        if end < start:
            _http_422("to must be on or after from")
        if (end - start).days + 1 > 92:
            _http_422("date range cannot exceed 92 days")
        pipeline = _department_trend_pipeline(department, start, end)
        collection = "employees"

    return {
        "endpoint": endpoint,
        "collection": collection,
        "explain": _run_explain(collection, pipeline),
    }

