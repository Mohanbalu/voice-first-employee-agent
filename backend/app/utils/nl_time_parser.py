"""Natural Language Time and Scheduling Expression Parser.

Deterministic, robust parsing of:
- Relative time ("in 30 minutes", "in 2 hours")
- Times of day ("at 5 PM", "at 10:30 AM", "at 9 AM", "morning", "afternoon", "evening", "tonight")
- Calendar days ("today", "tomorrow", "next Monday", "this Friday")
- Recurrences ("every Monday at 9 AM", "every day", "daily", "weekly", "every month")
- Task/reminder content extraction
- Ambiguity detection with targeted follow-up clarification prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

# Configurable project default timezone (matches HCL campus office location in India)
DEFAULT_TIMEZONE_NAME = "Asia/Kolkata"


@dataclass
class ParsedSchedule:
    """Structured output from natural language schedule parsing."""
    title: str
    scheduled_at: Optional[datetime]
    timezone_name: str
    recurrence_type: str  # "NONE" | "DAILY" | "WEEKLY" | "MONTHLY" | "WEEKDAYS"
    recurrence_rule: Optional[str]
    reminder_type: str  # "NOTIFICATION" | "TASK" | "MEETING" | "TIMESHEET"
    duration_minutes: int
    is_ambiguous: bool
    clarification_question: Optional[str]
    raw_query: str

    @property
    def requires_clarification(self) -> bool:
        return self.is_ambiguous

    @property
    def clean_title(self) -> str:
        return self.title


def get_default_tz(tz_name: Optional[str] = None) -> ZoneInfo:
    """Returns ZoneInfo object, falling back safely to UTC if invalid."""
    try:
        return ZoneInfo(tz_name or DEFAULT_TIMEZONE_NAME)
    except Exception:
        return ZoneInfo("UTC")


def parse_schedule_request(
    text: str,
    base_time: Optional[datetime] = None,
    tz_name: Optional[str] = None,
) -> ParsedSchedule:
    """Parses a natural language scheduling or reminder request.

    Args:
        text: User query string (e.g. "Remind me at 5 PM to submit my timesheet")
        base_time: Reference datetime (defaults to current time in tz_name)
        tz_name: Target timezone string (e.g. "Asia/Kolkata")

    Returns:
        ParsedSchedule dataclass with datetime, recurrence, and clean task title.
    """
    raw_query = text.strip()
    active_tz = get_default_tz(tz_name)
    now = base_time or datetime.now(active_tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=active_tz)

    lowered = raw_query.lower()

    # ── 1. Reminder type classification ─────────────────────────────────────
    reminder_type = "NOTIFICATION"
    if "timesheet" in lowered or "log hours" in lowered:
        reminder_type = "TIMESHEET"
    elif "meeting" in lowered or "call" in lowered or "sync" in lowered or "discussion" in lowered:
        reminder_type = "MEETING"
    elif "task" in lowered or "to-do" in lowered or "todo" in lowered:
        reminder_type = "TASK"

    # ── 2. Recurrence extraction ─────────────────────────────────────────────
    recurrence_type = "NONE"
    recurrence_rule = None

    # Check for specific days first e.g. "every monday", "every friday"
    days_map = {
        "monday": "MO",
        "tuesday": "TU",
        "wednesday": "WE",
        "thursday": "TH",
        "friday": "FR",
        "saturday": "SA",
        "sunday": "SU",
    }
    matched_day = False
    for day_name, day_code in days_map.items():
        if re.search(rf"\bevery\s+{day_name}\b", lowered):
            recurrence_type = "WEEKLY"
            recurrence_rule = f"FREQ=WEEKLY;BYDAY={day_code}"
            matched_day = True
            break

    if not matched_day:
        if re.search(r"\bevery\s*day\b|\bdaily\b", lowered):
            recurrence_type = "DAILY"
            recurrence_rule = "FREQ=DAILY"
        elif re.search(r"\bevery\s+(?:week\s+)?day\b|\bevery\s+monday\s+to\s+friday\b|\bweekdays\b", lowered):
            recurrence_type = "WEEKDAYS"
            recurrence_rule = "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
        elif re.search(r"\bevery\s+week\b|\bweekly\s+(?:reminder|schedule|meeting)\b", lowered):
            recurrence_type = "WEEKLY"
            recurrence_rule = "FREQ=WEEKLY"
        elif re.search(r"\bevery\s+month\b|\bmonthly\b", lowered):
            recurrence_type = "MONTHLY"
            recurrence_rule = "FREQ=MONTHLY"

    # ── 3. Relative time offset (e.g., "in 30 minutes", "in 2 hours") ─────────
    relative_match = re.search(
        r"\bin\s+(\d+)\s*(mins?|minutes?|hours?|hrs?|seconds?|secs?)\b",
        lowered,
    )
    if not relative_match:
        # Check "half an hour"
        if "in half an hour" in lowered or "half hour" in lowered:
            scheduled_at = now + timedelta(minutes=30)
            title = _clean_task_title(raw_query)
            return ParsedSchedule(
                title=title or "Reminder",
                scheduled_at=scheduled_at,
                timezone_name=active_tz.key,
                recurrence_type=recurrence_type,
                recurrence_rule=recurrence_rule,
                reminder_type=reminder_type,
                duration_minutes=15,
                is_ambiguous=False,
                clarification_question=None,
                raw_query=raw_query,
            )

    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        if "sec" in unit:
            delta = timedelta(seconds=amount)
        elif "min" in unit:
            delta = timedelta(minutes=amount)
        elif "hour" in unit or "hr" in unit:
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(minutes=amount)

        scheduled_at = now + delta
        title = _clean_task_title(raw_query)
        return ParsedSchedule(
            title=title or "Reminder",
            scheduled_at=scheduled_at,
            timezone_name=active_tz.key,
            recurrence_type=recurrence_type,
            recurrence_rule=recurrence_rule,
            reminder_type=reminder_type,
            duration_minutes=15,
            is_ambiguous=False,
            clarification_question=None,
            raw_query=raw_query,
        )

    # ── 4. Target Day Resolution ─────────────────────────────────────────────
    target_date = now.date()
    has_date_spec = False

    if "tomorrow" in lowered:
        target_date = now.date() + timedelta(days=1)
        has_date_spec = True
    elif "day after tomorrow" in lowered:
        target_date = now.date() + timedelta(days=2)
        has_date_spec = True
    elif "today" in lowered or "tonight" in lowered or "this evening" in lowered:
        target_date = now.date()
        has_date_spec = True
    else:
        # Next [day of week] e.g. "next monday", "on tuesday", "this friday"
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for idx, d_name in enumerate(day_names):
            if re.search(rf"\b(?:next|this|on)?\s*{d_name}\b", lowered):
                current_weekday = now.weekday()  # 0 is Monday
                days_ahead = idx - current_weekday
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = now.date() + timedelta(days=days_ahead)
                has_date_spec = True
                break

    # ── 5. Target Time Resolution ────────────────────────────────────────────
    target_time: Optional[time] = None
    has_explicit_time = False

    # Check for "at HH:MM [AM|PM]" or "at HH [AM|PM]" or "HH PM"
    time_match = re.search(
        r"(?:at\s+)?(\b\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        lowered,
    )
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        ampm = time_match.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        target_time = time(hour=hour, minute=minute)
        has_explicit_time = True
    else:
        # Check 24-hour time e.g. "at 14:00", "at 17:30"
        time_24_match = re.search(r"\bat\s+(\d{1,2}):(\d{2})\b", lowered)
        if time_24_match:
            hour = int(time_24_match.group(1))
            minute = int(time_24_match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                target_time = time(hour=hour, minute=minute)
                has_explicit_time = True

    # Check semantic times of day if no explicit time given
    if not target_time:
        if "morning" in lowered:
            target_time = time(hour=9, minute=0)
            has_explicit_time = True
        elif "afternoon" in lowered:
            target_time = time(hour=14, minute=0)
            has_explicit_time = True
        elif "evening" in lowered:
            target_time = time(hour=18, minute=0)
            has_explicit_time = True
        elif "tonight" in lowered:
            target_time = time(hour=20, minute=0)
            has_explicit_time = True
        elif "noon" in lowered or "midday" in lowered:
            target_time = time(hour=12, minute=0)
            has_explicit_time = True

    # ── 6. Ambiguity Checking ────────────────────────────────────────────────
    # If date is specified (e.g., "Remind me tomorrow") but NO time is provided:
    if has_date_spec and not has_explicit_time:
        day_label = "tomorrow" if "tomorrow" in lowered else "that day"
        return ParsedSchedule(
            title=_clean_task_title(raw_query) or "Scheduled Reminder",
            scheduled_at=None,
            timezone_name=active_tz.key,
            recurrence_type=recurrence_type,
            recurrence_rule=recurrence_rule,
            reminder_type=reminder_type,
            duration_minutes=15,
            is_ambiguous=True,
            clarification_question=f"What time should I remind you {day_label}?",
            raw_query=raw_query,
        )

    # If neither date nor time is provided (e.g. "Remind me to submit timesheet")
    if not has_date_spec and not has_explicit_time and recurrence_type == "NONE":
        task_name = _clean_task_title(raw_query) or "this"
        return ParsedSchedule(
            title=task_name,
            scheduled_at=None,
            timezone_name=active_tz.key,
            recurrence_type=recurrence_type,
            recurrence_rule=recurrence_rule,
            reminder_type=reminder_type,
            duration_minutes=15,
            is_ambiguous=True,
            clarification_question=f"When would you like to be reminded to {task_name}?",
            raw_query=raw_query,
        )

    # ── 7. Construct final datetime ──────────────────────────────────────────
    if target_time is None:
        target_time = time(hour=9, minute=0)  # default morning

    scheduled_dt = datetime.combine(target_date, target_time, tzinfo=active_tz)

    # If the time is for today but already in the past, adjust
    if not has_date_spec and scheduled_dt <= now:
        # Assume next occurrence (tomorrow or next week)
        if recurrence_type == "NONE":
            scheduled_dt = scheduled_dt + timedelta(days=1)

    title = _clean_task_title(raw_query)
    if not title:
        title = "Scheduled Reminder"

    return ParsedSchedule(
        title=title,
        scheduled_at=scheduled_dt,
        timezone_name=active_tz.key,
        recurrence_type=recurrence_type,
        recurrence_rule=recurrence_rule,
        reminder_type=reminder_type,
        duration_minutes=15,
        is_ambiguous=False,
        clarification_question=None,
        raw_query=raw_query,
    )


def _clean_task_title(text: str) -> str:
    """Strips scheduling command prefixes and timing phrases to extract task title."""
    s = text

    # Remove leading command phrases
    lead_patterns = [
        r"^remind\s+(?:me\s+)?(?:to\s+)?",
        r"^schedule\s+(?:a\s+)?(?:reminder\s+)?(?:to\s+)?(?:for\s+)?",
        r"^set\s+(?:a\s+)?reminder\s+(?:to\s+)?(?:for\s+)?",
        r"^create\s+(?:a\s+)?(?:reminder|task)\s+(?:to\s+)?(?:for\s+)?",
        r"^please\s+",
        r"^can\s+you\s+",
    ]
    for lp in lead_patterns:
        s = re.sub(lp, "", s, flags=re.IGNORECASE).strip()

    # Split on " to " e.g. "Remind me tomorrow at 10 AM to call HR" -> "call HR"
    if re.search(r"\bto\s+", s, flags=re.IGNORECASE):
        parts = re.split(r"\bto\s+", s, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1 and len(parts[1].strip()) > 3:
            s = parts[1].strip()

    # Strip remaining trailing timing phrases
    timing_phrases = [
        r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        r"\btomorrow\s*(?:morning|afternoon|evening|night)?\b",
        r"\btoday\b",
        r"\btonight\b",
        r"\bthis\s+evening\b",
        r"\bevery\s+(?:day|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\bdaily\b",
        r"\bweekly\b",
        r"\bin\s+\d+\s*(?:mins?|minutes?|hours?|hrs?)\b",
        r"\bfrom\s+now\b",
    ]
    for tp in timing_phrases:
        s = re.sub(tp, "", s, flags=re.IGNORECASE).strip()

    # Clean up punctuation and whitespace
    s = re.sub(r"^to\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"[.!?]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()

    # Capitalize first letter
    if s:
        s = s[0].upper() + s[1:]
    return s


# Convenient alias for natural language scheduling parsing
parse_natural_language_time = parse_schedule_request

