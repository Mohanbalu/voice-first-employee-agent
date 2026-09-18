"""Campus Navigation & Turn-by-Turn Directions Engine.

Provides authoritative HCL Campus layout, building coordinates, facilities directory,
and step-by-step pedestrian walking directions with estimated walking times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Authoritative HCL Campus Building Coordinates (Lat, Lon)
CAMPUS_LOCATIONS: Dict[str, Dict[str, Any]] = {
    "ENTRANCE_GATE": {
        "id": "entrance_gate",
        "name": "Entrance Gate",
        "aliases": ["main gate", "gate", "campus entrance", "security gate", "entrance"],
        "location_type": "CAMPUS_ENTRANCE",
        "latitude": 16.533000,
        "longitude": 80.791000,
        "description": "Main Campus Security, Visitor Registration, Shuttle Bus Drop-off",
        "facilities": [
            "Visitor Management Desk",
            "Security Checkpoint",
            "Shuttle & Cab Drop-off Bay",
            "Bicycle Parking",
        ],
    },
    "TOWER_1": {
        "id": "tower_1",
        "name": "Tower 1",
        "aliases": ["tower 1", "tower one", "t1", "tower1"],
        "location_type": "BUILDING",
        "latitude": 16.533700,
        "longitude": 80.791341,
        "description": "Engineering Labs, Cafeteria, HR Helpdesk, Main Reception",
        "floors": {
            "Ground Floor": "Central Reception, Main Food Court / Cafeteria, HR Helpdesk, Banking ATM",
            "Floor 1": "Client Collaboration Center, Training Rooms 101-105",
            "Floor 2": "IT Asset Center, Hardware Support Desk, Network NOC",
            "Floor 3": "SDC Engineering Labs, ODC-3, Breakout Zone",
            "Floor 4": "Executive Management Suites, Boardrooms A & B",
        },
        "facilities": [
            "Main Cafeteria / Food Court (Ground Floor)",
            "HR Helpdesk (Ground Floor)",
            "Central Reception (Ground Floor)",
            "IT Support & Laptop Repair (2nd Floor)",
            "Engineering ODC-3 (3rd Floor)",
        ],
    },
    "TOWER_2": {
        "id": "tower_2",
        "name": "Tower 2",
        "aliases": ["tower 2", "tower two", "t2", "tower2"],
        "location_type": "BUILDING",
        "latitude": 16.534200,
        "longitude": 80.792100,
        "description": "Techbees Academy, Conference Hall B, Indoor Recreation",
        "floors": {
            "Ground Floor": "South Atrium, Coffee Bar, First Aid & Medical Room",
            "Floor 1": "Conference Hall B, Recreation Zone (Table Tennis, Chess, Carrom)",
            "Floor 2": "Techbees Classrooms & Learning Labs 201-210",
            "Floor 3": "Global Delivery Center, Operations ODC",
            "Floor 4": "Finance & Legal Department, Auditoriums",
        },
        "facilities": [
            "Conference Hall B (1st Floor)",
            "Indoor Games & Play Area (1st Floor)",
            "Techbees Classrooms (2nd Floor)",
            "First Aid / Medical Room (Ground Floor)",
        ],
    },
    "SDC": {
        "id": "sdc",
        "name": "SDC",
        "aliases": ["sdc", "sdc building", "sdc block", "software development center"],
        "location_type": "BUILDING",
        "latitude": 16.534800,
        "longitude": 80.791600,
        "description": "Software Development Center, Dedicated High-Security ODCs",
        "floors": {
            "Ground Floor": "SDC Security Control, Badge Access Check",
            "Floor 1": "Shared ODC-1, Cloud Platform Engineering",
            "Floor 2": "Dedicated Project ODC-2, AI Research Center",
            "Floor 3": "Server Infrastructure & Telecom Hub",
        },
        "facilities": [
            "Software Development Centers (ODC 1 & 2)",
            "AI Research Center (2nd Floor)",
            "Badge Access & Security Control",
        ],
    },
}

# Walking routes between main nodes (Distance in meters, Walking time in minutes, Step-by-step turn directions)
WALKING_ROUTES: Dict[tuple[str, str], Dict[str, Any]] = {
    ("ENTRANCE_GATE", "TOWER_1"): {
        "distance_meters": 110,
        "walking_minutes": 2,
        "steps": [
            "Pass through the Main Entrance Security Gate turnstiles.",
            "Follow the central palm-lined pedestrian walkway straight ahead for approximately 80 meters.",
            "Tower 1 main entrance with glass revolving doors will be directly in front of you.",
            "Enter the lobby — the Central Reception and Cafeteria are immediately accessible.",
        ],
    },
    ("ENTRANCE_GATE", "TOWER_2"): {
        "distance_meters": 180,
        "walking_minutes": 3,
        "steps": [
            "Pass through the Main Entrance Security Gate turnstiles.",
            "Take the covered walkway veering right towards the East Campus Plaza.",
            "Walk past the central water fountain for about 120 meters.",
            "Tower 2 South Entrance is located directly across the East Courtyard.",
        ],
    },
    ("ENTRANCE_GATE", "SDC"): {
        "distance_meters": 220,
        "walking_minutes": 3,
        "steps": [
            "Pass through the Main Entrance Security Gate turnstiles.",
            "Walk straight along the central avenue past Tower 1 on your right.",
            "Continue through the covered North Link walkway for another 90 meters.",
            "The SDC building entrance with biometric security will be on your left.",
        ],
    },
    ("TOWER_1", "TOWER_2"): {
        "distance_meters": 120,
        "walking_minutes": 2,
        "steps": [
            "Exit Tower 1 through the East Lobby doors (near the Cafeteria).",
            "Take the shaded Skywalk / Central Plaza walkway directly connecting Tower 1 and Tower 2.",
            "Cross the 80-meter landscaped courtyard.",
            "Enter Tower 2 through the West Concourse doors.",
        ],
    },
    ("TOWER_1", "SDC"): {
        "distance_meters": 130,
        "walking_minutes": 2,
        "steps": [
            "Exit Tower 1 through the North Concourse.",
            "Follow the covered North corridor towards SDC.",
            "Cross the inner service lane using the marked pedestrian crosswalk.",
            "Enter SDC through the main security turnstiles.",
        ],
    },
    ("TOWER_2", "SDC"): {
        "distance_meters": 140,
        "walking_minutes": 2,
        "steps": [
            "Exit Tower 2 through the North Exit.",
            "Head north-west along the garden promenade.",
            "Walk past the Cafeteria outdoor seating area for about 100 meters.",
            "Enter SDC through the East Glass Entrance.",
        ],
    },
}


def normalize_campus_name(query: str) -> Optional[str]:
    """Finds canonical campus key for a query string."""
    q = query.lower().strip()
    for key, data in CAMPUS_LOCATIONS.items():
        if key.lower() == q or data["name"].lower() == q:
            return key
        for alias in data["aliases"]:
            if alias in q:
                return key
    return None


def calculate_gps_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula for straight-line meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return R * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def find_nearest_campus_location(latitude: float, longitude: float) -> Dict[str, Any]:
    """Finds closest campus building to live GPS coordinates."""
    nearest_key = None
    min_dist = float("inf")

    all_distances = []
    for key, loc in CAMPUS_LOCATIONS.items():
        dist_m = calculate_gps_distance_meters(latitude, longitude, loc["latitude"], loc["longitude"])
        all_distances.append({
            "key": key,
            "name": loc["name"],
            "distance_meters": round(dist_m, 1),
            "description": loc["description"],
        })
        if dist_m < min_dist:
            min_dist = dist_m
            nearest_key = key

    all_distances.sort(key=lambda x: x["distance_meters"])
    nearest_loc = CAMPUS_LOCATIONS[nearest_key]

    is_outside = min_dist > 500.0
    proximity = "OUTSIDE_CAMPUS" if is_outside else ("VERY_CLOSE" if min_dist <= 35.0 else "NEAR")

    return {
        "nearest_key": nearest_key,
        "nearest_name": nearest_loc["name"],
        "nearest_description": nearest_loc["description"],
        "distance_meters": round(min_dist, 1),
        "proximity_status": proximity,
        "is_outside": is_outside,
        "all_distances": all_distances,
        "facilities": nearest_loc.get("facilities", []),
    }


def get_navigation_route(origin_key: str, destination_key: str) -> Optional[Dict[str, Any]]:
    """Calculates step-by-step route between two campus locations."""
    origin = origin_key.upper()
    dest = destination_key.upper()

    if origin == dest:
        loc = CAMPUS_LOCATIONS.get(origin, {})
        return {
            "origin": loc.get("name", origin),
            "destination": loc.get("name", dest),
            "distance_meters": 0,
            "walking_minutes": 0,
            "steps": ["You are already at your destination!"],
            "description": loc.get("description", ""),
        }

    # Check forward or reverse route
    if (origin, dest) in WALKING_ROUTES:
        route = WALKING_ROUTES[(origin, dest)]
        return {
            "origin": CAMPUS_LOCATIONS[origin]["name"],
            "destination": CAMPUS_LOCATIONS[dest]["name"],
            "distance_meters": route["distance_meters"],
            "walking_minutes": route["walking_minutes"],
            "steps": route["steps"],
        }

    if (dest, origin) in WALKING_ROUTES:
        route = WALKING_ROUTES[(dest, origin)]
        # Reverse steps
        rev_steps = [s.replace("Exit", "Enter").replace("Enter", "Exit") for s in reversed(route["steps"])]
        return {
            "origin": CAMPUS_LOCATIONS[origin]["name"],
            "destination": CAMPUS_LOCATIONS[dest]["name"],
            "distance_meters": route["distance_meters"],
            "walking_minutes": route["walking_minutes"],
            "steps": route["steps"],  # Or natural walking steps
        }

    # Default fallback route through central campus walkway
    loc_dest = CAMPUS_LOCATIONS.get(dest, {})
    return {
        "origin": CAMPUS_LOCATIONS.get(origin, {}).get("name", origin),
        "destination": loc_dest.get("name", dest),
        "distance_meters": 150,
        "walking_minutes": 2,
        "steps": [
            f"Exit your current building towards the central campus courtyard.",
            f"Follow the designated covered walkway towards {loc_dest.get('name', dest)}.",
            f"Enter {loc_dest.get('name', dest)} through the main lobby.",
        ],
    }


def resolve_facility_location(facility_query: str) -> Optional[Dict[str, Any]]:
    """Maps a facility keyword (e.g. 'cafeteria', 'techbees', 'it team') to its building & floor."""
    fq = facility_query.lower()

    if any(w in fq for w in ("cafeteria", "canteen", "food", "lunch", "eat", "coffee", "snack")):
        return {
            "facility": "Cafeteria / Food Court",
            "building_key": "TOWER_1",
            "building_name": "Tower 1",
            "floor": "Ground Floor",
            "details": "Central Food Court with multi-cuisine counters, coffee stalls, and outdoor patio.",
        }
    if any(w in fq for w in ("techbees", "techbee", "classroom", "classrooms", "trainee", "training lab")):
        return {
            "facility": "Techbees Classrooms & Labs",
            "building_key": "TOWER_2",
            "building_name": "Tower 2",
            "floor": "Floor 2",
            "details": "Techbees Academy Learning Labs 201-210 with dedicated training terminals.",
        }
    if any(w in fq for w in ("recreation", "play area", "indoor games", "table tennis", "chess", "carrom", "tt")):
        return {
            "facility": "Recreation & Indoor Games",
            "building_key": "TOWER_2",
            "building_name": "Tower 2",
            "floor": "Floor 1",
            "details": "Table tennis tables, chess boards, carrom boards, and relaxation lounge.",
        }
    if any(w in fq for w in ("it team", "it support", "laptop support", "hardware", "broken laptop", "it helpdesk")):
        return {
            "facility": "IT Support & Asset Desk",
            "building_key": "TOWER_1",
            "building_name": "Tower 1",
            "floor": "Floor 2",
            "details": "Walk-in IT hardware repair, laptop exchange, and peripheral provisioning.",
        }
    if any(w in fq for w in ("hr helpdesk", "hr", "human resources", "hr desk")):
        return {
            "facility": "HR Helpdesk",
            "building_key": "TOWER_1",
            "building_name": "Tower 1",
            "floor": "Ground Floor",
            "details": "Employee relations, identity badges, letter requests, and onboarding support.",
        }
    if any(w in fq for w in ("conference room b", "conference hall b", "seminar")):
        return {
            "facility": "Conference Hall B",
            "building_key": "TOWER_2",
            "building_name": "Tower 2",
            "floor": "Floor 1",
            "details": "120-seat executive conference room with video conferencing capabilities.",
        }
    if any(w in fq for w in ("medical", "doctor", "first aid", "clinic", "nurse", "health")):
        return {
            "facility": "First Aid & Medical Room",
            "building_key": "TOWER_2",
            "building_name": "Tower 2",
            "floor": "Ground Floor",
            "details": "Full-time campus nurse, emergency medical kit, and rest beds.",
        }
    if any(w in fq for w in ("sdc", "software development center", "odc")):
        return {
            "facility": "Software Development Center (SDC)",
            "building_key": "SDC",
            "building_name": "SDC",
            "floor": "Floors 1 & 2",
            "details": "High-security project ODCs and software engineering workspace.",
        }

    return None
