"""Navigation Agent Node — Module 5 & 6.

Processes employee navigation queries, detects live GPS coordinates,
determines campus proximity, and generates step-by-step pedestrian walking directions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.app.agents.agent_state import AgentState
from backend.app.navigation.directions import (
    CAMPUS_LOCATIONS,
    find_nearest_campus_location,
    get_navigation_route,
    normalize_campus_name,
    resolve_facility_location,
)

logger = logging.getLogger("agents.navigation_agent")


class NavigationAgentNode:
    """LangGraph node: handles NAVIGATION intents with live GPS and campus routing."""

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        request = state.get("request", "")
        lat = state.get("latitude")
        lon = state.get("longitude")
        accuracy = state.get("accuracy")

        logger.info(
            "Navigation request received: '%s' (lat=%s, lon=%s, acc=%s)",
            request, lat, lon, accuracy,
        )

        has_coords = lat is not None and lon is not None

        # 1. Determine origin
        origin_info: Optional[Dict[str, Any]] = None
        origin_key = "ENTRANCE_GATE"  # default reference if no GPS
        is_outside = False

        if has_coords:
            origin_info = find_nearest_campus_location(float(lat), float(lon))
            origin_key = origin_info["nearest_key"]
            is_outside = origin_info["is_outside"]

        # 2. Check if a specific destination or facility was requested
        facility_info = resolve_facility_location(request)
        dest_building_key: Optional[str] = None
        destination_name: Optional[str] = None
        destination_floor: Optional[str] = None
        destination_details: Optional[str] = None

        if facility_info:
            dest_building_key = facility_info["building_key"]
            destination_name = facility_info["facility"]
            destination_floor = facility_info["floor"]
            destination_details = facility_info["details"]
        else:
            # Check if building name was queried directly (e.g. "where is Tower 2?")
            matched_key = normalize_campus_name(request)
            if matched_key:
                dest_building_key = matched_key
                bldg = CAMPUS_LOCATIONS[matched_key]
                destination_name = bldg["name"]
                destination_details = bldg["description"]

        # 3. Handle query scenarios:
        # Scenario A: User asking about current location ("Where am I located?", "What is my location?")
        is_location_check = any(phrase in request.lower() for phrase in (
            "where am i", "my location", "current location", "where is my desk",
            "which building am i in", "am i near", "campus location"
        )) and not dest_building_key

        if is_location_check:
            if has_coords and origin_info:
                if is_outside:
                    resp_text = (
                        f"You appear to be outside the HCL campus perimeter. "
                        f"The nearest campus point is {origin_info['nearest_name']}, "
                        f"approximately {int(origin_info['distance_meters'])} meters away in straight-line distance.\n\n"
                        f"Once you arrive at the campus, I can guide you to your desk, meeting rooms, or facilities."
                    )
                elif origin_info["proximity_status"] == "VERY_CLOSE":
                    fac_str = "\n".join(f"  • {f}" for f in origin_info.get("facilities", [])[:4])
                    resp_text = (
                        f"You are currently located at {origin_info['nearest_name']} "
                        f"(approximately {int(origin_info['distance_meters'])} meters from the reference point).\n\n"
                        f"Facilities at this location:\n{fac_str}\n\n"
                        f"Where would you like to navigate to?"
                    )
                else:
                    fac_str = "\n".join(f"  • {f}" for f in origin_info.get("facilities", [])[:4])
                    resp_text = (
                        f"You appear to be near {origin_info['nearest_name']} "
                        f"(approximately {int(origin_info['distance_meters'])} meters away).\n\n"
                        f"Facilities at this location:\n{fac_str}\n\n"
                        f"Tell me your destination or choose a quick location below to get walking directions."
                    )
            else:
                resp_text = (
                    "To determine your exact location on campus, please allow location access on your device.\n\n"
                    "HCL Campus Directory:\n"
                    "  • Tower 1: Main Cafeteria (Ground Floor), HR Helpdesk, IT Support (Floor 2), SDC Engineering Labs (Floor 3)\n"
                    "  • Tower 2: Conference Hall B (Floor 1), Indoor Games & Recreation (Floor 1), Techbees Classrooms (Floor 2)\n"
                    "  • SDC Building: Software Development Center ODCs & Cloud Engineering\n"
                    "  • Entrance Gate: Main Security, Visitor Registration, Cab/Shuttle Bay\n\n"
                    "Which building or facility would you like directions to?"
                )

            location_payload = {
                "has_coordinates": has_coords,
                "latitude": lat,
                "longitude": lon,
                "current_location": origin_info["nearest_name"] if origin_info else None,
                "proximity_status": origin_info["proximity_status"] if origin_info else "NO_COORDINATES",
                "distance_to_nearest_meters": origin_info["distance_meters"] if origin_info else None,
                "destination": None,
                "steps": [],
                "quick_destinations": [
                    {"label": "Cafeteria (Tower 1)", "query": "Navigate to Cafeteria"},
                    {"label": "IT Support (Tower 1)", "query": "Where is IT Support?"},
                    {"label": "Techbees (Tower 2)", "query": "Navigate to Techbees"},
                    {"label": "Recreation Area (Tower 2)", "query": "Where is the Recreation Area?"},
                    {"label": "SDC Building", "query": "Navigate to SDC Building"},
                ],
            }

            return {
                "agent_mode": "tool_intent",
                "tool_intents": [{
                    "tool": "indoor_navigation_system",
                    "intent": "navigation",
                    "status": "success",
                    "message": resp_text,
                    "requires_action": False,
                    "action_description": "Campus location identification",
                }],
                "location_data": location_payload,
                "final_response": resp_text,
            }

        # Scenario B: Route to specific facility or building requested
        target_key = dest_building_key or "TOWER_1"
        route = get_navigation_route(origin_key, target_key)

        steps_formatted = "\n".join(f"  {idx + 1}. {step}" for idx, step in enumerate(route["steps"]))
        from_label = origin_info["nearest_name"] if origin_info else "Entrance Gate"
        to_label = destination_name or CAMPUS_LOCATIONS[target_key]["name"]

        floor_note = f" on {destination_floor}" if destination_floor else ""
        resp_text = (
            f"Here are the walking directions to {to_label}{floor_note} from {from_label} "
            f"(approx. {route['distance_meters']} meters, {route['walking_minutes']} min walk):\n\n"
            f"{steps_formatted}\n\n"
            f"Note: Follow designated pedestrian walkways. Floor directories are available at each building lobby."
        )

        location_payload = {
            "has_coordinates": has_coords,
            "latitude": lat,
            "longitude": lon,
            "origin": from_label,
            "destination": to_label,
            "destination_floor": destination_floor,
            "distance_meters": route["distance_meters"],
            "walking_minutes": route["walking_minutes"],
            "steps": route["steps"],
            "quick_destinations": [
                {"label": "Cafeteria (Tower 1)", "query": "Navigate to Cafeteria"},
                {"label": "IT Support (Tower 1)", "query": "Where is IT Support?"},
                {"label": "Techbees (Tower 2)", "query": "Navigate to Techbees"},
                {"label": "Recreation Area (Tower 2)", "query": "Where is the Recreation Area?"},
            ],
        }

        return {
            "agent_mode": "tool_intent",
            "tool_intents": [{
                "tool": "indoor_navigation_system",
                "intent": "navigation",
                "status": "success",
                "message": resp_text,
                "requires_action": False,
                "action_description": f"Directions to {to_label}",
            }],
            "location_data": location_payload,
            "final_response": resp_text,
        }
