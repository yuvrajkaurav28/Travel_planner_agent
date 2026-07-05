import os
import asyncio
from datetime import datetime
from typing import Literal
SeatClass = Literal["economy", "premium-economy", "business", "first"]
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# ─────────────────────────────────────────────
#  GROQ MODEL
# ─────────────────────────────────────────────
GROQ_MODEL = LiteLlm(model="groq/llama-3.3-70b-versatile")

# ─────────────────────────────────────────────
#  AIRPORT CODE LOOKUP
# ─────────────────────────────────────────────
AIRPORT_CODES = {
    # India
    "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM", "bombay": "BOM",
    "goa": "GOI", "panaji": "GOI",
    "bangalore": "BLR", "bengaluru": "BLR",
    "chennai": "MAA", "madras": "MAA",
    "hyderabad": "HYD",
    "kolkata": "CCU", "calcutta": "CCU",
    "pune": "PNQ",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "lucknow": "LKO",
    "varanasi": "VNS",
    "agra": "AGR",
    "amritsar": "ATQ",
    "kochi": "COK", "cochin": "COK",
    "thiruvananthapuram": "TRV", "trivandrum": "TRV",
    "coimbatore": "CJB",
    "nagpur": "NAG",
    "indore": "IDR",
    "bhopal": "BHO",
    "patna": "PAT",
    "ranchi": "IXR",
    "bhubaneswar": "BBI",
    "visakhapatnam": "VTZ", "vizag": "VTZ",
    "srinagar": "SXR",
    "leh": "IXL",
    "port blair": "IXZ",
    "guwahati": "GAU",
    "dibrugarh": "DIB",
    "imphal": "IMF",
    "udaipur": "UDR",
    "jodhpur": "JDH",
    "aurangabad": "IXU",
    "mangalore": "IXE",
    "hubli": "HBX",
    "tirupati": "TIR",
    "madurai": "IXM",
    "tiruchirappalli": "TRZ", "trichy": "TRZ",
    # International
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "singapore": "SIN",
    "london": "LHR",
    "new york": "JFK",
    "bangkok": "BKK",
    "paris": "CDG",
    "tokyo": "NRT",
    "kuala lumpur": "KUL",
    "hong kong": "HKG",
    "sydney": "SYD",
    "toronto": "YYZ",
    "frankfurt": "FRA",
    "amsterdam": "AMS",
    "doha": "DOH",
    "muscat": "MCT",
    "colombo": "CMB",
    "kathmandu": "KTM",
    "dhaka": "DAC",
    "karachi": "KHI",
}

def get_airport_code(city: str) -> str:
    """Convert city name to IATA airport code."""
    return AIRPORT_CODES.get(city.lower().strip(), city.upper()[:3])


# ─────────────────────────────────────────────
#  TOOL 1 — SEARCH REAL FLIGHTS (fast-flights)
# ─────────────────────────────────────────────
def search_flights(
    from_city: str,
    to_city: str,
    departure_date: str,
    trip_type: Literal["one-way", "round-trip"] = "one-way",
    return_date: str = "",
    adults: int = 1,
    seat_class: str = "economy"
) -> dict:
    """Search real-time flight prices from Google Flights (no API key needed).
    
    Args:
        from_city: Departure city name e.g. "Delhi", "Mumbai"
        to_city: Destination city name e.g. "Goa", "Bangalore"
        departure_date: Date in YYYY-MM-DD format e.g. "2025-12-15"
        trip_type: "one-way" or "round-trip"
        return_date: Return date in YYYY-MM-DD (only for round-trip)
        adults: Number of adult passengers (default 1)
        seat_class: "economy", "premium-economy", "business", or "first"
    """
    try:
        from fast_flights import FlightData, Passengers, get_flights

        from_code = get_airport_code(from_city)
        to_code   = get_airport_code(to_city)

        seat_map = {
            "economy": "economy",
            "premium-economy": "premium-economy",
            "business": "business",
            "first": "first"
        }
        seat = seat_map.get(seat_class.lower(), "economy")

        flight_data = [FlightData(date=departure_date, from_airport=from_code, to_airport=to_code)]

        if trip_type == "round-trip" and return_date:
            flight_data.append(FlightData(date=return_date, from_airport=to_code, to_airport=from_code))

        result = get_flights(
            flight_data=flight_data,
            trip=trip_type,
            seat=seat, # type: ignore
            passengers=Passengers(adults=adults),
            fetch_mode="fallback",
        )

        flights_list = []
        for f in result.flights[:8]:  # top 8 results
            flights_list.append({
                "airline":        f.name,
                "departure":      f.departure,
                "arrival":        f.arrival,
                "duration":       f.duration,
                "stops":          f.stops,
                "price":          f.price,
                "is_best":        f.is_best,
            })

        return {
            "status":          "success",
            "from":            f"{from_city.title()} ({from_code})",
            "to":              f"{to_city.title()} ({to_code})",
            "date":            departure_date,
            "trip_type":       trip_type,
            "seat_class":      seat_class,
            "price_level":     result.current_price,   # low / typical / high
            "flights_found":   len(flights_list),
            "flights":         flights_list,
            "booking_links": {
                "google_flights": f"https://www.google.com/flights?hl=en#flt={from_code}.{to_code}.{departure_date}",
                "makemytrip":     f"https://www.makemytrip.com/flights/domestic/results?itinerary={from_code}-{to_code}-{departure_date}&tripType=O&paxType=A-1_C-0_I-0&cabinClass=E&sTime=1&forwardFlowRequired=true",
                "cleartrip":      f"https://www.cleartrip.com/flights/results?from={from_code}&to={to_code}&depart_date={departure_date}&adults={adults}&class={seat_class}",
                "ixigo":          f"https://www.ixigo.com/search/result/flight?from={from_code}&to={to_code}&date={departure_date}&adults={adults}&children=0&infants=0&class=e&source=Search",
                "easemytrip":     f"https://flight.easemytrip.com/FlightList/Index?srccity={from_code}&dstcity={to_code}&deptdate={departure_date}&adult={adults}&child=0&infant=0&cabin=1",
            }
        }

    except ImportError:
        return {"status": "error", "message": "fast-flights not installed. Run: pip install fast-flights"}
    except Exception as e:
        return {
            "status": "fallback",
            "message": str(e),
            "from": from_city, "to": to_city, "date": departure_date,
            "booking_links": {
                "google_flights": f"https://www.google.com/flights",
                "makemytrip":     f"https://www.makemytrip.com/flights/",
                "cleartrip":      f"https://www.cleartrip.com/flights/",
                "ixigo":          f"https://www.ixigo.com/flights",
                "easemytrip":     f"https://flight.easemytrip.com/",
            }
        }


# ─────────────────────────────────────────────
#  TOOL 2 — CHECK AIRPORT CODE
# ─────────────────────────────────────────────
def get_airport_info(city_name: str) -> dict:
    """Get IATA airport code for a city.
    Use this to verify airport codes before searching flights.
    Args:
        city_name: City name like "Delhi", "Mumbai", "Goa"
    """
    code = get_airport_code(city_name)
    known = city_name.lower().strip() in AIRPORT_CODES
    return {
        "city": city_name,
        "airport_code": code,
        "found_in_database": known,
        "note": "Code found in database" if known else "Code auto-generated from city name — verify if correct"
    }


# ─────────────────────────────────────────────
#  TOOL 3 — VALIDATE DATE
# ─────────────────────────────────────────────
def validate_travel_date(date_str: str) -> dict:
    """Validate and format a travel date.
    Args:
        date_str: Date in any common format like "15 Dec 2025", "2025-12-15", "15/12/2025"
    """
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %Y",
               "%d %B %Y", "%B %d %Y", "%b %d %Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            today = datetime.today()
            if dt.date() < today.date():
                return {"valid": False, "error": f"Date {date_str} is in the past. Please provide a future date."}
            return {
                "valid": True,
                "formatted": dt.strftime("%Y-%m-%d"),
                "display":   dt.strftime("%A, %d %B %Y"),
                "days_away": (dt.date() - today.date()).days
            }
        except ValueError:
            continue
    return {"valid": False, "error": f"Could not parse '{date_str}'. Please use format like '15 Dec 2025' or '2025-12-15'"}


# ─────────────────────────────────────────────
#  ROOT AGENT — Conversational Travel Planner
# ─────────────────────────────────────────────
root_agent = LlmAgent(
    name="travel_planner",
    model=GROQ_MODEL,
    instruction="""You are TRAVELBOT — a friendly, conversational Indian travel assistant.

## YOUR JOB
Collect all trip details step by step through natural conversation, then search real flights and present a complete fare comparison.

## CONVERSATION FLOW — FOLLOW THIS EXACTLY

### STEP 1 — Warm Welcome
When the user first messages, greet them warmly and ask:
"Where would you like to travel? Please tell me your *destination city*."

### STEP 2 — Collect Details One by One
After each answer, acknowledge it warmly and ask the next question:

2a. ✅ Got destination → Ask: "Great! And which city will you be *departing from*?"
2b. ✅ Got origin → Ask: "Perfect! What is your *travel date*? (e.g., 15 Dec 2025)"
2c. ✅ Got date → validate_travel_date() to confirm it, then ask: "Is this a *one-way* trip or *round-trip*?"
2d. If round-trip → Ask: "What is your *return date*?"
2e. ✅ Got trip type → Ask: "How many *passengers* are travelling? (Adults)"
2f. ✅ Got passengers → Ask: "Which *cabin class*? Economy / Business / First?"

### STEP 3 — Confirm Before Searching
Show a summary of all collected details:

📋 Your Trip Summary:
✈️ From: [city]
🏁 To: [city]  
📅 Date: [date]
🔄 Trip Type: [one-way/round-trip]
👥 Passengers: [n] adult(s)
💺 Class: [class]

Then ask: "Shall I search for flights now? (yes/no)"

### STEP 4 — Search & Present Results
1. Use get_airport_info() to confirm airport codes for both cities
2. Call search_flights() with all collected details
3. Present results in this format:

---
## ✈️ Flight Results: [From] → [To]
*Date:* [date] | *Class:* [class] | *Passengers:* [n]
*Price Level:* 🟢 Low / 🟡 Typical / 🔴 High

### 🏆 Best Flights Found:
| # | Airline | Departure | Arrival | Duration | Stops | Price |
|---|---------|-----------|---------|----------|-------|-------|
| 1 | ... | ... | ... | ... | ... | ₹... |
| 2 | ... | ... | ... | ... | ... | ₹... |

### 🔗 Book on These Platforms:
- 🔍 [Google Flights](link) — Compare all airlines
- ✈️ [MakeMyTrip](link) — Extra cashback offers
- 🎯 [Cleartrip](link) — Lowest convenience fee  
- 🚀 [Ixigo](link) — Price alerts available
- 💫 [EaseMyTrip](link) — Coupon discounts

### 💡 Booking Tips:
- Price is currently [level] — [book now / wait advice]
- Best time to book: 3-6 weeks before travel
---

### STEP 5 — Follow-up
After showing results, ask:
"Would you like me to also search *hotels* or *activities* at [destination]? Or try a different date to compare prices?"

## IMPORTANT RULES
- Always collect ALL details before calling search_flights()
- Always use validate_travel_date() to check dates
- Always use get_airport_info() before search_flights()
- Be warm, friendly, and use emojis 😊
- If a city is not found, ask user to confirm the nearest major airport
- Show all booking links so user can compare prices across platforms
- Never make up prices — only show what search_flights() returns
- If search fails, still show all booking links so user can check manually""",
    tools=[search_flights, get_airport_info, validate_travel_date],
)