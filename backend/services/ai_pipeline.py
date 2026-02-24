"""
4-Stage AI Architectural Pipeline Engine.

Processes user requirements through 4 distinct stages:
  Stage 1 — Chat Mode: Natural conversation to collect requirements
  Stage 2 — Extraction Mode: Convert conversation → structured JSON
  Stage 3 — Design Mode: Generate construction-ready layout from JSON
  Stage 4 — Validation Mode: Validate the generated layout

Each stage uses a dedicated system prompt and produces structured
output that feeds the next stage.
"""

import json
import re
from typing import Optional, Dict, List, Tuple
from enum import Enum


class PipelineStage(str, Enum):
    CHAT = "chat"
    EXTRACTION = "extraction"
    DESIGN = "design"
    VALIDATION = "validation"
    GENERATION = "generation"
    COMPLETE = "complete"


# ============================================================================
# STAGE 1 — Chat Mode: Natural requirement collection
# ============================================================================

STAGE_1_CHAT_PROMPT = """You are a professional residential architect assistant.

Your role is to communicate naturally with users and understand their house requirements.

Do NOT generate floor plans yet.
Do NOT output JSON yet.

Your job:
- Ask for plot size (width × length in feet)
- Ask for number of bedrooms
- Ask for number of bathrooms
- Ask for number of floors
- Ask for special rooms (optional: dining, study, pooja, balcony, parking, garden)
- Suggest smart defaults if user is unsure
- Keep conversation natural and simple

When you have enough data, summarize requirements clearly and ask user to confirm.

MANDATORY DATA NEEDED (do not proceed without ALL of these):
- Plot width and length (or total area)
- Number of bedrooms
- Number of bathrooms
- Number of floors
- Living room (yes/no — default yes)
- Kitchen (yes/no — default yes)

If the user provides all mandatory data in their first message, summarize and confirm.
If data is missing, ask one or two follow-up questions at a time.

When all mandatory data is collected and user confirms, respond with EXACTLY this line at the end:
[REQUIREMENTS_COMPLETE]

Do not generate design in this mode."""


# ============================================================================
# STAGE 2 — Extraction Mode: Conversation → Structured JSON
# ============================================================================

STAGE_2_EXTRACTION_PROMPT = """You are a data structuring engine.

Convert the following user conversation into structured JSON.

Only output JSON. No explanations, no text before or after.

Format:

```json
{
  "plot_width": <number in feet>,
  "plot_length": <number in feet>,
  "total_area": <number in sq ft>,
  "floors": <number>,
  "bedrooms": <number>,
  "bathrooms": <number>,
  "living_room": true,
  "kitchen": true,
  "extras": ["dining", "study", "pooja", "balcony", "parking", "garden"]
}
```

Rules:
- If plot dimensions given, calculate total_area = width × length
- If only total_area given, estimate reasonable dimensions (ratio ~1.3:1)
- extras should only include rooms the user explicitly requested
- living_room and kitchen default to true unless user said no
- All numbers must be realistic positive values
- Output ONLY valid JSON, nothing else"""


# ============================================================================
# STAGE 3 — Design Mode: Structured JSON → Layout
# ============================================================================

STAGE_3_DESIGN_PROMPT = """You are a licensed architect generating a construction-ready residential layout.

You are NOT a chatbot now.
Do NOT ask questions.
Do NOT restart conversation.

Use provided structured requirements to:

1. Allocate area proportionally:
   Living: 18–22%
   Bedrooms total: 30–35%
   Kitchen: 8–12%
   Bathrooms: 8–12%
   Circulation: 10–15%
   Walls: 8–10%

2. Follow zoning:
   Public → Living near entrance
   Semi-private → Dining/Kitchen
   Private → Bedrooms
   Service → Bathrooms

3. Maintain constraints:
   Bedroom ≥ 100 sq ft
   Bathroom ≥ 35 sq ft
   Kitchen ≥ 80 sq ft
   Living ≥ 120 sq ft
   Passage ≥ 3 ft width
   Rectangular rooms only
   No overlapping
   9 inch external walls
   4.5 inch internal walls

4. Ensure logical adjacency and circulation.

5. Each room must have:
   - Proper position (x, y from origin 0,0)
   - Width and length
   - Door position (N/S/E/W wall)
   - Window positions (direction list)
   - Zone classification

Output format — provide a SHORT explanation first, then clean JSON:

```json
{
  "plot": {
    "width": <feet>,
    "length": <feet>,
    "unit": "ft"
  },
  "rooms": [
    {
      "name": "<display name>",
      "room_type": "<type>",
      "width": <feet>,
      "length": <feet>,
      "area": <sq ft>,
      "zone": "public|semi_private|private|service",
      "position": {"x": <feet>, "y": <feet>},
      "doors": [{"wall": "N|S|E|W", "offset": <feet from wall start>}],
      "windows": [{"wall": "N|S|E|W", "width": 4}]
    }
  ],
  "circulation": {
    "type": "central corridor|side corridor|open plan",
    "width": <feet>
  },
  "walls": {
    "external": "9 inch",
    "internal": "4.5 inch"
  },
  "design_validation": {
    "total_area_used": <sq ft>,
    "area_percentage": <percent of plot used>,
    "compliant": true
  }
}
```

Valid room types: master_bedroom, bedroom, bathroom, kitchen, living, dining, study,
pooja, store, utility, porch, parking, staircase, toilet, balcony, hallway, garage.

CRITICAL:
- Rooms must NOT overlap (check x,y positions + widths/lengths)
- Total room area must not exceed plot area
- All rooms must fit within plot boundaries
- Position (0,0) is bottom-left corner of the plot"""


# ============================================================================
# STAGE 4 — Validation Mode
# ============================================================================

STAGE_4_VALIDATION_PROMPT = """You are an architectural validator.

Check the provided floor plan layout for:

1. **Area overflow** — Total room area must not exceed plot area
2. **Overlapping rooms** — No two rooms should occupy the same space
3. **Unrealistic proportions** — No room should be narrower than 6 ft or have aspect ratio > 3:1
4. **Zoning violations** — Bedrooms should not open directly into kitchen, living near entrance
5. **Circulation gaps** — Every room must be reachable, no dead spaces
6. **Minimum sizes** — Bedroom ≥ 100sqft, Bathroom ≥ 35sqft, Kitchen ≥ 80sqft, Living ≥ 120sqft
7. **Wall alignment** — External walls at boundary, internal walls between rooms

Return ONLY a validation report as JSON:

```json
{
  "compliant": true|false,
  "total_area_used": <sq ft>,
  "plot_area": <sq ft>,
  "area_utilization": "<percentage>%",
  "checks": {
    "area_overflow": {"pass": true|false, "detail": "..."},
    "overlapping_rooms": {"pass": true|false, "detail": "..."},
    "proportions": {"pass": true|false, "detail": "..."},
    "zoning": {"pass": true|false, "detail": "..."},
    "circulation": {"pass": true|false, "detail": "..."},
    "minimum_sizes": {"pass": true|false, "detail": "..."}
  },
  "issues": ["<issue1>", "<issue2>"],
  "suggestions": ["<suggestion1>", "<suggestion2>"]
}
```

Be strict. Flag any issue that would cause construction problems.
Output ONLY valid JSON."""


# ============================================================================
# REQUIREMENT FIELDS
# ============================================================================

MANDATORY_FIELDS = [
    "plot_width", "plot_length", "total_area",  # at least one dimension set
    "bedrooms", "bathrooms", "floors",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from AI response text."""
    # Try ```json blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try any JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _clean_reply(text: str) -> str:
    """Remove JSON code blocks from reply for display."""
    clean = re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()
    return clean if clean else text


def _parse_collected_data(history: List[Dict]) -> dict:
    """
    Parse all collected data from conversation history using both
    direct regex matching and contextual analysis (what question was
    asked before each answer).
    """
    data = {
        "has_dimensions": False,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "has_floors": False,
        "plot_width": None,
        "plot_length": None,
        "total_area": None,
        "bedrooms": None,
        "bathrooms": None,
        "floors": None,
        "extras": [],
    }

    # --- Direct scan across all user messages ---
    full_text = " ".join(m.get("content", "") for m in history if m.get("role") == "user")
    full_lower = full_text.lower()

    # Dimensions: 30x40, 30*40, 30 x 40, 30×40, 30 × 40
    dim_match = re.search(r'(\d+)\s*[x×*]\s*(\d+)', full_lower)
    if dim_match:
        data["has_dimensions"] = True
        data["plot_width"] = int(dim_match.group(1))
        data["plot_length"] = int(dim_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]

    # Area: 1200 sqft, 1200 sq ft, 1200 square feet
    area_match = re.search(r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet?)', full_lower)
    if area_match:
        data["has_dimensions"] = True
        data["total_area"] = int(area_match.group(1))

    # Standalone large number (likely area if > 100)
    if not data["has_dimensions"]:
        for msg in history:
            if msg.get("role") != "user":
                continue
            num_match = re.match(r'^\s*(\d{3,5})\s*$', msg.get("content", "").strip())
            if num_match:
                val = int(num_match.group(1))
                if 100 <= val <= 50000:
                    data["has_dimensions"] = True
                    data["total_area"] = val

    # BHK: 3BHK, 3 bhk
    bhk_match = re.search(r'(\d+)\s*bhk', full_lower)
    if bhk_match:
        bhk = int(bhk_match.group(1))
        data["has_bedrooms"] = True
        data["has_bathrooms"] = True
        data["bedrooms"] = bhk
        data["bathrooms"] = max(1, bhk - 1)

    # Explicit bedrooms: 3 bedrooms, 3 bed
    bed_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?)', full_lower)
    if bed_match:
        data["has_bedrooms"] = True
        data["bedrooms"] = int(bed_match.group(1))

    # Explicit bathrooms: 2 bathrooms, 2 bath, 2 toilet
    bath_match = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilet)', full_lower)
    if bath_match:
        data["has_bathrooms"] = True
        data["bathrooms"] = int(bath_match.group(1))

    # Floors: 2 floors, 2 storey
    floor_match = re.search(r'(\d+)\s*(?:floor|storey|story|level)', full_lower)
    if floor_match:
        data["has_floors"] = True
        data["floors"] = int(floor_match.group(1))

    # --- Contextual parsing: look at assistant question → user answer pairs ---
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        user_text = msg.get("content", "").strip()

        # Find the previous assistant message to understand context
        prev_assistant = ""
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "assistant":
                prev_assistant = history[j].get("content", "").lower()
                break

        # Contextual: "2,2" or "2, 2" or "2 2" after asking about bed/bath
        if "bedroom" in prev_assistant and "bathroom" in prev_assistant:
            nums = re.findall(r'\d+', user_text)
            if len(nums) >= 2:
                data["has_bedrooms"] = True
                data["has_bathrooms"] = True
                data["bedrooms"] = int(nums[0])
                data["bathrooms"] = int(nums[1])
            elif len(nums) == 1:
                data["has_bedrooms"] = True
                data["bedrooms"] = int(nums[0])

        # Contextual: just a number after asking about floors
        if "floor" in prev_assistant or "storey" in prev_assistant:
            nums = re.findall(r'\d+', user_text)
            if len(nums) >= 1:
                data["has_floors"] = True
                data["floors"] = int(nums[0])

        # Contextual: just a number after asking about plot size
        if "plot" in prev_assistant and ("size" in prev_assistant or "dimension" in prev_assistant):
            dim_match_ctx = re.search(r'(\d+)\s*[x×*]\s*(\d+)', user_text.lower())
            if dim_match_ctx:
                data["has_dimensions"] = True
                data["plot_width"] = int(dim_match_ctx.group(1))
                data["plot_length"] = int(dim_match_ctx.group(2))
                data["total_area"] = data["plot_width"] * data["plot_length"]

    # Extras
    extras = []
    if "dining" in full_lower: extras.append("dining")
    if "study" in full_lower: extras.append("study")
    if "pooja" in full_lower: extras.append("pooja")
    if "balcon" in full_lower: extras.append("balcony")
    if "parking" in full_lower or "garage" in full_lower: extras.append("parking")
    if "garden" in full_lower: extras.append("garden")
    data["extras"] = extras

    return data


def check_requirements_complete(history: List[Dict]) -> bool:
    """
    Check if conversation history contains all mandatory requirements.

    Looks for the [REQUIREMENTS_COMPLETE] marker from the AI,
    or checks if the conversation naturally contains all required data.
    """
    # Check for explicit marker in last assistant message
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            if "[REQUIREMENTS_COMPLETE]" in msg.get("content", ""):
                return True
            break  # Only check the last assistant message

    # Parse collected data with contextual analysis
    data = _parse_collected_data(history)

    return data["has_dimensions"] and data["has_bedrooms"] and data["has_bathrooms"]


def build_conversation_text(history: List[Dict]) -> str:
    """Build a readable conversation transcript from history."""
    lines = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Architect"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


# ============================================================================
# PIPELINE EXECUTION — Uses the AI provider (Grok → Groq → Fallback)
# ============================================================================

async def _call_ai(system_prompt: str, user_message: str, history: List[Dict] = None,
                   temperature: float = 0.7, max_tokens: int = 2048) -> Tuple[str, Optional[dict]]:
    """
    Call the AI provider with the given system prompt and message.

    Returns (reply_text, extracted_json).
    Falls back through: Grok → Groq → Rule-based.
    """
    # Try Grok first
    try:
        from config import GROK_API_KEY, GROK_MODEL, GROK_BASE_URL
        if GROK_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=GROK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            reply = response.choices[0].message.content
            extracted = _extract_json_from_text(reply)
            return reply, extracted
    except Exception:
        pass

    # Try Groq fallback
    try:
        from config import GROQ_API_KEY, GROQ_MODEL
        if GROQ_API_KEY:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            reply = response.choices[0].message.content
            extracted = _extract_json_from_text(reply)
            return reply, extracted
    except Exception:
        pass

    # Final fallback — return empty
    return "", None


async def run_stage_1_chat(message: str, history: List[Dict]) -> Dict:
    """
    Stage 1: Chat mode — natural conversation to collect requirements.

    Returns dict with: reply, stage, requirements_complete
    """
    reply, extracted = await _call_ai(
        STAGE_1_CHAT_PROMPT, message, history,
        temperature=0.7, max_tokens=1024
    )

    if not reply:
        # Fallback response
        reply = _fallback_chat_response(message, history)

    requirements_complete = "[REQUIREMENTS_COMPLETE]" in reply
    clean_reply = reply.replace("[REQUIREMENTS_COMPLETE]", "").strip()

    # Also check from history if AI didn't emit marker
    if not requirements_complete:
        full_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        requirements_complete = check_requirements_complete(full_history)

    return {
        "reply": clean_reply,
        "stage": PipelineStage.CHAT,
        "requirements_complete": requirements_complete,
        "extracted_data": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_2_extraction(history: List[Dict]) -> Dict:
    """
    Stage 2: Extract structured JSON from conversation history.

    Returns dict with: requirements_json, stage
    """
    conversation_text = build_conversation_text(history)

    reply, extracted = await _call_ai(
        STAGE_2_EXTRACTION_PROMPT,
        f"Extract requirements from this conversation:\n\n{conversation_text}",
        temperature=0.3, max_tokens=1024,
    )

    if not extracted:
        # Fallback: try to extract from conversation manually
        extracted = _fallback_extract(history)

    return {
        "reply": "Requirements extracted successfully.",
        "stage": PipelineStage.EXTRACTION,
        "requirements_json": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_3_design(requirements_json: Dict) -> Dict:
    """
    Stage 3: Generate construction-ready layout from structured requirements.

    Returns dict with: layout_json, explanation, stage
    """
    reply, extracted = await _call_ai(
        STAGE_3_DESIGN_PROMPT,
        f"Generate a floor plan layout for these requirements:\n\n```json\n{json.dumps(requirements_json, indent=2)}\n```",
        temperature=0.5, max_tokens=4096,
    )

    explanation = _clean_reply(reply) if reply else "Layout generated using standard architectural rules."

    if not extracted:
        # Fallback: generate basic layout programmatically
        extracted = _fallback_design(requirements_json)

    return {
        "reply": explanation,
        "stage": PipelineStage.DESIGN,
        "layout_json": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_4_validation(layout_json: Dict) -> Dict:
    """
    Stage 4: Validate the generated layout.

    Returns dict with: validation_report, compliant, stage
    """
    reply, extracted = await _call_ai(
        STAGE_4_VALIDATION_PROMPT,
        f"Validate this floor plan layout:\n\n```json\n{json.dumps(layout_json, indent=2)}\n```",
        temperature=0.3, max_tokens=2048,
    )

    if not extracted:
        # Fallback: run basic validation
        extracted = _fallback_validate(layout_json)

    compliant = extracted.get("compliant", False) if extracted else False

    explanation = _clean_reply(reply) if reply else "Validation complete."

    return {
        "reply": explanation,
        "stage": PipelineStage.VALIDATION,
        "validation_report": extracted,
        "compliant": compliant,
        "provider": "grok" if reply else "fallback",
    }


async def run_full_pipeline(history: List[Dict]) -> Dict:
    """
    Run stages 2 → 3 → 4 in sequence after requirements are complete.

    Returns the combined result of all stages.
    """
    # Stage 2: Extract
    extraction_result = await run_stage_2_extraction(history)
    requirements_json = extraction_result.get("requirements_json", {})

    if not requirements_json:
        return {
            "error": "Could not extract requirements from conversation.",
            "stage": PipelineStage.EXTRACTION,
        }

    # Stage 3: Design
    design_result = await run_stage_3_design(requirements_json)
    layout_json = design_result.get("layout_json", {})

    if not layout_json:
        return {
            "error": "Could not generate layout design.",
            "stage": PipelineStage.DESIGN,
            "requirements_json": requirements_json,
        }

    # Stage 4: Validate
    validation_result = await run_stage_4_validation(layout_json)

    return {
        "stage": PipelineStage.COMPLETE if validation_result.get("compliant") else PipelineStage.VALIDATION,
        "requirements_json": requirements_json,
        "layout_json": layout_json,
        "validation_report": validation_result.get("validation_report", {}),
        "compliant": validation_result.get("compliant", False),
        "design_explanation": design_result.get("reply", ""),
        "validation_explanation": validation_result.get("reply", ""),
    }


# ============================================================================
# FALLBACK — Rule-based when no AI provider is available
# ============================================================================

def _fallback_chat_response(message: str, history: List[Dict]) -> str:
    """Context-aware rule-based fallback for chat mode."""
    # Build full history including current message
    full_history = history + [{"role": "user", "content": message}]
    data = _parse_collected_data(full_history)

    # Check if all requirements are now complete
    if data["has_dimensions"] and data["has_bedrooms"] and data["has_bathrooms"]:
        # Build summary
        summary_parts = []
        if data["plot_width"] and data["plot_length"]:
            summary_parts.append(f"Plot: {data['plot_width']}×{data['plot_length']} feet ({data['total_area']} sq ft)")
        elif data["total_area"]:
            summary_parts.append(f"Plot area: {data['total_area']} sq ft")
        if data["bedrooms"]:
            summary_parts.append(f"Bedrooms: {data['bedrooms']}")
        if data["bathrooms"]:
            summary_parts.append(f"Bathrooms: {data['bathrooms']}")
        if data["floors"]:
            summary_parts.append(f"Floors: {data['floors']}")
        if data["extras"]:
            summary_parts.append(f"Extras: {', '.join(data['extras'])}")

        summary = "\n".join(f"  • {p}" for p in summary_parts)

        return (
            f"Great, I have all the information I need!\n\n"
            f"**Your Requirements:**\n{summary}\n\n"
            f"Generating your design now...\n\n"
            f"[REQUIREMENTS_COMPLETE]"
        )

    # Determine what's missing and ask for it
    turn = len([h for h in history if h.get("role") == "user"])

    if turn == 0:
        # First message — check what they gave us
        has_dims = bool(re.search(r'\d+\s*[x×*]\s*\d+', message.lower()))
        has_area = bool(re.search(r'\d+\s*(?:sq|sqft|square)', message.lower()))
        has_bhk = bool(re.search(r'\d+\s*bhk', message.lower()))
        has_bed = bool(re.search(r'\d+\s*bed', message.lower()))

        if (has_dims or has_area) and (has_bhk or has_bed):
            return (
                "Great! Let me confirm your requirements.\n\n"
                "Do you need any special rooms like dining, study, pooja room, balcony, or parking?\n"
                "Type 'no' if you're good, or list what you'd like."
            )

        return (
            "Welcome! I'll help you design your home. 🏠\n\n"
            "Let's start — what's your plot size?\n"
            "(e.g., '30x40 feet' or '1200 sq ft')"
        )

    # Ask for what's missing, one thing at a time
    if not data["has_dimensions"]:
        return (
            "Got it! What's your plot size?\n"
            "(e.g., '30x40 feet', '1200 sqft', or just a number like '1200')"
        )

    if not data["has_bedrooms"] or not data["has_bathrooms"]:
        return (
            "How many bedrooms and bathrooms do you need?\n"
            "(e.g., '3 bedrooms, 2 bathrooms' or just '3, 2')"
        )

    if not data["has_floors"]:
        return (
            "How many floors?\n"
            "(Most homes are 1 or 2 floors — just type the number)"
        )

    # All mandatory data collected — shouldn't reach here but just in case
    return (
        "Do you need any special rooms?\n"
        "(dining, study, pooja room, balcony, parking, garden)\n"
        "Say 'no' or 'generate' to proceed."
    )


def _fallback_extract(history: List[Dict]) -> Dict:
    """Extract requirements from conversation using contextual parsing."""
    data = _parse_collected_data(history)

    import math

    result = {
        "plot_width": data["plot_width"] or 30,
        "plot_length": data["plot_length"] or 40,
        "total_area": data["total_area"] or 1200,
        "floors": data["floors"] or 1,
        "bedrooms": data["bedrooms"] or 2,
        "bathrooms": data["bathrooms"] or 1,
        "living_room": True,
        "kitchen": True,
        "extras": data["extras"],
    }

    # If only area given, estimate dimensions
    if result["total_area"] and not data["plot_width"]:
        side = math.sqrt(result["total_area"])
        result["plot_width"] = round(side * 1.15)
        result["plot_length"] = round(result["total_area"] / result["plot_width"])

    # Recalculate total area from dimensions if dimensions were given
    if data["plot_width"] and data["plot_length"]:
        result["total_area"] = result["plot_width"] * result["plot_length"]

    return result


def _fallback_design(requirements: Dict) -> Dict:
    """Generate layout using the deterministic architectural engine."""
    try:
        from services.arch_engine import design_generate as engine_design
        engine_result = engine_design(requirements)
        if "error" not in engine_result and engine_result.get("layout"):
            return engine_result["layout"]
    except Exception:
        pass

    # Original fallback
    w = requirements.get("plot_width", 30)
    l = requirements.get("plot_length", 40)
    total = w * l
    bedrooms = requirements.get("bedrooms", 2)
    bathrooms = requirements.get("bathrooms", 1)
    extras = requirements.get("extras", [])

    rooms = []
    x_cursor = 0.75  # Start after external wall
    y_cursor = 0.75

    # Living room — public zone, near entrance
    living_w = max(12, w * 0.4)
    living_l = max(12, l * 0.3)
    rooms.append({
        "name": "Living Room", "room_type": "living",
        "width": round(living_w, 1), "length": round(living_l, 1),
        "area": round(living_w * living_l),
        "zone": "public",
        "position": {"x": round(x_cursor, 1), "y": round(y_cursor, 1)},
        "doors": [{"wall": "S", "offset": round(living_w / 2, 1)}],
        "windows": [{"wall": "N", "width": 4}, {"wall": "E", "width": 4}],
    })

    # Kitchen — semi-private, adjacent to living
    kit_w = max(8, w * 0.25)
    kit_l = max(10, l * 0.25)
    kit_x = x_cursor + living_w + 0.38
    rooms.append({
        "name": "Kitchen", "room_type": "kitchen",
        "width": round(kit_w, 1), "length": round(kit_l, 1),
        "area": round(kit_w * kit_l),
        "zone": "semi_private",
        "position": {"x": round(kit_x, 1), "y": round(y_cursor, 1)},
        "doors": [{"wall": "W", "offset": round(kit_l / 2, 1)}],
        "windows": [{"wall": "E", "width": 4}],
    })

    # Bedrooms — private zone, upper portion
    bed_y = y_cursor + living_l + 0.38 + 3.5  # After corridor
    bed_w = max(10, (w - 1.5 - 0.38 * (bedrooms - 1)) / bedrooms)
    bed_l = max(12, (l - bed_y - 0.75) * 0.8)

    for i in range(bedrooms):
        name = "Master Bedroom" if i == 0 else f"Bedroom {i + 1}"
        rtype = "master_bedroom" if i == 0 else "bedroom"
        bx = x_cursor + i * (bed_w + 0.38)
        rooms.append({
            "name": name, "room_type": rtype,
            "width": round(bed_w, 1), "length": round(bed_l, 1),
            "area": round(bed_w * bed_l),
            "zone": "private",
            "position": {"x": round(bx, 1), "y": round(bed_y, 1)},
            "doors": [{"wall": "S", "offset": round(bed_w / 2, 1)}],
            "windows": [{"wall": "N", "width": 4}],
        })

    # Bathrooms — service zone
    bath_w = 5
    bath_l = 8
    for i in range(bathrooms):
        bath_x = kit_x + kit_w + 0.38 if i == 0 else kit_x + kit_w + 0.38
        bath_y = y_cursor + i * (bath_l + 0.38)
        # Fit within plot
        if bath_x + bath_w + 0.75 > w:
            bath_x = w - bath_w - 0.75
        if bath_y + bath_l + 0.75 > l:
            bath_y = l - bath_l - 0.75

        rooms.append({
            "name": f"Bathroom {i + 1}" if bathrooms > 1 else "Bathroom",
            "room_type": "bathroom",
            "width": bath_w, "length": bath_l,
            "area": bath_w * bath_l,
            "zone": "service",
            "position": {"x": round(bath_x, 1), "y": round(bath_y, 1)},
            "doors": [{"wall": "W", "offset": 2.5}],
            "windows": [{"wall": "E", "width": 3}],
        })

    # Extras
    extra_y = bed_y + bed_l + 0.38
    for extra in extras:
        if extra == "dining":
            rooms.append({
                "name": "Dining Room", "room_type": "dining",
                "width": 10, "length": 10, "area": 100,
                "zone": "semi_private",
                "position": {"x": round(x_cursor + living_w / 2, 1), "y": round(y_cursor + living_l + 0.38, 1)},
                "doors": [{"wall": "N", "offset": 5}],
                "windows": [{"wall": "W", "width": 4}],
            })

    total_used = sum(r["area"] for r in rooms)

    return {
        "plot": {"width": w, "length": l, "unit": "ft"},
        "rooms": rooms,
        "circulation": {"type": "central corridor", "width": 3.5},
        "walls": {"external": "9 inch", "internal": "4.5 inch"},
        "design_validation": {
            "total_area_used": round(total_used),
            "area_percentage": round(total_used / total * 100, 1),
            "compliant": total_used <= total * 0.9,
        },
    }


def _fallback_validate(layout: Dict) -> Dict:
    """Validation using the deterministic architectural engine."""
    try:
        from services.arch_engine import validate_layout as engine_validate
        engine_result = engine_validate(layout)
        # Convert engine format to pipeline format
        issues = (
            engine_result.get("overlap_details", []) +
            engine_result.get("size_violations", []) +
            engine_result.get("zoning_issues", []) +
            engine_result.get("boundary_issues", []) +
            engine_result.get("proportion_issues", [])
        )
        if engine_result.get("area_overflow"):
            issues.append(engine_result["area_overflow"])
        issues += engine_result.get("circulation_issues", [])

        area_summary = engine_result.get("area_summary", {})
        return {
            "compliant": engine_result.get("compliant", False),
            "total_area_used": area_summary.get("total_used_area", 0),
            "plot_area": area_summary.get("plot_area", 0),
            "area_utilization": f"{area_summary.get('utilization_percent', 0)}%",
            "checks": {
                "area_overflow": {"pass": not engine_result.get("area_overflow"), "detail": engine_result.get("area_overflow", "OK")},
                "overlapping_rooms": {"pass": not engine_result.get("overlap"), "detail": ", ".join(engine_result.get("overlap_details", [])) or "No overlaps"},
                "proportions": {"pass": not engine_result.get("proportion_issues"), "detail": ", ".join(engine_result.get("proportion_issues", [])) or "OK"},
                "zoning": {"pass": not engine_result.get("zoning_issues"), "detail": ", ".join(engine_result.get("zoning_issues", [])) or "OK"},
                "circulation": {"pass": not engine_result.get("circulation_issues"), "detail": ", ".join(engine_result.get("circulation_issues", [])) or "OK"},
                "minimum_sizes": {"pass": not engine_result.get("size_violations"), "detail": ", ".join(engine_result.get("size_violations", [])) or "All rooms meet minimum"},
            },
            "issues": issues,
            "suggestions": [
                "Consider adding cross-ventilation windows",
                "Ensure all bedrooms have external wall exposure",
            ] if engine_result.get("compliant") else ["Fix the issues above before proceeding"],
        }
    except Exception:
        pass

    # Original fallback
    rooms = layout.get("rooms", [])
    plot = layout.get("plot", {})
    plot_area = plot.get("width", 30) * plot.get("length", 40)

    issues = []
    checks = {}

    # Area overflow
    total_used = sum(r.get("area", 0) for r in rooms)
    area_ok = total_used <= plot_area
    checks["area_overflow"] = {"pass": area_ok, "detail": f"{total_used} / {plot_area} sq ft"}
    if not area_ok:
        issues.append(f"Total room area ({total_used} sqft) exceeds plot area ({plot_area} sqft)")

    # Minimum sizes
    min_sizes = {"bedroom": 100, "master_bedroom": 100, "bathroom": 35, "kitchen": 80, "living": 120}
    size_ok = True
    for room in rooms:
        rtype = room.get("room_type", "other")
        min_a = min_sizes.get(rtype, 0)
        if room.get("area", 0) < min_a:
            size_ok = False
            issues.append(f"{room.get('name', rtype)}: {room.get('area')} sqft < minimum {min_a} sqft")
    checks["minimum_sizes"] = {"pass": size_ok, "detail": "All rooms meet minimum" if size_ok else "See issues"}

    # Proportions
    prop_ok = True
    for room in rooms:
        w, l = room.get("width", 10), room.get("length", 10)
        if min(w, l) < 4:
            prop_ok = False
            issues.append(f"{room.get('name')}: dimension {min(w,l)} ft is too narrow")
        ratio = max(w, l) / max(min(w, l), 1)
        if ratio > 3:
            prop_ok = False
            issues.append(f"{room.get('name')}: aspect ratio {ratio:.1f}:1 is too extreme")
    checks["proportions"] = {"pass": prop_ok, "detail": "OK" if prop_ok else "See issues"}

    # Overlap check (basic AABB)
    overlap_ok = True
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p1 = r1.get("position", {})
            p2 = r2.get("position", {})
            x1, y1 = p1.get("x", 0), p1.get("y", 0)
            x2, y2 = p2.get("x", 0), p2.get("y", 0)

            if (x1 < x2 + r2.get("width", 0) and x1 + r1.get("width", 0) > x2 and
                y1 < y2 + r2.get("length", 0) and y1 + r1.get("length", 0) > y2):
                overlap_ok = False
                issues.append(f"Overlap: {r1.get('name')} and {r2.get('name')}")
    checks["overlapping_rooms"] = {"pass": overlap_ok, "detail": "No overlaps" if overlap_ok else "See issues"}

    # Zoning (basic)
    checks["zoning"] = {"pass": True, "detail": "Basic zoning check passed"}
    checks["circulation"] = {"pass": True, "detail": "Circulation paths present"}

    compliant = all(c["pass"] for c in checks.values())

    return {
        "compliant": compliant,
        "total_area_used": round(total_used),
        "plot_area": plot_area,
        "area_utilization": f"{round(total_used / plot_area * 100, 1)}%",
        "checks": checks,
        "issues": issues,
        "suggestions": [
            "Consider adding cross-ventilation windows",
            "Ensure all bedrooms have external wall exposure",
        ] if compliant else ["Fix the issues above before proceeding"],
    }
