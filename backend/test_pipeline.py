"""
Tests for the 4-Stage AI Architectural Pipeline.

Tests the pipeline logic, requirement checking, extraction fallback,
design fallback, and validation fallback — all without requiring API keys.
"""
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from services.ai_pipeline import (
    PipelineStage,
    check_requirements_complete,
    _parse_collected_data,
    _fallback_extract,
    _fallback_design,
    _fallback_validate,
    _fallback_chat_response,
    _extract_json_from_text,
    build_conversation_text,
)


# ============================================================================
# Stage 1: Requirement completeness checking
# ============================================================================

class TestCheckRequirementsComplete:
    def test_complete_with_marker(self):
        history = [
            {"role": "user", "content": "I want 3BHK on 30x40 plot"},
            {"role": "assistant", "content": "Great! All info collected. [REQUIREMENTS_COMPLETE]"},
        ]
        assert check_requirements_complete(history) is True

    def test_complete_from_conversation(self):
        history = [
            {"role": "user", "content": "30x40 feet plot, 3 bedrooms, 2 bathrooms"},
        ]
        assert check_requirements_complete(history) is True

    def test_bhk_implies_bed_bath(self):
        history = [
            {"role": "user", "content": "I have a 1200 sqft plot and need 3BHK"},
        ]
        assert check_requirements_complete(history) is True

    def test_incomplete_no_dimensions(self):
        history = [
            {"role": "user", "content": "I need 3 bedrooms and 2 bathrooms"},
        ]
        assert check_requirements_complete(history) is False

    def test_incomplete_no_rooms(self):
        history = [
            {"role": "user", "content": "My plot is 30x40 feet"},
        ]
        assert check_requirements_complete(history) is False

    def test_area_instead_of_dimensions(self):
        history = [
            {"role": "user", "content": "1200 sq ft, 2 bedrooms, 1 bathroom"},
        ]
        assert check_requirements_complete(history) is True

    def test_empty_history(self):
        assert check_requirements_complete([]) is False

    def test_star_dimension_format(self):
        """User types 40*30 instead of 40x30"""
        history = [
            {"role": "user", "content": "40*30 plot, 3 bedrooms, 2 bathrooms"},
        ]
        assert check_requirements_complete(history) is True

    def test_contextual_comma_separated(self):
        """User answers '2,2' when asked about bedrooms and bathrooms"""
        history = [
            {"role": "user", "content": "1200 sqft"},
            {"role": "assistant", "content": "How many bedrooms and bathrooms do you need?"},
            {"role": "user", "content": "2,2"},
        ]
        assert check_requirements_complete(history) is True

    def test_standalone_large_number_as_area(self):
        """User types just '1700' as plot area"""
        history = [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "What's your plot size?"},
            {"role": "user", "content": "1700"},
            {"role": "assistant", "content": "How many bedrooms and bathrooms?"},
            {"role": "user", "content": "3, 2"},
        ]
        assert check_requirements_complete(history) is True

    def test_real_user_flow(self):
        """Simulates the exact flow the user reported as broken"""
        history = [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "Welcome! Let's start — what's your plot size?"},
            {"role": "user", "content": "1200 sqft"},
            {"role": "assistant", "content": "Got it! How many bedrooms and bathrooms do you need?"},
            {"role": "user", "content": "2,2"},
            {"role": "assistant", "content": "Perfect. Do you need any special rooms?"},
            {"role": "user", "content": "garden, study"},
            {"role": "assistant", "content": "Excellent! How many floors?"},
            {"role": "user", "content": "3"},
        ]
        assert check_requirements_complete(history) is True


# ============================================================================
# Stage 2: Extraction fallback
# ============================================================================

class TestFallbackExtract:
    def test_extract_dimensions(self):
        history = [
            {"role": "user", "content": "30x40 plot, 3 bedrooms, 2 bathrooms"},
        ]
        result = _fallback_extract(history)
        assert result["plot_width"] == 30
        assert result["plot_length"] == 40
        assert result["total_area"] == 1200
        assert result["bedrooms"] == 3
        assert result["bathrooms"] == 2

    def test_extract_bhk(self):
        history = [
            {"role": "user", "content": "1500 sqft 4bhk house"},
        ]
        result = _fallback_extract(history)
        assert result["total_area"] == 1500
        assert result["bedrooms"] == 4
        assert result["bathrooms"] >= 1

    def test_extract_extras(self):
        history = [
            {"role": "user", "content": "30x40, 2 bed, 1 bath, with dining, pooja room, balcony, parking"},
        ]
        result = _fallback_extract(history)
        assert "dining" in result["extras"]
        assert "pooja" in result["extras"]
        assert "balcony" in result["extras"]
        assert "parking" in result["extras"]

    def test_defaults(self):
        history = [
            {"role": "user", "content": "small house"},
        ]
        result = _fallback_extract(history)
        assert result["living_room"] is True
        assert result["kitchen"] is True
        assert result["floors"] == 1


# ============================================================================
# Stage 3: Design fallback
# ============================================================================

class TestFallbackDesign:
    def test_basic_layout(self):
        req = {
            "plot_width": 30,
            "plot_length": 40,
            "total_area": 1200,
            "bedrooms": 2,
            "bathrooms": 1,
            "extras": [],
        }
        result = _fallback_design(req)

        assert "plot" in result
        assert result["plot"]["width"] == 30
        assert result["plot"]["length"] == 40
        assert "rooms" in result
        assert len(result["rooms"]) >= 4  # Living, Kitchen, 2 Bedrooms, 1 Bath

        # Check room types present
        room_types = [r["room_type"] for r in result["rooms"]]
        assert "living" in room_types
        assert "kitchen" in room_types
        assert "master_bedroom" in room_types or "bedroom" in room_types
        assert "bathroom" in room_types

    def test_has_positions(self):
        req = {
            "plot_width": 30,
            "plot_length": 40,
            "total_area": 1200,
            "bedrooms": 1,
            "bathrooms": 1,
            "extras": [],
        }
        result = _fallback_design(req)
        for room in result["rooms"]:
            assert "position" in room
            assert "x" in room["position"]
            assert "y" in room["position"]
            assert room["width"] > 0
            assert room["length"] > 0
            assert room["area"] > 0

    def test_extras_included(self):
        req = {
            "plot_width": 40,
            "plot_length": 50,
            "total_area": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "extras": ["dining"],
        }
        result = _fallback_design(req)
        room_types = [r["room_type"] for r in result["rooms"]]
        assert "dining" in room_types

    def test_output_structure(self):
        req = {
            "plot_width": 30,
            "plot_length": 40,
            "total_area": 1200,
            "bedrooms": 2,
            "bathrooms": 1,
            "extras": [],
        }
        result = _fallback_design(req)
        assert "circulation" in result
        assert "walls" in result
        assert "design_validation" in result
        assert "total_area_used" in result["design_validation"]
        assert "area_percentage" in result["design_validation"]


# ============================================================================
# Stage 4: Validation fallback
# ============================================================================

class TestFallbackValidate:
    def test_valid_layout(self):
        layout = {
            "plot": {"width": 30, "length": 40},
            "rooms": [
                {"name": "Living", "room_type": "living", "width": 14, "length": 12,
                 "area": 168, "position": {"x": 0, "y": 0}},
                {"name": "Kitchen", "room_type": "kitchen", "width": 10, "length": 10,
                 "area": 100, "position": {"x": 15, "y": 0}},
                {"name": "Bedroom", "room_type": "bedroom", "width": 12, "length": 12,
                 "area": 144, "position": {"x": 0, "y": 14}},
                {"name": "Bathroom", "room_type": "bathroom", "width": 8, "length": 6,
                 "area": 48, "position": {"x": 15, "y": 14}},
            ],
        }
        result = _fallback_validate(layout)
        assert result["compliant"] is True
        assert result["total_area_used"] > 0
        assert result["plot_area"] == 1200
        assert len(result["issues"]) == 0

    def test_area_overflow(self):
        layout = {
            "plot": {"width": 10, "length": 10},
            "rooms": [
                {"name": "Living", "room_type": "living", "width": 20, "length": 20,
                 "area": 400, "position": {"x": 0, "y": 0}},
            ],
        }
        result = _fallback_validate(layout)
        assert result["checks"]["area_overflow"]["pass"] is False
        assert result["compliant"] is False

    def test_too_small_room(self):
        layout = {
            "plot": {"width": 30, "length": 40},
            "rooms": [
                {"name": "Living", "room_type": "living", "width": 8, "length": 8,
                 "area": 64, "position": {"x": 0, "y": 0}},
            ],
        }
        result = _fallback_validate(layout)
        assert result["checks"]["minimum_sizes"]["pass"] is False

    def test_overlapping_rooms(self):
        layout = {
            "plot": {"width": 30, "length": 40},
            "rooms": [
                {"name": "Room A", "room_type": "living", "width": 15, "length": 12,
                 "area": 180, "position": {"x": 0, "y": 0}},
                {"name": "Room B", "room_type": "bedroom", "width": 12, "length": 12,
                 "area": 144, "position": {"x": 5, "y": 0}},
            ],
        }
        result = _fallback_validate(layout)
        assert result["checks"]["overlapping_rooms"]["pass"] is False
        assert result["compliant"] is False

    def test_extreme_aspect_ratio(self):
        layout = {
            "plot": {"width": 30, "length": 40},
            "rooms": [
                {"name": "Hallway", "room_type": "hallway", "width": 3, "length": 20,
                 "area": 60, "position": {"x": 0, "y": 0}},
            ],
        }
        result = _fallback_validate(layout)
        assert result["checks"]["proportions"]["pass"] is False


# ============================================================================
# Helpers
# ============================================================================

class TestHelpers:
    def test_extract_json_from_code_block(self):
        text = """Here's the data:
```json
{"total_area": 1200, "bedrooms": 3}
```
Done."""
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["total_area"] == 1200

    def test_extract_json_inline(self):
        text = 'The result is {"total_area": 800}.'
        result = _extract_json_from_text(text)
        assert result is not None
        assert result["total_area"] == 800

    def test_extract_json_none(self):
        result = _extract_json_from_text("No JSON here.")
        assert result is None

    def test_build_conversation_text(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = build_conversation_text(history)
        assert "User: Hello" in text
        assert "Architect: Hi there" in text

    def test_pipeline_stage_enum(self):
        assert PipelineStage.CHAT == "chat"
        assert PipelineStage.EXTRACTION == "extraction"
        assert PipelineStage.DESIGN == "design"
        assert PipelineStage.VALIDATION == "validation"
        assert PipelineStage.COMPLETE == "complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
