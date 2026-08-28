from run_prompt5_e01_r2d1c_r3a_loop_closure_counter_reprocess import (
    effective_terminal_boundary,
    should_start_focused_capture,
)


def test_normal_terminal_route_trigger_band():
    assert effective_terminal_boundary(72, False, None) == 72
    assert should_start_focused_capture(67, 72) is True
    assert should_start_focused_capture(66, 72) is False


def test_duplicate_closure_route_uses_last_unique_preclosure():
    assert effective_terminal_boundary(97, True, 96) == 96
    assert should_start_focused_capture(91, 97, True, 96) is True
    assert should_start_focused_capture(90, 97, True, 96) is False


def test_string_sequence_input():
    assert should_start_focused_capture("91", "97", True, "96") is True


def test_null_sequence_is_not_triggered():
    assert should_start_focused_capture(None, 72) is False
    assert should_start_focused_capture(70, None) is False


def test_multiple_vehicle_route_inputs_remain_route_local():
    rows = [
        {"current_sequence": 91, "terminal_sequence": 97, "duplicate_loop_closure": True, "last_unique_preclosure_sequence": 96},
        {"current_sequence": 66, "terminal_sequence": 72, "duplicate_loop_closure": False, "last_unique_preclosure_sequence": None},
    ]
    assert [should_start_focused_capture(**row) for row in rows] == [True, False]
