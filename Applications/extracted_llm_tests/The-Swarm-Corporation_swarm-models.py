# The-Swarm-Corporation/swarm-models
# 3 LLM-backed test functions across 19 test files
# Source: https://github.com/The-Swarm-Corporation/swarm-models

# --- tests/test_fuyu.py ---

def test_run_invalid_text_input(fuyu_instance):
    with pytest.raises(Exception):
        fuyu_instance.run(None, "valid/path/to/image.png")

def test_run_empty_text_input(fuyu_instance):
    with pytest.raises(Exception):
        fuyu_instance.run("", "valid/path/to/image.png")

def test_run_very_long_text_input(fuyu_instance):
    with pytest.raises(Exception):
        fuyu_instance.run("A" * 10000, "valid/path/to/image.png")

