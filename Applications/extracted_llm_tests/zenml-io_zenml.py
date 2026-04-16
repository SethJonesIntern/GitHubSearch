# zenml-io/zenml
# 1 LLM-backed test functions across 249 test files
# Source: https://github.com/zenml-io/zenml

# --- tests/integration/integrations/huggingface/steps/test_accelerate_runner.py ---

def test_accelerate_runner_fails_on_functional_use(clean_client):
    """Tests whether the run_with_accelerate wrapper works as expected."""

    @pipeline(enable_cache=False)
    def train_pipe():
        _ = run_with_accelerate(train, num_processes=2, use_cpu=True)

    with pytest.raises(RuntimeError):
        train_pipe()

