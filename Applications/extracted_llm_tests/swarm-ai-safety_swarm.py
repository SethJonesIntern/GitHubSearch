# swarm-ai-safety/swarm
# 8 LLM-backed test functions across 210 test files
# Source: https://github.com/swarm-ai-safety/swarm

# --- tests/test_langgraph_study.py ---

    def test_handles_empty_messages(self) -> None:
        completed, agent = detect_task_completed([])
        assert completed is False
        assert agent == "none"

    def test_handles_dict_messages(self) -> None:
        messages = [
            {"content": "Working...", "name": "writer"},
            {"content": "FINAL ANSWER: Done.", "name": "coordinator"},
        ]
        completed, agent = detect_task_completed(messages)
        assert completed is True

    def test_empty_provenance(self) -> None:
        logger = ProvenanceLogger()
        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 0
        assert analysis["risk_level"] == "low"

    def test_all_approved(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer"))
        logger.log(self._make_record("writer", "reviewer"))
        logger.log(self._make_record("reviewer", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 4
        assert analysis["approved_handoffs"] == 4
        assert analysis["denied_handoffs"] == 0
        assert analysis["denial_rate"] == 0.0
        assert analysis["risk_level"] == "low"

    def test_with_denials(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer", "denied", 0.8))
        logger.log(self._make_record("researcher", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["total_handoffs"] == 3
        assert analysis["denied_handoffs"] == 1
        assert analysis["denial_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_with_escalation(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer", "escalated", 0.9))

        analysis = analyze_swarm_run(logger)
        assert analysis["escalated_handoffs"] == 1
        assert analysis["risk_level"] in ("high", "critical")

    def test_cycle_detection(self) -> None:
        logger = ProvenanceLogger()
        # Create a writer <-> reviewer ping-pong
        for _ in range(4):
            logger.log(self._make_record("writer", "reviewer"))
            logger.log(self._make_record("reviewer", "writer"))

        analysis = analyze_swarm_run(logger)
        assert len(analysis["cycle_pairs"]) > 0

    def test_chain_depth(self) -> None:
        logger = ProvenanceLogger()
        logger.log(self._make_record("coordinator", "researcher"))
        logger.log(self._make_record("researcher", "writer"))
        logger.log(self._make_record("writer", "reviewer"))
        logger.log(self._make_record("reviewer", "coordinator"))

        analysis = analyze_swarm_run(logger)
        assert analysis["max_chain_depth"] == 3  # 0-indexed, 4th record is depth 3

