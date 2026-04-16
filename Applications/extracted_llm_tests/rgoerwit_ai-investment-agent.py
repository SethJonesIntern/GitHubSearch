# rgoerwit/ai-investment-agent
# 14 LLM-backed test functions across 165 test files
# Source: https://github.com/rgoerwit/ai-investment-agent

# --- tests/agents/test_consultant_edge_cases.py ---

    def test_report_includes_consultant_review(self):
        """Test that generated report includes consultant section."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis here",
            "sentiment_report": "Sentiment analysis",
            "news_report": "News analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "CONSULTANT REVIEW: APPROVED\n\nAnalysis is sound.",
            "trader_investment_plan": "Trading plan",
            "final_trade_decision": "FINAL DECISION: BUY\n\nRationale: Good fundamentals.",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "External Consultant Review" in report
        assert "CONSULTANT REVIEW: APPROVED" in report

    def test_report_excludes_consultant_error(self):
        """Test that report excludes consultant review if it's an error."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "Consultant Review Error: OpenAI API timeout",
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should NOT include consultant section if it's an error
        assert "External Consultant Review" not in report

    def test_report_excludes_consultant_na(self):
        """Test that report excludes consultant review if N/A (disabled)."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            "consultant_review": "N/A (consultant disabled or unavailable)",
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should NOT include consultant section if N/A
        assert "External Consultant Review" not in report

    def test_report_handles_missing_consultant_field(self):
        """Test report generation when consultant_review field missing entirely."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "BUY recommendation",
            # consultant_review field missing entirely
            "final_trade_decision": "FINAL DECISION: BUY",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Should not crash, should generate valid report
        assert "BUY" in report
        assert "TEST" in report
        assert "External Consultant Review" not in report

    def test_rendered_report_strips_reasoning_dict_repr_single_quotes(self):
        """Single-quote Python repr of a reasoning dict is stripped from rendered output (A3)."""
        leaked_line = "{'id': 'rs_abc123', 'summary': [], 'type': 'reasoning'}"
        review = f"Analysis is thorough and well-supported.\n{leaked_line}\nRisk is acceptable."
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "rs_abc123" not in report
        assert "Analysis is thorough" in report
        assert "Risk is acceptable" in report

    def test_rendered_report_strips_reasoning_dict_repr_double_quotes(self):
        """JSON-style double-quote repr of a reasoning dict is stripped from rendered output (A3)."""
        leaked_line = '{"id": "rs_xyz999", "summary": [], "type": "reasoning"}'
        review = f"Cross-validation complete.\n{leaked_line}\nNo anomalies found."
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "rs_xyz999" not in report
        assert "Cross-validation complete" in report
        assert "No anomalies found" in report

    def test_rendered_report_preserves_legitimate_review_text(self):
        """Legitimate review prose is preserved when a leaked reasoning line is stripped."""
        leaked_line = "{'id': 'rs_abc123', 'summary': [], 'type': 'reasoning'}"
        review = (
            "The analysis correctly identifies key risks.\n"
            f"{leaked_line}\n"
            "Valuation metrics support the BUY recommendation."
        )
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert "correctly identifies key risks" in report
        assert "Valuation metrics support" in report

    def test_sanitizer_does_not_strip_unrelated_dict_like_lines(self):
        """Lines that look like dicts but lack rs_ prefix and reasoning type are preserved."""
        review = (
            'Analysis summary: {"key": "value", "score": 95}\nConclusion: strong BUY.'
        )
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")
        result = {
            "company_of_interest": "TEST",
            "consultant_review": review,
            "final_trade_decision": "FINAL DECISION: BUY",
        }
        report = reporter.generate_report(result, brief_mode=False)
        assert '{"key": "value"' in report or '"key": "value"' in report
        assert "Conclusion: strong BUY" in report

    def test_report_omits_research_manager_recommendation_when_pm_decision_exists(self):
        """The public report should not publish a second recommendation section."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "investment_plan": "INVESTMENT RECOMMENDATION: HOLD",
            "final_trade_decision": "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "Investment Recommendation" not in report
        assert "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE" in report

    def test_report_surfaces_verification_caveats_before_executive_summary(self):
        """Consultant disputes should be elevated before the main writeup."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "consultant_review": (
                "CONSULTANT REVIEW: CONDITIONAL\n\n"
                "The insider-selling claim is unsubstantiated.\n"
                "The 100 new vessels claim is likely wrong."
            ),
            "artifact_statuses": {
                "consultant_review": {"complete": True, "ok": False},
            },
            "final_trade_decision": "FINAL DECISION: HOLD",
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert "## Verification Caveats" in report
        assert "insider-selling claim is unsubstantiated" in report
        assert report.index("## Verification Caveats") < report.index(
            "## Executive Summary"
        )

    def test_report_rewrites_false_consultant_unavailable_claim(self):
        """Public report should not claim the consultant was unavailable when review exists."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "market_report": "Market analysis",
            "fundamentals_report": "Fundamentals",
            "consultant_review": "CONSULTANT REVIEW: CONDITIONAL APPROVAL\n\nCoverage gaps remain.",
            "artifact_statuses": {
                "consultant_review": {"complete": True, "ok": False},
            },
            "final_trade_decision": (
                'DECISION RATIONALE: The pre-screening flagged a "Consultant Conditional" '
                "warning, but as the external consultant was unavailable to provide "
                "specific conditions, the verified `DATA_BLOCK` fundamentals and moat "
                "signals take absolute precedence."
            ),
        }

        report = reporter.generate_report(result, brief_mode=False)

        assert (
            "external consultant was unavailable to provide specific conditions"
            not in report
        )
        assert "tool-coverage gaps" in report

    def test_report_repairs_glued_structured_block_boundary_before_demoting_headers(
        self,
    ):
        """Rendered reports should clean older glued block boundaries without other changes."""
        reporter = QuietModeReporter(ticker="TEST", company_name="Test Company")

        result = {
            "company_of_interest": "TEST",
            "fundamentals_report": (
                "### --- START DATA_BLOCK ---\n"
                "SECTOR: Energy\n"
                "### --- END DATA_BLOCK ---### FINANCIAL HEALTH DETAIL\n"
                "**Score**: 9/12\n"
            ),
            "final_trade_decision": "PORTFOLIO MANAGER VERDICT: HOLD",
        }

        report = reporter.generate_report(result, brief_mode=False)

        # Both the DATA_BLOCK content and the prose section must survive
        assert "SECTOR: Energy" in report
        assert "FINANCIAL HEALTH DETAIL" in report
        # DATA_BLOCK is repositioned to end of section — prose appears before it
        assert report.index("FINANCIAL HEALTH DETAIL") < report.index(
            "START DATA_BLOCK"
        )

    def test_token_tracker_has_openai_pricing(self):
        """Test that token tracker includes OpenAI model pricing."""
        from src.token_tracker import TokenUsage

        # Test gpt-4o pricing
        usage = TokenUsage(
            timestamp="2025-12-13",
            agent_name="Consultant",
            model_name="gpt-4o",
            prompt_tokens=4000,
            completion_tokens=800,
            total_tokens=4800,
        )

        cost = usage.estimated_cost_usd

        # gpt-4o: $2.50/1M input, $10.00/1M output
        # 4000 * 2.50 / 1M + 800 * 10.00 / 1M = 0.01 + 0.008 = 0.018
        expected_cost = (4000 * 2.50 / 1_000_000) + (800 * 10.00 / 1_000_000)
        assert abs(cost - expected_cost) < 0.001

    def test_token_tracker_has_gpt4o_mini_pricing(self):
        """Test token tracker pricing for gpt-4o-mini."""
        from src.token_tracker import TokenUsage

        usage = TokenUsage(
            timestamp="2025-12-13",
            agent_name="Consultant",
            model_name="gpt-4o-mini",
            prompt_tokens=100000,
            completion_tokens=50000,
            total_tokens=150000,
        )

        cost = usage.estimated_cost_usd

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        # 100k * 0.15 / 1M + 50k * 0.60 / 1M = 0.015 + 0.030 = 0.045
        expected_cost = (100000 * 0.15 / 1_000_000) + (50000 * 0.60 / 1_000_000)
        assert abs(cost - expected_cost) < 0.001

