"""Automated tests for the intake and outcome matching module."""

import unittest

import pandas as pd

from intake_outcome_matching import (
    match_outcomes_to_intakes,
    summarize_matches,
)


class IntakeOutcomeMatchingTests(unittest.TestCase):
    """Test the grouped temporal matching behavior."""

    def test_one_intake_matches_one_outcome(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A100",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "First Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A100",
                    "datetime": pd.Timestamp("2024-01-03 08:00:00"),
                    "outcome_type": "Adoption",
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "matched")
        self.assertEqual(
            results.loc[0, "found_location"],
            "First Location",
        )
        self.assertEqual(
            results.loc[0, "intake_datetime"],
            pd.Timestamp("2024-01-01 08:00:00"),
        )
        self.assertEqual(
            results.loc[0, "length_of_stay_days"],
            2.0,
        )

    def test_latest_eligible_intake_is_selected(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A200",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Older Location",
                },
                {
                    "animal_id": "A200",
                    "datetime": pd.Timestamp("2024-02-01 08:00:00"),
                    "found_location": "Newest Eligible Location",
                },
                {
                    "animal_id": "A200",
                    "datetime": pd.Timestamp("2024-03-01 08:00:00"),
                    "found_location": "Future Location",
                },
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A200",
                    "datetime": pd.Timestamp("2024-02-15 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(
            results.loc[0, "found_location"],
            "Newest Eligible Location",
        )
        self.assertEqual(
            results.loc[0, "intake_datetime"],
            pd.Timestamp("2024-02-01 08:00:00"),
        )

    def test_future_intake_is_not_matched(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A300",
                    "datetime": pd.Timestamp("2024-04-01 08:00:00"),
                    "found_location": "Future Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A300",
                    "datetime": pd.Timestamp("2024-03-01 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "unmatched")
        self.assertEqual(
            results.loc[0, "unmatched_reason"],
            "no_earlier_intake",
        )
        self.assertTrue(
            pd.isna(results.loc[0, "intake_datetime"])
        )
        self.assertTrue(
            pd.isna(results.loc[0, "found_location"])
        )

    def test_outcome_without_intake_is_unmatched(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A400",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Different Animal Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A500",
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "unmatched")
        self.assertEqual(
            results.loc[0, "unmatched_reason"],
            "no_earlier_intake",
        )

    def test_missing_animal_id_is_retained_and_marked(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A600",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Known Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": pd.NA,
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "unmatched")
        self.assertEqual(
            results.loc[0, "unmatched_reason"],
            "missing_animal_id",
        )
        self.assertTrue(
            pd.isna(results.loc[0, "found_location"])
        )

    def test_invalid_outcome_datetime_is_retained_and_marked(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A700",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Known Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A700",
                    "datetime": pd.NaT,
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "unmatched")
        self.assertEqual(
            results.loc[0, "unmatched_reason"],
            "invalid_outcome_datetime",
        )

    def test_invalid_intake_datetime_is_excluded(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A800",
                    "datetime": pd.NaT,
                    "found_location": "Invalid Intake Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A800",
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)
        summary = summarize_matches(results)

        self.assertEqual(results.loc[0, "match_status"], "unmatched")
        self.assertEqual(
            results.loc[0, "unmatched_reason"],
            "no_earlier_intake",
        )
        self.assertEqual(summary["invalid_intake_datetime"], 1)
        self.assertEqual(summary["excluded_intakes"], 1)

    def test_exact_timestamp_match_is_allowed(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A900",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Exact Match Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A900",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(results.loc[0, "match_status"], "matched")
        self.assertEqual(
            results.loc[0, "length_of_stay_days"],
            0.0,
        )

    def test_duplicate_intake_timestamp_uses_last_source_row(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A1000",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "First Duplicate Location",
                },
                {
                    "animal_id": "A1000",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Last Duplicate Location",
                },
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A1000",
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                }
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)

        self.assertEqual(
            results.loc[0, "found_location"],
            "Last Duplicate Location",
        )

    def test_summary_reports_matched_and_unmatched_counts(self) -> None:
        intakes = pd.DataFrame(
            [
                {
                    "animal_id": "A1100",
                    "datetime": pd.Timestamp("2024-01-01 08:00:00"),
                    "found_location": "Matched Location",
                }
            ]
        )

        outcomes = pd.DataFrame(
            [
                {
                    "animal_id": "A1100",
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                },
                {
                    "animal_id": "A1200",
                    "datetime": pd.Timestamp("2024-01-02 08:00:00"),
                },
            ]
        )

        results = match_outcomes_to_intakes(intakes, outcomes)
        summary = summarize_matches(results)

        self.assertEqual(summary["total_outcomes"], 2)
        self.assertEqual(summary["matched_outcomes"], 1)
        self.assertEqual(summary["unmatched_outcomes"], 1)
        self.assertEqual(
            summary["unmatched_reasons"],
            {"no_earlier_intake": 1},
        )


if __name__ == "__main__":
    unittest.main()