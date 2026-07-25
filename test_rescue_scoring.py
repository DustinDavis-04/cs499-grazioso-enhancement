"""
Automated tests for the rescue suitability scoring module.

These tests verify individual scoring rules, candidate filtering,
historical-record handling, ranking behavior, validation, and empty
result handling.
"""

import unittest

import pandas as pd

from rescue_scoring import (
    RESCUE_CRITERIA,
    calculate_maximum_score,
    parse_age_to_weeks,
    score_age,
    score_and_rank_candidates,
    score_breed,
    score_candidate,
    score_sex,
    validate_rescue_type,
)


def create_candidate_dataframe(
    records: list[dict],
) -> pd.DataFrame:
    """
    Create a scoring DataFrame with all required columns.
    """

    default_values = {
        "animal_id": None,
        "outcome_animal_type": "Dog",
        "outcome_breed": None,
        "outcome_age_upon_outcome_in_weeks": None,
        "outcome_age_upon_outcome": None,
        "outcome_sex_upon_outcome": None,
        "outcome_datetime": None,
    }

    completed_records = []

    for record in records:
        completed_record = default_values.copy()
        completed_record.update(record)
        completed_records.append(completed_record)

    return pd.DataFrame(completed_records)


class TestAgeParsing(unittest.TestCase):
    """
    Test conversion of text ages into weeks.
    """

    def test_numeric_age_is_already_weeks(self):
        self.assertEqual(
            parse_age_to_weeks(52),
            52.0,
        )

    def test_years_are_converted_to_weeks(self):
        self.assertEqual(
            parse_age_to_weeks("2 years"),
            104.0,
        )

    def test_months_are_converted_to_weeks(self):
        self.assertAlmostEqual(
            parse_age_to_weeks("10 months"),
            43.45,
            places=2,
        )

    def test_weeks_remain_weeks(self):
        self.assertEqual(
            parse_age_to_weeks("26 weeks"),
            26.0,
        )

    def test_days_are_converted_to_weeks(self):
        self.assertAlmostEqual(
            parse_age_to_weeks("14 days"),
            2.0,
            places=2,
        )

    def test_invalid_age_returns_none(self):
        self.assertIsNone(
            parse_age_to_weeks("unknown")
        )

    def test_missing_age_returns_none(self):
        self.assertIsNone(
            parse_age_to_weeks(None)
        )


class TestBreedScoring(unittest.TestCase):
    """
    Test exact, mixed, unmatched, and missing breed scores.
    """

    def setUp(self):
        self.water_config = RESCUE_CRITERIA[
            "water"
        ]

    def test_exact_preferred_breed_receives_full_points(self):
        score, match_type, reason = score_breed(
            "Labrador Retriever",
            self.water_config,
        )

        self.assertEqual(score, 50)
        self.assertEqual(match_type, "exact")
        self.assertIn(
            "exact preferred breed",
            reason,
        )

    def test_exact_match_is_case_insensitive(self):
        score, match_type, _ = score_breed(
            "labrador retriever",
            self.water_config,
        )

        self.assertEqual(score, 50)
        self.assertEqual(match_type, "exact")

    def test_mixed_preferred_breed_receives_partial_points(self):
        score, match_type, reason = score_breed(
            "Labrador Retriever Mix",
            self.water_config,
        )

        self.assertEqual(score, 40)
        self.assertEqual(match_type, "mixed")
        self.assertIn(
            "Labrador Retriever",
            reason,
        )

    def test_preferred_breed_with_secondary_breed_is_mixed(self):
        score, match_type, _ = score_breed(
            "Labrador Retriever/German Shepherd",
            self.water_config,
        )

        self.assertEqual(score, 40)
        self.assertEqual(match_type, "mixed")

    def test_nonpreferred_breed_receives_zero_points(self):
        score, match_type, _ = score_breed(
            "Beagle",
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(match_type, "none")

    def test_unknown_breed_receives_zero_points(self):
        score, match_type, _ = score_breed(
            "Unknown Mix",
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(match_type, "missing")

    def test_missing_breed_receives_zero_points(self):
        score, match_type, _ = score_breed(
            None,
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(match_type, "missing")


class TestAgeScoring(unittest.TestCase):
    """
    Test rescue age-range scoring.
    """

    def setUp(self):
        self.water_config = RESCUE_CRITERIA[
            "water"
        ]

    def test_age_inside_range_receives_full_points(self):
        score, match_type, _, age_weeks = score_age(
            52,
            None,
            self.water_config,
        )

        self.assertEqual(score, 30)
        self.assertEqual(
            match_type,
            "within_range",
        )
        self.assertEqual(age_weeks, 52.0)

    def test_minimum_boundary_receives_full_points(self):
        score, match_type, _, _ = score_age(
            26,
            None,
            self.water_config,
        )

        self.assertEqual(score, 30)
        self.assertEqual(
            match_type,
            "within_range",
        )

    def test_maximum_boundary_receives_full_points(self):
        score, match_type, _, _ = score_age(
            156,
            None,
            self.water_config,
        )

        self.assertEqual(score, 30)
        self.assertEqual(
            match_type,
            "within_range",
        )

    def test_age_below_range_receives_zero_points(self):
        score, match_type, _, _ = score_age(
            25,
            None,
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(
            match_type,
            "outside_range",
        )

    def test_age_above_range_receives_zero_points(self):
        score, match_type, _, _ = score_age(
            157,
            None,
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(
            match_type,
            "outside_range",
        )

    def test_text_age_is_used_when_numeric_age_is_missing(self):
        score, match_type, _, age_weeks = score_age(
            None,
            "2 years",
            self.water_config,
        )

        self.assertEqual(score, 30)
        self.assertEqual(
            match_type,
            "within_range",
        )
        self.assertEqual(age_weeks, 104.0)

    def test_missing_age_receives_zero_points(self):
        score, match_type, _, age_weeks = score_age(
            None,
            None,
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(match_type, "missing")
        self.assertIsNone(age_weeks)


class TestSexScoring(unittest.TestCase):
    """
    Test preferred sex and intact-status scoring.
    """

    def setUp(self):
        self.water_config = RESCUE_CRITERIA[
            "water"
        ]

    def test_preferred_sex_receives_full_points(self):
        score, match_type, _ = score_sex(
            "Intact Male",
            self.water_config,
        )

        self.assertEqual(score, 20)
        self.assertEqual(
            match_type,
            "preferred",
        )

    def test_preferred_sex_is_case_insensitive(self):
        score, match_type, _ = score_sex(
            "intact male",
            self.water_config,
        )

        self.assertEqual(score, 20)
        self.assertEqual(
            match_type,
            "preferred",
        )

    def test_nonpreferred_sex_receives_zero_points(self):
        score, match_type, _ = score_sex(
            "Neutered Male",
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(
            match_type,
            "not_preferred",
        )

    def test_missing_sex_receives_zero_points(self):
        score, match_type, _ = score_sex(
            None,
            self.water_config,
        )

        self.assertEqual(score, 0)
        self.assertEqual(match_type, "missing")


class TestCandidateScoring(unittest.TestCase):
    """
    Test complete scoring of individual candidates.
    """

    def test_perfect_water_candidate_scores_100(self):
        candidate = pd.Series(
            {
                "animal_id": "A100001",
                "outcome_animal_type": "Dog",
                "outcome_breed": (
                    "Labrador Retriever"
                ),
                "outcome_age_upon_outcome_in_weeks": 52,
                "outcome_age_upon_outcome": "1 year",
                "outcome_sex_upon_outcome": (
                    "Intact Male"
                ),
                "outcome_datetime": (
                    "2025-01-01 12:00:00"
                ),
            }
        )

        result = score_candidate(
            candidate,
            "water",
        )

        self.assertEqual(
            result["breed_score"],
            50,
        )
        self.assertEqual(
            result["age_score"],
            30,
        )
        self.assertEqual(
            result["sex_score"],
            20,
        )
        self.assertEqual(
            result["total_score"],
            100,
        )
        self.assertEqual(
            result["rescue_type"],
            "water",
        )
        self.assertIn(
            "Total suitability score: 100",
            result["score_explanation"],
        )

    def test_mixed_breed_candidate_scores_90(self):
        candidate = pd.Series(
            {
                "animal_id": "A100002",
                "outcome_animal_type": "Dog",
                "outcome_breed": (
                    "Labrador Retriever Mix"
                ),
                "outcome_age_upon_outcome_in_weeks": 52,
                "outcome_age_upon_outcome": "1 year",
                "outcome_sex_upon_outcome": (
                    "Intact Male"
                ),
                "outcome_datetime": (
                    "2025-01-01 12:00:00"
                ),
            }
        )

        result = score_candidate(
            candidate,
            "water",
        )

        self.assertEqual(
            result["breed_score"],
            40,
        )
        self.assertEqual(
            result["total_score"],
            90,
        )

    def test_maximum_score_is_derived_from_configuration(self):
        maximum_score = calculate_maximum_score(
            RESCUE_CRITERIA["water"]
        )

        self.assertEqual(maximum_score, 100)


class TestCandidateFilteringAndRanking(unittest.TestCase):
    """
    Test filtering, duplicate handling, and ranking.
    """

    def test_non_dogs_are_excluded(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A200001",
                    "outcome_animal_type": "Dog",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
                {
                    "animal_id": "A200002",
                    "outcome_animal_type": "Cat",
                    "outcome_breed": "Domestic Shorthair",
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
            ]
        )

        result = score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.iloc[0]["animal_id"],
            "A200001",
        )

        summary = result.attrs[
            "rescue_scoring"
        ]

        self.assertEqual(
            summary["input_record_count"],
            2,
        )
        self.assertEqual(
            summary[
                "excluded_animal_type_count"
            ],
            1,
        )
        self.assertEqual(
            summary["eligible_record_count"],
            1,
        )
        self.assertEqual(
            summary["candidate_count"],
            1,
        )

    def test_latest_record_is_kept_for_duplicate_animal(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A300001",
                    "outcome_animal_type": "Dog",
                    "outcome_breed": "Beagle",
                    "outcome_age_upon_outcome_in_weeks": 20,
                    "outcome_sex_upon_outcome": (
                        "Neutered Male"
                    ),
                    "outcome_datetime": (
                        "2024-01-01 10:00:00"
                    ),
                },
                {
                    "animal_id": "A300001",
                    "outcome_animal_type": "Dog",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": (
                        "2025-01-01 10:00:00"
                    ),
                },
            ]
        )

        result = score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.iloc[0]["outcome_breed"],
            "Labrador Retriever",
        )
        self.assertEqual(
            result.iloc[0]["total_score"],
            100,
        )

        summary = result.attrs[
            "rescue_scoring"
        ]

        self.assertEqual(
            summary["eligible_record_count"],
            2,
        )
        self.assertEqual(
            summary["duplicate_record_count"],
            1,
        )
        self.assertEqual(
            summary["candidate_count"],
            1,
        )

    def test_rankings_are_ordered_by_total_score(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A400003",
                    "outcome_breed": "Beagle",
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Neutered Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
                {
                    "animal_id": "A400001",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
                {
                    "animal_id": "A400002",
                    "outcome_breed": (
                        "Labrador Retriever Mix"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
            ]
        )

        result = score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertEqual(
            result["animal_id"].tolist(),
            [
                "A400001",
                "A400002",
                "A400003",
            ],
        )

        self.assertEqual(
            result["total_score"].tolist(),
            [100, 90, 30],
        )

        self.assertEqual(
            result["rescue_rank"].tolist(),
            [1, 2, 3],
        )

    def test_animal_id_breaks_complete_score_tie(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A500003",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
                {
                    "animal_id": "A500001",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
                {
                    "animal_id": "A500002",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
            ]
        )

        result = score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertEqual(
            result["animal_id"].tolist(),
            [
                "A500001",
                "A500002",
                "A500003",
            ],
        )

    def test_original_dataframe_is_not_modified(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A600001",
                    "outcome_breed": (
                        "Labrador Retriever"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
            ]
        )

        original_columns = dataframe.columns.tolist()

        score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertEqual(
            dataframe.columns.tolist(),
            original_columns,
        )
        self.assertNotIn(
            "total_score",
            dataframe.columns,
        )


class TestMissionConfiguration(unittest.TestCase):
    """
    Test mission selection and validation.
    """

    def test_all_three_rescue_types_exist(self):
        self.assertEqual(
            set(RESCUE_CRITERIA),
            {
                "water",
                "mountain",
                "disaster",
            },
        )

    def test_rescue_type_is_case_insensitive(self):
        result = validate_rescue_type(
            "WATER"
        )

        self.assertEqual(
            result["display_name"],
            "Water Rescue",
        )

    def test_invalid_rescue_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            validate_rescue_type(
                "urban"
            )

    def test_nonstring_rescue_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            validate_rescue_type(
                123
            )


class TestEmptyResults(unittest.TestCase):
    """
    Test scoring when no eligible candidates are available.
    """

    def test_dataframe_with_only_cats_returns_empty_result(self):
        dataframe = create_candidate_dataframe(
            [
                {
                    "animal_id": "A700001",
                    "outcome_animal_type": "Cat",
                    "outcome_breed": (
                        "Domestic Shorthair"
                    ),
                    "outcome_age_upon_outcome_in_weeks": 52,
                    "outcome_sex_upon_outcome": (
                        "Intact Male"
                    ),
                    "outcome_datetime": "2025-01-01",
                },
            ]
        )

        result = score_and_rank_candidates(
            dataframe,
            "water",
        )

        self.assertTrue(result.empty)

        expected_score_columns = {
            "breed_score",
            "age_score",
            "sex_score",
            "total_score",
            "rescue_rank",
            "score_explanation",
        }

        self.assertTrue(
            expected_score_columns.issubset(
                result.columns
            )
        )

        summary = result.attrs[
            "rescue_scoring"
        ]

        self.assertEqual(
            summary["candidate_count"],
            0,
        )
        self.assertEqual(
            summary[
                "excluded_animal_type_count"
            ],
            1,
        )
        self.assertEqual(
            summary["maximum_score"],
            100,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)