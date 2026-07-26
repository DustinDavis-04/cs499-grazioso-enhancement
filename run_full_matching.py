"""
Run the complete data preparation, matching, and rescue scoring process.

This script loads the intake and outcome datasets, matches each outcome
to its most recent eligible intake, and creates ranked candidate lists
for each Grazioso Salvare rescue mission.
"""

from __future__ import annotations

import pandas as pd

from data_preparation import load_prepared_data
from intake_outcome_matching import (
    match_outcomes_to_intakes,
    summarize_matches,
)
from rescue_scoring import (
    RESCUE_CRITERIA,
    score_and_rank_candidates,
)


def build_matched_data() -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """
    Load the prepared data and create the intake-to-outcome matches.
    """

    # Load both cleaned datasets before running the matching process.
    intakes, outcomes = load_prepared_data()

    # Match each outcome to the most recent intake that happened before it.
    matches = match_outcomes_to_intakes(
        intakes,
        outcomes,
    )

    # Keep the matching summary available for reports and testing.
    summary = summarize_matches(matches)

    return matches, summary


def build_ranked_candidate_data(
    matches: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """
    Create the matched dataset and ranked results for every rescue mission.
    """

    # Let callers reuse an existing matched DataFrame so the matching
    # process does not have to run again when it is already available.
    if matches is None:
        matches, _ = build_matched_data()

    ranked_candidates: dict[str, pd.DataFrame] = {}

    # Score the same matched records for each rescue mission.
    for rescue_type in RESCUE_CRITERIA:
        ranked_candidates[rescue_type] = (
            score_and_rank_candidates(
                matches,
                rescue_type,
            )
        )

    return matches, ranked_candidates


def print_matching_summary(
    summary: dict[str, object],
) -> None:
    """
    Display the matching summary in the console.
    """

    print("MATCHING SUMMARY")
    print("----------------")
    print(f"Total intakes: {summary['total_intakes']}")
    print(f"Valid intakes: {summary['valid_intakes']}")
    print(f"Excluded intakes: {summary['excluded_intakes']}")
    print(f"Total outcomes: {summary['total_outcomes']}")
    print(f"Valid outcomes: {summary['valid_outcomes']}")
    print(f"Invalid outcomes: {summary['invalid_outcomes']}")
    print(f"Matched outcomes: {summary['matched_outcomes']}")
    print(f"Unmatched outcomes: {summary['unmatched_outcomes']}")
    print(
        "Unmatched reasons: "
        f"{summary['unmatched_reasons']}"
    )


def print_matching_preview(
    matches: pd.DataFrame,
) -> None:
    """
    Display the first matching results in the console.
    """

    print()
    print("FIRST 10 MATCHING RESULTS")
    print("-------------------------")

    result_columns = [
        "animal_id",
        "outcome_datetime",
        "intake_datetime",
        "found_location",
        "length_of_stay_days",
        "match_status",
        "unmatched_reason",
    ]

    available_columns = [
        column
        for column in result_columns
        if column in matches.columns
    ]

    print(
        matches[available_columns]
        .head(10)
        .to_string(index=False)
    )


def print_matching_validation(
    matches: pd.DataFrame,
) -> None:
    """
    Display checks that confirm the matching results are valid.
    """

    print()
    print("MATCHING VALIDATION CHECKS")
    print("--------------------------")

    future_match_count = (
        matches["intake_datetime"].notna()
        & (
            matches["intake_datetime"]
            > matches["outcome_datetime"]
        )
    ).sum()

    matched_without_location = (
        matches["match_status"].eq("matched")
        & matches["found_location"].isna()
    ).sum()

    unmatched_with_location = (
        matches["match_status"].eq("unmatched")
        & matches["found_location"].notna()
    ).sum()

    negative_stay_count = (
        matches["length_of_stay_days"].notna()
        & (
            matches["length_of_stay_days"] < 0
        )
    ).sum()

    print(
        "Future intake matches: "
        f"{future_match_count}"
    )
    print(
        "Matched records without location: "
        f"{matched_without_location}"
    )
    print(
        "Unmatched records with location: "
        f"{unmatched_with_location}"
    )
    print(
        "Negative length of stay values: "
        f"{negative_stay_count}"
    )


def print_scoring_results(
    ranked_candidates: dict[str, pd.DataFrame],
) -> None:
    """
    Display the scoring summary and top candidates for each mission.
    """

    for rescue_type, candidates in ranked_candidates.items():
        scoring_summary = candidates.attrs[
            "rescue_scoring"
        ]

        mission_name = scoring_summary[
            "rescue_mission"
        ]

        print()
        print(mission_name.upper())
        print("-" * len(mission_name))

        print(
            "Input records: "
            f"{scoring_summary['input_record_count']}"
        )
        print(
            "Eligible dog records: "
            f"{scoring_summary['eligible_record_count']}"
        )
        print(
            "Repeated historical records removed: "
            f"{scoring_summary['duplicate_record_count']}"
        )
        print(
            "Unique dog candidates: "
            f"{scoring_summary['candidate_count']}"
        )
        print(
            "Other animal types excluded: "
            f"{scoring_summary['excluded_animal_type_count']}"
        )
        print(
            "Maximum score: "
            f"{scoring_summary['maximum_score']}"
        )

        print()
        print("TOP 10 RANKED CANDIDATES")
        print("------------------------")

        result_columns = [
            "rescue_rank",
            "animal_id",
            "outcome_breed",
            "calculated_age_weeks",
            "outcome_sex_upon_outcome",
            "breed_score",
            "age_score",
            "sex_score",
            "total_score",
        ]

        available_columns = [
            column
            for column in result_columns
            if column in candidates.columns
        ]

        print(
            candidates[available_columns]
            .head(10)
            .to_string(index=False)
        )


def main() -> None:
    """
    Run the full pipeline and display the validation report.
    """

    matches, summary = build_matched_data()

    _, ranked_candidates = (
        build_ranked_candidate_data(matches)
    )

    print_matching_summary(summary)
    print_matching_preview(matches)
    print_matching_validation(matches)
    print_scoring_results(ranked_candidates)


if __name__ == "__main__":
    main()