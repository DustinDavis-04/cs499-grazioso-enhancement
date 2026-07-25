"""
Run the complete data preparation, matching, and rescue scoring process.

This script loads the intake and outcome datasets, matches each outcome
to its most recent eligible intake, and creates ranked candidate lists
for each Grazioso Salvare rescue mission.
"""

from data_preparation import load_prepared_data
from intake_outcome_matching import (
    match_outcomes_to_intakes,
    summarize_matches,
)
from rescue_scoring import (
    RESCUE_CRITERIA,
    score_and_rank_candidates,
)


intakes, outcomes = load_prepared_data()

matches = match_outcomes_to_intakes(
    intakes,
    outcomes,
)

summary = summarize_matches(matches)


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
    f"Unmatched reasons: "
    f"{summary['unmatched_reasons']}"
)


print()
print("FIRST 10 MATCHING RESULTS")
print("-------------------------")
print(
    matches[
        [
            "animal_id",
            "outcome_datetime",
            "intake_datetime",
            "found_location",
            "length_of_stay_days",
            "match_status",
            "unmatched_reason",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


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
    f"Future intake matches: "
    f"{future_match_count}"
)
print(
    f"Matched records without location: "
    f"{matched_without_location}"
)
print(
    f"Unmatched records with location: "
    f"{unmatched_with_location}"
)
print(
    f"Negative length of stay values: "
    f"{negative_stay_count}"
)


# Score the matched records for each rescue mission.
ranked_candidates = {}

for rescue_type in RESCUE_CRITERIA:
    ranked_candidates[rescue_type] = (
        score_and_rank_candidates(
            matches,
            rescue_type,
        )
    )


for rescue_type, candidates in ranked_candidates.items():
    scoring_summary = candidates.attrs[
        "rescue_scoring"
    ]

    print()
    print(
        scoring_summary["rescue_mission"].upper()
    )
    print(
        "-" * len(
            scoring_summary["rescue_mission"]
        )
    )

    print(
        f"Input records: "
        f"{scoring_summary['input_record_count']}"
    )
    print(
        f"Eligible dog records: "
        f"{scoring_summary['eligible_record_count']}"
    )
    print(
        f"Repeated historical records removed: "
        f"{scoring_summary['duplicate_record_count']}"
    )
    print(
        f"Unique dog candidates: "
        f"{scoring_summary['candidate_count']}"
    )
    print(
        f"Other animal types excluded: "
        f"{scoring_summary['excluded_animal_type_count']}"
    )
    print(
        f"Maximum score: "
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

    print(
        candidates[result_columns]
        .head(10)
        .to_string(index=False)
    )