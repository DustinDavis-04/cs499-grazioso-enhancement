from data_preparation import load_prepared_data
from intake_outcome_matching import (
    match_outcomes_to_intakes,
    summarize_matches,
)


intakes, outcomes = load_prepared_data()

matches = match_outcomes_to_intakes(intakes, outcomes)
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
print(f"Unmatched reasons: {summary['unmatched_reasons']}")

print()
print("FIRST 10 RESULTS")
print("----------------")
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
print("VALIDATION CHECKS")
print("-----------------")

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
    & (matches["length_of_stay_days"] < 0)
).sum()

print(f"Future intake matches: {future_match_count}")
print(f"Matched records without location: {matched_without_location}")
print(f"Unmatched records with location: {unmatched_with_location}")
print(f"Negative length of stay values: {negative_stay_count}")