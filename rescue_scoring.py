"""
Weighted rescue suitability scoring for the Grazioso Salvare dashboard.

This module scores and ranks animal candidates for each rescue mission.
It keeps the rescue requirements in one configuration instead of placing
the rules throughout the dashboard callbacks.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# Keep the rescue mission rules together so they can be updated without
# changing the scoring functions.
RESCUE_CRITERIA: dict[str, dict[str, Any]] = {
    "water": {
        "display_name": "Water Rescue",
        "animal_type": "Dog",
        "preferred_breeds": (
            "Labrador Retriever",
            "Chesapeake Bay Retriever",
            "Newfoundland",
        ),
        "age_range_weeks": {
            "minimum": 26,
            "maximum": 156,
        },
        "preferred_sex": "Intact Male",
        "weights": {
            "breed": {
                "exact": 50,
                "mixed": 40,
                "none": 0,
                "missing": 0,
            },
            "age": {
                "within_range": 30,
                "outside_range": 0,
                "missing": 0,
            },
            "sex": {
                "preferred": 20,
                "not_preferred": 0,
                "missing": 0,
            },
        },
    },
    "mountain": {
        "display_name": "Mountain or Wilderness Rescue",
        "animal_type": "Dog",
        "preferred_breeds": (
            "German Shepherd",
            "Alaskan Malamute",
            "Old English Sheepdog",
            "Siberian Husky",
            "Rottweiler",
        ),
        "age_range_weeks": {
            "minimum": 26,
            "maximum": 156,
        },
        "preferred_sex": "Intact Male",
        "weights": {
            "breed": {
                "exact": 50,
                "mixed": 40,
                "none": 0,
                "missing": 0,
            },
            "age": {
                "within_range": 30,
                "outside_range": 0,
                "missing": 0,
            },
            "sex": {
                "preferred": 20,
                "not_preferred": 0,
                "missing": 0,
            },
        },
    },
    "disaster": {
        "display_name": "Disaster or Individual Tracking",
        "animal_type": "Dog",
        "preferred_breeds": (
            "Doberman Pinscher",
            "German Shepherd",
            "Golden Retriever",
            "Bloodhound",
            "Rottweiler",
        ),
        "age_range_weeks": {
            "minimum": 26,
            "maximum": 156,
        },
        "preferred_sex": "Intact Male",
        "weights": {
            "breed": {
                "exact": 50,
                "mixed": 40,
                "none": 0,
                "missing": 0,
            },
            "age": {
                "within_range": 30,
                "outside_range": 0,
                "missing": 0,
            },
            "sex": {
                "preferred": 20,
                "not_preferred": 0,
                "missing": 0,
            },
        },
    },
}


# These are the default columns created by the intake and outcome
# matching process. A different map can be provided when needed.
DEFAULT_COLUMN_MAP = {
    "animal_id": "animal_id",
    "animal_type": "outcome_animal_type",
    "breed": "outcome_breed",
    "age_weeks": "outcome_age_upon_outcome_in_weeks",
    "age_text": "outcome_age_upon_outcome",
    "sex": "outcome_sex_upon_outcome",
    "outcome_datetime": "outcome_datetime",
}


SCORE_COLUMNS = (
    "rescue_type",
    "rescue_mission",
    "breed_score",
    "breed_match_type",
    "breed_reason",
    "age_score",
    "age_match_type",
    "age_reason",
    "calculated_age_weeks",
    "sex_score",
    "sex_match_type",
    "sex_reason",
    "total_score",
    "score_explanation",
    "rescue_rank",
)


def validate_rescue_type(
    rescue_type: str,
) -> dict[str, Any]:
    """
    Verify that the requested rescue mission exists.
    """

    if not isinstance(rescue_type, str):
        raise TypeError(
            "Rescue type must be a string."
        )

    normalized_type = rescue_type.strip().lower()

    if normalized_type not in RESCUE_CRITERIA:
        available_types = ", ".join(
            sorted(RESCUE_CRITERIA)
        )

        raise ValueError(
            f"Unknown rescue type: {rescue_type}. "
            f"Available rescue types: {available_types}"
        )

    return RESCUE_CRITERIA[normalized_type]


def validate_scoring_dataframe(
    dataframe: pd.DataFrame,
    column_map: dict[str, str],
) -> None:
    """
    Verify that the scoring data contains the required columns.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Scoring data must be a pandas DataFrame."
        )

    required_columns = {
        column_map["animal_id"],
        column_map["animal_type"],
        column_map["breed"],
        column_map["sex"],
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    # Age may already be stored as weeks, or it may still be stored
    # as text such as 2 years or 6 months.
    has_numeric_age = (
        column_map["age_weeks"] in dataframe.columns
    )

    has_text_age = (
        column_map["age_text"] in dataframe.columns
    )

    if not has_numeric_age and not has_text_age:
        missing_columns.add(
            f"{column_map['age_weeks']} or "
            f"{column_map['age_text']}"
        )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Scoring data is missing required columns: "
            f"{missing_list}"
        )


def normalize_text(
    value: Any,
) -> str | None:
    """
    Clean a text value before using it for scoring.
    """

    if pd.isna(value):
        return None

    normalized_value = str(value).strip()

    if not normalized_value:
        return None

    # Replace extra spaces so differently formatted values can still
    # be compared consistently.
    return re.sub(
        r"\s+",
        " ",
        normalized_value,
    )


def parse_age_to_weeks(
    value: Any,
) -> float | None:
    """
    Convert an age value into an approximate number of weeks.
    """

    if pd.isna(value):
        return None

    # Numeric age values are already expected to be measured in weeks.
    if isinstance(value, (int, float)):
        return float(value)

    normalized_value = normalize_text(value)

    if normalized_value is None:
        return None

    lowered_value = normalized_value.lower()

    age_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*"
        r"(year|years|month|months|week|weeks|day|days)",
        lowered_value,
    )

    if age_match is None:
        return None

    age_amount = float(age_match.group(1))
    age_unit = age_match.group(2)

    # These conversions keep all ages in one format so they can be
    # compared against the rescue mission age range.
    weeks_per_unit = {
        "year": 52,
        "years": 52,
        "month": 4.345,
        "months": 4.345,
        "week": 1,
        "weeks": 1,
        "day": 1 / 7,
        "days": 1 / 7,
    }

    return age_amount * weeks_per_unit[age_unit]


def calculate_maximum_score(
    mission_config: dict[str, Any],
) -> int:
    """
    Calculate the highest possible score from the mission weights.
    """

    weights = mission_config["weights"]

    return int(
        weights["breed"]["exact"]
        + weights["age"]["within_range"]
        + weights["sex"]["preferred"]
    )


def score_breed(
    breed: Any,
    mission_config: dict[str, Any],
) -> tuple[int, str, str]:
    """
    Score an animal breed for one rescue mission.
    """

    normalized_breed = normalize_text(breed)
    breed_weights = mission_config["weights"]["breed"]

    if normalized_breed is None:
        return (
            breed_weights["missing"],
            "missing",
            "Breed information is missing.",
        )

    lowered_breed = normalized_breed.casefold()

    # These values are too general to compare against a preferred breed.
    unknown_breed_values = {
        "unknown",
        "unknown mix",
        "other",
        "mixed breed",
        "mix",
    }

    if lowered_breed in unknown_breed_values:
        return (
            breed_weights["missing"],
            "missing",
            "Breed information is unknown or too general "
            "to evaluate.",
        )

    preferred_breeds = mission_config[
        "preferred_breeds"
    ]

    # Check for an exact match before checking for a mixed breed match.
    for preferred_breed in preferred_breeds:
        if (
            lowered_breed
            == preferred_breed.casefold()
        ):
            return (
                breed_weights["exact"],
                "exact",
                f"{normalized_breed} is an exact preferred "
                "breed match.",
            )

    # A preferred breed that appears as part of a mixed breed still
    # receives most of the available breed points.
    for preferred_breed in preferred_breeds:
        if (
            preferred_breed.casefold()
            in lowered_breed
        ):
            return (
                breed_weights["mixed"],
                "mixed",
                f"{normalized_breed} contains the preferred "
                f"breed {preferred_breed}.",
            )

    return (
        breed_weights["none"],
        "none",
        f"{normalized_breed} is not a preferred breed "
        "for this mission.",
    )


def score_age(
    age_weeks: Any,
    age_text: Any,
    mission_config: dict[str, Any],
) -> tuple[int, str, str, float | None]:
    """
    Score an animal age for one rescue mission.
    """

    # Use the numeric age when it is available. Otherwise, try to
    # convert the original text age into weeks.
    calculated_age_weeks = parse_age_to_weeks(
        age_weeks
    )

    if calculated_age_weeks is None:
        calculated_age_weeks = parse_age_to_weeks(
            age_text
        )

    age_weights = mission_config["weights"]["age"]
    age_range = mission_config["age_range_weeks"]

    if calculated_age_weeks is None:
        return (
            age_weights["missing"],
            "missing",
            "Age information is missing or could not "
            "be interpreted.",
            None,
        )

    minimum_age = age_range["minimum"]
    maximum_age = age_range["maximum"]

    if (
        minimum_age
        <= calculated_age_weeks
        <= maximum_age
    ):
        return (
            age_weights["within_range"],
            "within_range",
            "Age is within the preferred range of "
            f"{minimum_age} to {maximum_age} weeks.",
            calculated_age_weeks,
        )

    return (
        age_weights["outside_range"],
        "outside_range",
        "Age is outside the preferred range of "
        f"{minimum_age} to {maximum_age} weeks.",
        calculated_age_weeks,
    )


def score_sex(
    sex: Any,
    mission_config: dict[str, Any],
) -> tuple[int, str, str]:
    """
    Score an animal sex and intact status for one rescue mission.
    """

    normalized_sex = normalize_text(sex)
    sex_weights = mission_config["weights"]["sex"]
    preferred_sex = mission_config["preferred_sex"]

    if normalized_sex is None:
        return (
            sex_weights["missing"],
            "missing",
            "Sex and intact-status information is missing.",
        )

    if (
        normalized_sex.casefold()
        == preferred_sex.casefold()
    ):
        return (
            sex_weights["preferred"],
            "preferred",
            f"{normalized_sex} matches the preferred "
            "sex and intact status.",
        )

    return (
        sex_weights["not_preferred"],
        "not_preferred",
        f"{normalized_sex} does not match the preferred "
        f"value of {preferred_sex}.",
    )


def build_score_explanation(
    mission_name: str,
    breed_score: int,
    breed_reason: str,
    age_score: int,
    age_reason: str,
    sex_score: int,
    sex_reason: str,
    total_score: int,
) -> str:
    """
    Combine the scoring results into one readable explanation.
    """

    return (
        f"{mission_name}: "
        f"Breed {breed_score} points. "
        f"{breed_reason} "
        f"Age {age_score} points. "
        f"{age_reason} "
        f"Sex {sex_score} points. "
        f"{sex_reason} "
        f"Total suitability score: {total_score}."
    )


def score_candidate(
    row: pd.Series,
    rescue_type: str,
    column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Score one animal candidate for one rescue mission.
    """

    mission_config = validate_rescue_type(
        rescue_type
    )

    normalized_rescue_type = (
        rescue_type.strip().lower()
    )

    resolved_columns = (
        DEFAULT_COLUMN_MAP | (column_map or {})
    )

    breed_value = row.get(
        resolved_columns["breed"]
    )

    sex_value = row.get(
        resolved_columns["sex"]
    )

    age_weeks_value = row.get(
        resolved_columns["age_weeks"]
    )

    age_text_value = row.get(
        resolved_columns["age_text"]
    )

    (
        breed_score,
        breed_match_type,
        breed_reason,
    ) = score_breed(
        breed_value,
        mission_config,
    )

    (
        age_score,
        age_match_type,
        age_reason,
        calculated_age_weeks,
    ) = score_age(
        age_weeks_value,
        age_text_value,
        mission_config,
    )

    (
        sex_score,
        sex_match_type,
        sex_reason,
    ) = score_sex(
        sex_value,
        mission_config,
    )

    total_score = (
        breed_score
        + age_score
        + sex_score
    )

    mission_name = mission_config[
        "display_name"
    ]

    score_explanation = build_score_explanation(
        mission_name=mission_name,
        breed_score=breed_score,
        breed_reason=breed_reason,
        age_score=age_score,
        age_reason=age_reason,
        sex_score=sex_score,
        sex_reason=sex_reason,
        total_score=total_score,
    )

    # Return each part of the score separately so the dashboard can
    # display, filter, or sort the results without rescoring.
    return {
        "rescue_type": normalized_rescue_type,
        "rescue_mission": mission_name,
        "breed_score": breed_score,
        "breed_match_type": breed_match_type,
        "breed_reason": breed_reason,
        "age_score": age_score,
        "age_match_type": age_match_type,
        "age_reason": age_reason,
        "calculated_age_weeks": calculated_age_weeks,
        "sex_score": sex_score,
        "sex_match_type": sex_match_type,
        "sex_reason": sex_reason,
        "total_score": total_score,
        "score_explanation": score_explanation,
    }


def add_empty_score_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the expected scoring columns to an empty result.
    """

    empty_result = dataframe.copy()

    # Keep the same structure even when no eligible records are found.
    for column in SCORE_COLUMNS:
        empty_result[column] = pd.Series(
            dtype="object"
        )

    return empty_result


def keep_latest_candidate_record(
    dataframe: pd.DataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    """
    Keep the most recent outcome record for each animal.
    """

    candidate_dataframe = dataframe.copy()

    animal_id_column = column_map["animal_id"]
    outcome_datetime_column = column_map[
        "outcome_datetime"
    ]

    # Keep the original order available as a final stable tie breaker.
    candidate_dataframe["_candidate_order"] = range(
        len(candidate_dataframe)
    )

    if (
        outcome_datetime_column
        in candidate_dataframe.columns
    ):
        candidate_dataframe[
            "_candidate_outcome_datetime"
        ] = pd.to_datetime(
            candidate_dataframe[
                outcome_datetime_column
            ],
            errors="coerce",
        )

        candidate_dataframe = (
            candidate_dataframe.sort_values(
                by=[
                    animal_id_column,
                    "_candidate_outcome_datetime",
                    "_candidate_order",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
                kind="mergesort",
                na_position="last",
            )
        )
    else:
        candidate_dataframe = (
            candidate_dataframe.sort_values(
                by=[
                    animal_id_column,
                    "_candidate_order",
                ],
                ascending=[
                    True,
                    False,
                ],
                kind="mergesort",
                na_position="last",
            )
        )

    # Records with a known animal ID can be reduced to one current
    # candidate. Records without an ID are kept as separate records.
    animal_ids = (
        candidate_dataframe[animal_id_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    identified_records = candidate_dataframe.loc[
        animal_ids.ne("")
    ].copy()

    unidentified_records = candidate_dataframe.loc[
        animal_ids.eq("")
    ].copy()

    identified_records = (
        identified_records.drop_duplicates(
            subset=[animal_id_column],
            keep="first",
        )
    )

    candidate_dataframe = pd.concat(
        [
            identified_records,
            unidentified_records,
        ],
        ignore_index=False,
    )

    temporary_columns = [
        "_candidate_order",
        "_candidate_outcome_datetime",
    ]

    candidate_dataframe = (
        candidate_dataframe.drop(
            columns=[
                column
                for column in temporary_columns
                if column in candidate_dataframe.columns
            ]
        )
        .copy()
    )

    return candidate_dataframe


def score_and_rank_candidates(
    dataframe: pd.DataFrame,
    rescue_type: str,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Score every eligible candidate and return the results in ranked order.
    """

    resolved_columns = (
        DEFAULT_COLUMN_MAP | (column_map or {})
    )

    mission_config = validate_rescue_type(
        rescue_type
    )

    normalized_rescue_type = (
        rescue_type.strip().lower()
    )

    validate_scoring_dataframe(
        dataframe,
        resolved_columns,
    )

    scored_dataframe = dataframe.copy()

    animal_type_column = resolved_columns[
        "animal_type"
    ]

    preferred_animal_type = mission_config[
        "animal_type"
    ]

    # Keep the original requirement that rescue candidates must be dogs.
    # Breed, age, and sex are scored instead of used as strict filters.
    animal_type_values = (
        scored_dataframe[animal_type_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    eligible_mask = (
        animal_type_values
        == preferred_animal_type.casefold()
    )

    excluded_animal_type_count = int(
        (~eligible_mask).sum()
    )

    scored_dataframe = (
        scored_dataframe.loc[eligible_mask]
        .copy()
    )

    eligible_record_count = int(
        len(scored_dataframe)
    )

    # One animal may have several historical outcome records. Keep only
    # the most recent record so each animal receives one final ranking.
    scored_dataframe = keep_latest_candidate_record(
        scored_dataframe,
        resolved_columns,
    )

    duplicate_record_count = int(
        eligible_record_count
        - len(scored_dataframe)
    )

    maximum_score = calculate_maximum_score(
        mission_config
    )

    scoring_summary = {
        "rescue_type": normalized_rescue_type,
        "rescue_mission": mission_config[
            "display_name"
        ],
        "input_record_count": int(
            len(dataframe)
        ),
        "eligible_record_count": (
            eligible_record_count
        ),
        "candidate_count": int(
            len(scored_dataframe)
        ),
        "duplicate_record_count": (
            duplicate_record_count
        ),
        "excluded_animal_type_count": (
            excluded_animal_type_count
        ),
        "maximum_score": maximum_score,
        "ranking_order": [
            "total_score descending",
            "breed_score descending",
            "age_score descending",
            "sex_score descending",
            "animal_id ascending",
        ],
    }

    if scored_dataframe.empty:
        scored_dataframe = add_empty_score_columns(
            scored_dataframe
        )

        scored_dataframe.attrs[
            "rescue_scoring"
        ] = scoring_summary

        return scored_dataframe

    # Score each eligible animal without changing the original
    # matching data.
    score_records = scored_dataframe.apply(
        lambda row: score_candidate(
            row,
            normalized_rescue_type,
            resolved_columns,
        ),
        axis=1,
        result_type="expand",
    )

    # Add the structured scoring fields to the copied DataFrame.
    for score_column in score_records.columns:
        scored_dataframe[score_column] = (
            score_records[score_column]
        )

    animal_id_column = resolved_columns[
        "animal_id"
    ]

    # Use a cleaned animal ID for the final tie breaker.
    scored_dataframe["_ranking_animal_id"] = (
        scored_dataframe[animal_id_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    scored_dataframe = scored_dataframe.sort_values(
        by=[
            "total_score",
            "breed_score",
            "age_score",
            "sex_score",
            "_ranking_animal_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Assign the final rank after all score and tie-breaker sorting
    # has been completed.
    scored_dataframe["rescue_rank"] = (
        scored_dataframe.index + 1
    )

    scored_dataframe = scored_dataframe.drop(
        columns=["_ranking_animal_id"]
    )

    scored_dataframe.attrs[
        "rescue_scoring"
    ] = scoring_summary

    return scored_dataframe