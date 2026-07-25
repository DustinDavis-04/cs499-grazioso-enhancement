"""Match animal outcomes to their most recent eligible intake records.

This module links each outcome record to the latest intake record for the
same animal that occurred at or before the outcome datetime.

The matching process uses pandas.merge_asof instead of nested loops. This
provides an efficient grouped temporal match while ensuring that an outcome
can never be matched to an intake that occurred in the future.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_INTAKE_COLUMNS = {
    "animal_id",
    "datetime",
    "found_location",
}

REQUIRED_OUTCOME_COLUMNS = {
    "animal_id",
    "datetime",
}


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Confirm that a DataFrame contains all required columns.

    Args:
        dataframe: DataFrame being validated.
        required_columns: Columns required by the matching process.
        dataframe_name: Descriptive name used in the error message.

    Raises:
        TypeError: If dataframe is not a pandas DataFrame.
        ValueError: If one or more required columns are missing.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{dataframe_name} must be a pandas DataFrame.")

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_list}"
        )


def _prepare_intakes(
    intakes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prepare valid intake records for temporal matching.

    Intake columns other than animal_id are renamed with an intake_ prefix.
    The original row position is preserved so duplicate intake timestamps
    can be handled consistently.

    When the same animal has multiple intake records with an identical
    datetime, the record appearing last in the original dataset becomes
    the final eligible record at that timestamp.

    Args:
        intakes: Prepared intake DataFrame.

    Returns:
        A tuple containing the valid intake records and preparation
        statistics.
    """

    _validate_required_columns(
        intakes,
        REQUIRED_INTAKE_COLUMNS,
        "Intake data",
    )

    prepared = intakes.copy()

    prepared["intake_source_row"] = range(len(prepared))

    missing_animal_id = prepared["animal_id"].isna()
    invalid_datetime = prepared["datetime"].isna()

    valid_mask = ~missing_animal_id & ~invalid_datetime
    valid = prepared.loc[valid_mask].copy()

    rename_map = {
        column: f"intake_{column}"
        for column in valid.columns
        if column not in {
            "animal_id",
            "intake_source_row",
        }
    }

    valid = valid.rename(columns=rename_map)

    # merge_asof requires the temporal matching column to be sorted.
    # A stable sort and the source row provide deterministic behavior
    # when duplicate intake timestamps exist.
    valid = valid.sort_values(
        [
            "intake_datetime",
            "animal_id",
            "intake_source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    statistics = {
        "total_intakes": int(len(prepared)),
        "valid_intakes": int(len(valid)),
        "excluded_intakes": int((~valid_mask).sum()),
        "missing_intake_animal_id": int(missing_animal_id.sum()),
        "invalid_intake_datetime": int(invalid_datetime.sum()),
    }

    return valid, statistics


def _prepare_outcomes(
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Prepare outcome records for temporal matching.

    Valid outcomes are separated from outcomes that have a missing Animal ID
    or an invalid datetime. Invalid outcomes are retained so every original
    outcome still appears in the final results.

    Args:
        outcomes: Prepared outcome DataFrame.

    Returns:
        A tuple containing valid outcomes, invalid outcomes, and preparation
        statistics.
    """

    _validate_required_columns(
        outcomes,
        REQUIRED_OUTCOME_COLUMNS,
        "Outcome data",
    )

    prepared = outcomes.copy()

    prepared["outcome_source_row"] = range(len(prepared))

    missing_animal_id = prepared["animal_id"].isna()
    invalid_datetime = prepared["datetime"].isna()

    valid_mask = ~missing_animal_id & ~invalid_datetime

    valid = prepared.loc[valid_mask].copy()
    invalid = prepared.loc[~valid_mask].copy()

    rename_map = {
        column: f"outcome_{column}"
        for column in prepared.columns
        if column not in {
            "animal_id",
            "outcome_source_row",
        }
    }

    valid = valid.rename(columns=rename_map)
    invalid = invalid.rename(columns=rename_map)

    valid = valid.sort_values(
        [
            "outcome_datetime",
            "animal_id",
            "outcome_source_row",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    invalid["match_status"] = "unmatched"
    invalid["unmatched_reason"] = pd.NA

    invalid.loc[
        invalid["animal_id"].isna(),
        "unmatched_reason",
    ] = "missing_animal_id"

    invalid.loc[
        invalid["animal_id"].notna()
        & invalid["outcome_datetime"].isna(),
        "unmatched_reason",
    ] = "invalid_outcome_datetime"

    statistics = {
        "total_outcomes": int(len(prepared)),
        "valid_outcomes": int(len(valid)),
        "invalid_outcomes": int((~valid_mask).sum()),
        "missing_outcome_animal_id": int(missing_animal_id.sum()),
        "invalid_outcome_datetime": int(invalid_datetime.sum()),
    }

    return valid, invalid, statistics


def match_outcomes_to_intakes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Match each outcome to its latest eligible intake.

    An intake is eligible only when:

    1. The intake and outcome have the same Animal ID.
    2. The intake datetime is less than or equal to the outcome datetime.

    The function performs a backward grouped temporal join. This ensures
    that the most recent valid intake is selected and that future intakes
    are never matched to earlier outcomes.

    Every outcome is retained. Outcomes without an eligible intake are
    marked as unmatched, and no found location is invented for them.

    Args:
        intakes: Prepared intake records.
        outcomes: Prepared outcome records.

    Returns:
        A DataFrame containing every outcome and its corresponding intake
        information when a valid match is found.
    """

    valid_intakes, intake_statistics = _prepare_intakes(intakes)

    (
        valid_outcomes,
        invalid_outcomes,
        outcome_statistics,
    ) = _prepare_outcomes(outcomes)

    if valid_outcomes.empty:
        matched = valid_outcomes.copy()
    elif valid_intakes.empty:
        matched = valid_outcomes.copy()

        for column in valid_intakes.columns:
            if column != "animal_id" and column not in matched.columns:
                if column == "intake_datetime":
                    matched[column] = pd.NaT
                else:
                    matched[column] = pd.NA

    else:
        matched = pd.merge_asof(
            valid_outcomes,
            valid_intakes,
            left_on="outcome_datetime",
            right_on="intake_datetime",
            by="animal_id",
            direction="backward",
            allow_exact_matches=True,
        )

    if "intake_datetime" not in matched.columns:
        matched["intake_datetime"] = pd.NaT

    matched["outcome_datetime"] = pd.to_datetime(
        matched["outcome_datetime"],
        errors="coerce",
    )

    matched["intake_datetime"] = pd.to_datetime(
        matched["intake_datetime"],
        errors="coerce",
    )

    has_matching_intake = matched["intake_datetime"].notna()

    matched["match_status"] = "unmatched"

    matched.loc[
        has_matching_intake,
        "match_status",
    ] = "matched"

    matched["unmatched_reason"] = pd.NA

    matched.loc[
        ~has_matching_intake,
        "unmatched_reason",
    ] = "no_earlier_intake"

    matched["length_of_stay_days"] = (
        matched["outcome_datetime"] - matched["intake_datetime"]
    ).dt.total_seconds() / 86_400

    # Present the matched location with a clear final column name.
    # Unmatched records remain empty because no location is invented.
    if "intake_found_location" in matched.columns:
        matched["found_location"] = matched[
            "intake_found_location"
        ]

        matched = matched.drop(
            columns=["intake_found_location"]
        )
    else:
        matched["found_location"] = pd.NA

    if not invalid_outcomes.empty:
        for column in matched.columns:
            if column not in invalid_outcomes.columns:
                invalid_outcomes[column] = pd.NA

        invalid_outcomes["found_location"] = pd.NA
        invalid_outcomes["length_of_stay_days"] = pd.NA

        all_results = pd.concat(
            [
                matched,
                invalid_outcomes,
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        all_results = matched.copy()

    # Return results in the same order as the original outcome dataset.
    all_results = all_results.sort_values(
        "outcome_source_row",
        kind="mergesort",
    ).reset_index(drop=True)

    matched_count = int(
        (all_results["match_status"] == "matched").sum()
    )

    unmatched_count = int(
        (all_results["match_status"] == "unmatched").sum()
    )

    all_results.attrs["matching_statistics"] = {
        **intake_statistics,
        **outcome_statistics,
        "matched_outcomes": matched_count,
        "unmatched_outcomes": unmatched_count,
    }

    return all_results


def summarize_matches(
    matches: pd.DataFrame,
) -> dict[str, Any]:
    """Create a summary of the matching results.

    Args:
        matches: DataFrame returned by match_outcomes_to_intakes.

    Returns:
        A dictionary containing matched and unmatched totals, unmatched
        reasons, and any stored preparation statistics.

    Raises:
        TypeError: If matches is not a pandas DataFrame.
        ValueError: If required result columns are missing.
    """

    if not isinstance(matches, pd.DataFrame):
        raise TypeError("Matches must be a pandas DataFrame.")

    required_result_columns = {
        "match_status",
        "unmatched_reason",
    }

    missing_columns = required_result_columns.difference(
        matches.columns
    )

    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Matching results are missing required columns: "
            f"{missing_list}"
        )

    matched_count = int(
        (matches["match_status"] == "matched").sum()
    )

    unmatched_count = int(
        (matches["match_status"] == "unmatched").sum()
    )

    unmatched_reasons = (
        matches.loc[
            matches["match_status"] == "unmatched",
            "unmatched_reason",
        ]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary: dict[str, Any] = {
        "total_outcomes": int(len(matches)),
        "matched_outcomes": matched_count,
        "unmatched_outcomes": unmatched_count,
        "unmatched_reasons": unmatched_reasons,
    }

    stored_statistics = matches.attrs.get(
        "matching_statistics"
    )

    if stored_statistics:
        summary.update(stored_statistics)

    return summary