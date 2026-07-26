"""
Reusable dashboard helpers for the Grazioso Salvare enhancement.

This module prepares the ranked scoring results for display in the
dashboard. It keeps data preparation, candidate selection, and export
logic outside the dashboard callbacks so each part can be tested
separately.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# These are the only columns shown in the ranked candidate table.
# Additional scoring fields remain in the table data so the selected
# candidate panel can explain why each animal received its ranking.
TABLE_COLUMN_DEFINITIONS = [
    {
        "name": "Rank",
        "id": "rescue_rank",
        "type": "numeric",
    },
    {
        "name": "Animal ID",
        "id": "animal_id",
        "type": "text",
    },
    {
        "name": "Rescue Mission",
        "id": "rescue_mission",
        "type": "text",
    },
    {
        "name": "Name",
        "id": "outcome_name",
        "type": "text",
    },
    {
        "name": "Breed",
        "id": "outcome_breed",
        "type": "text",
    },
    {
        "name": "Age",
        "id": "outcome_age_upon_outcome",
        "type": "text",
    },
    {
        "name": "Found Location",
        "id": "found_location",
        "type": "text",
    },
]


# These fields are visible in the main table.
TABLE_COLUMN_IDS = [
    column["id"]
    for column in TABLE_COLUMN_DEFINITIONS
]


# These additional fields are kept in each table record so the
# candidate information panel can display the scoring details.
CANDIDATE_DETAIL_COLUMN_IDS = [
    "total_score",
    "breed_score",
    "age_score",
    "sex_score",
    "outcome_sex_upon_outcome",
    "calculated_age_weeks",
    "score_explanation",
    "breed_reason",
    "age_reason",
    "sex_reason",
]


DASHBOARD_DATA_COLUMN_IDS = (
    TABLE_COLUMN_IDS
    + CANDIDATE_DETAIL_COLUMN_IDS
)


VALID_RESCUE_TYPES = {
    "all",
    "water",
    "mountain",
    "disaster",
}


def validate_ranked_results(
    ranked_results: dict[str, pd.DataFrame],
) -> None:
    """
    Make sure the dashboard received all three mission results.
    """

    if not isinstance(ranked_results, dict):
        raise TypeError(
            "Ranked results must be provided as a dictionary."
        )

    required_missions = {
        "water",
        "mountain",
        "disaster",
    }

    missing_missions = required_missions.difference(
        ranked_results
    )

    if missing_missions:
        missing_list = ", ".join(
            sorted(missing_missions)
        )

        raise ValueError(
            "Ranked results are missing rescue missions: "
            f"{missing_list}"
        )

    for rescue_type in required_missions:
        if not isinstance(
            ranked_results[rescue_type],
            pd.DataFrame,
        ):
            raise TypeError(
                f"Ranked results for {rescue_type} "
                "must be a pandas DataFrame."
            )


def build_all_candidates_view(
    ranked_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Show each animal once using its strongest rescue mission result.
    """

    validate_ranked_results(
        ranked_results
    )

    mission_frames = []

    # Each mission scores the same dogs differently. Combining the
    # results lets us keep the strongest mission for each animal.
    for rescue_type in (
        "water",
        "mountain",
        "disaster",
    ):
        mission_dataframe = (
            ranked_results[rescue_type]
            .copy()
        )

        if not mission_dataframe.empty:
            mission_frames.append(
                mission_dataframe
            )

    if not mission_frames:
        return pd.DataFrame(
            columns=DASHBOARD_DATA_COLUMN_IDS
        )

    combined_dataframe = pd.concat(
        mission_frames,
        ignore_index=True,
    )

    # Sort by the overall score first. The individual score parts
    # provide consistent tie-breaking when multiple missions have
    # the same total score.
    combined_dataframe = (
        combined_dataframe.sort_values(
            by=[
                "total_score",
                "breed_score",
                "age_score",
                "sex_score",
                "rescue_mission",
                "animal_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
    )

    animal_ids = (
        combined_dataframe["animal_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Keep only the strongest rescue mission for animals that have
    # a usable animal ID.
    identified_candidates = (
        combined_dataframe.loc[
            animal_ids.ne("")
        ]
        .drop_duplicates(
            subset=["animal_id"],
            keep="first",
        )
    )

    # Records without an animal ID cannot safely be treated as the
    # same animal, so they remain separate.
    unidentified_candidates = (
        combined_dataframe.loc[
            animal_ids.eq("")
        ]
    )

    all_candidates = pd.concat(
        [
            identified_candidates,
            unidentified_candidates,
        ],
        ignore_index=True,
    )

    all_candidates = (
        all_candidates.sort_values(
            by=[
                "total_score",
                "breed_score",
                "age_score",
                "sex_score",
                "rescue_mission",
                "animal_id",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # The All view receives its own overall rank so the rank still
    # matches the order shown in the dashboard.
    all_candidates["rescue_rank"] = (
        all_candidates.index + 1
    )

    return all_candidates


def prepare_dashboard_results(
    ranked_results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Prepare and cache every dashboard view before callbacks begin.
    """

    validate_ranked_results(
        ranked_results
    )

    dashboard_results = {
        rescue_type: dataframe.copy()
        for rescue_type, dataframe
        in ranked_results.items()
    }

    dashboard_results["all"] = (
        build_all_candidates_view(
            ranked_results
        )
    )

    return dashboard_results


def get_table_dataframe(
    dashboard_results: dict[str, pd.DataFrame],
    rescue_type: str | None,
) -> pd.DataFrame:
    """
    Return one clean candidate view for the selected rescue mission.
    """

    normalized_type = (
        rescue_type.strip().lower()
        if isinstance(rescue_type, str)
        else "all"
    )

    if normalized_type not in VALID_RESCUE_TYPES:
        normalized_type = "all"

    source_dataframe = dashboard_results.get(
        normalized_type,
        pd.DataFrame(),
    )

    if source_dataframe.empty:
        return pd.DataFrame(
            columns=DASHBOARD_DATA_COLUMN_IDS
        )

    available_columns = [
        column
        for column in DASHBOARD_DATA_COLUMN_IDS
        if column in source_dataframe.columns
    ]

    table_dataframe = (
        source_dataframe[
            available_columns
        ]
        .copy()
    )

    text_columns = [
        "animal_id",
        "rescue_mission",
        "outcome_name",
        "outcome_breed",
        "outcome_age_upon_outcome",
        "outcome_sex_upon_outcome",
        "score_explanation",
        "breed_reason",
        "age_reason",
        "sex_reason",
    ]

    # Replace missing text with a clear value instead of showing
    # pandas missing-value markers in the dashboard.
    for column in text_columns:
        if column in table_dataframe.columns:
            table_dataframe[column] = (
                table_dataframe[column]
                .fillna("Not available")
                .astype(str)
                .str.strip()
                .replace(
                    "",
                    "Not available",
                )
            )

    # Location uses a more direct message because it is also used
    # by the map panel.
    if "found_location" in table_dataframe.columns:
        table_dataframe["found_location"] = (
            table_dataframe["found_location"]
            .fillna("No location information")
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": "No location information",
                    "<NA>": "No location information",
                    "nan": "No location information",
                    "None": "No location information",
                    "Not available": (
                        "No location information"
                    ),
                }
            )
        )

    numeric_columns = [
        "rescue_rank",
        "total_score",
        "breed_score",
        "age_score",
        "sex_score",
        "calculated_age_weeks",
    ]

    for column in numeric_columns:
        if column in table_dataframe.columns:
            table_dataframe[column] = (
                pd.to_numeric(
                    table_dataframe[column],
                    errors="coerce",
                )
            )

    return table_dataframe


def get_selected_candidate(
    view_data: list[dict[str, Any]] | None,
    active_cell: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Return the candidate from the row the user clicked.

    The active cell identifies the row, but the dashboard will
    highlight the entire row instead of leaving one cell selected.
    """

    if not view_data:
        return None

    selected_index = 0

    if isinstance(active_cell, dict):
        possible_index = active_cell.get(
            "row"
        )

        if (
            isinstance(possible_index, int)
            and 0 <= possible_index < len(view_data)
        ):
            selected_index = possible_index

    candidate = view_data[
        selected_index
    ]

    if not isinstance(candidate, dict):
        return None

    return candidate


def build_selected_row_styles(
    active_cell: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Build table styles that highlight only the active row.
    """

    styles = [
        {
            "if": {
                "row_index": "odd",
            },
            "backgroundColor": "#f8fafc",
        },
        {
            "if": {
                "row_index": "even",
            },
            "backgroundColor": "white",
        },

        # Keep the clicked cell visually consistent with the row.
        {
            "if": {
                "state": "active",
            },
            "backgroundColor": "#dbeafe",
            "border": "1px solid #2563eb",
        },
    ]

    if not isinstance(active_cell, dict):
        return styles

    selected_row = active_cell.get(
        "row"
    )

    if not isinstance(selected_row, int):
        return styles

    styles.append(
        {
            "if": {
                "row_index": selected_row,
            },
            "backgroundColor": "#dbeafe",
            "borderTop": "1px solid #2563eb",
            "borderBottom": "1px solid #2563eb",
        }
    )

    return styles


def prepare_export_dataframe(
    view_data: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Prepare the current filtered and sorted table view for CSV export.
    """

    if not view_data:
        return pd.DataFrame(
            columns=TABLE_COLUMN_IDS
        )

    export_dataframe = pd.DataFrame(
        view_data
    )

    export_columns = [
        column
        for column in TABLE_COLUMN_IDS
        if column in export_dataframe.columns
    ]

    return (
        export_dataframe[
            export_columns
        ]
        .copy()
    )