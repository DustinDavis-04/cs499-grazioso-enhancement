"""
Reusable data preparation pipeline for the Grazioso Salvare dashboard.

This module loads, validates, cleans, and standardizes the Austin Animal
Center intake and outcome datasets. It does not perform intake-outcome
matching, rescue suitability scoring, ranking, or dashboard operations.
"""


from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"

DEFAULT_INTAKE_PATH = (
    DEFAULT_DATA_DIRECTORY / "Austin Animal Center Intakes.csv"
)

DEFAULT_OUTCOME_PATH = (
    DEFAULT_DATA_DIRECTORY / "Austin Animal Center Outcomes.csv"
)


# These are the columns the intake file must contain before we can
# prepare it for the rest of the project.
INTAKE_REQUIRED_COLUMNS = {
    "Animal ID",
    "Name",
    "DateTime",
    "Found Location",
    "Animal Type",
    "Sex upon Intake",
    "Age upon Intake",
    "Breed",
}


# These are the columns the outcome file must contain before we can
# prepare it for later matching and analysis.
OUTCOME_REQUIRED_COLUMNS = {
    "Animal ID",
    "Name",
    "DateTime",
    "Date of Birth",
    "Animal Type",
    "Sex upon Outcome",
    "Age upon Outcome",
    "Breed",
}


def standardize_column_name(column_name: str) -> str:
    """
    Convert one column name into a consistent Python-friendly format.
    """
    standardized_name = column_name.strip().lower()

    # Replace spaces and other special characters with underscores so
    # every column follows the same naming style.
    standardized_name = re.sub(r"[^a-z0-9]+", "_", standardized_name)

    # Remove any extra underscores that may have been added at the
    # beginning or end of the column name.
    return standardized_name.strip("_")


def standardize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with standardized column names.
    """
    prepared_dataframe = dataframe.copy()

    # Work with a copy so the original DataFrame is not changed
    # unexpectedly somewhere else in the project.
    prepared_dataframe.columns = [
        standardize_column_name(column_name)
        for column_name in prepared_dataframe.columns
    ]

    return prepared_dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Verify that a dataset contains all of the columns needed before
    preparing it for the rest of the project.
    """

    # Compare the required columns against the columns that actually
    # exist in the CSV file.
    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        missing_column_list = ", ".join(missing_columns)

        raise ValueError(
            f"{dataset_name} dataset is missing the following "
            f"required column(s): {missing_column_list}"
        )


def load_csv_file(
    file_path: Path | str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a CSV file and return it as a Pandas DataFrame.
    """

    # Resolve the path before loading so any error message shows the
    # exact file location Python attempted to use.
    resolved_path = Path(file_path).expanduser().resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} data file was not found: {resolved_path}"
        )

    try:
        return pd.read_csv(
            resolved_path,
            low_memory=False,
        )
    except pd.errors.ParserError as error:
        raise ValueError(
            f"{dataset_name} data file could not be parsed: "
            f"{resolved_path}"
        ) from error


def prepare_intake_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare an intake DataFrame for use by the project.
    """

    # Check the original column names before changing them so we can
    # clearly report if the source file is missing anything important.
    validate_required_columns(
        dataframe,
        INTAKE_REQUIRED_COLUMNS,
        "Intake",
    )

    prepared_dataframe = standardize_column_names(dataframe)

    # Convert the intake date into a Pandas datetime value so later
    # parts of the project can sort, compare, and match dates correctly.
    # Invalid dates become missing values instead of stopping the program.
    prepared_dataframe["datetime"] = pd.to_datetime(
        prepared_dataframe["datetime"],
        errors="coerce",
    )

    return prepared_dataframe


def prepare_outcome_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare an outcome DataFrame for use by the project.
    """

    # Check the original column names first so the error message matches
    # the headings someone would see when opening the source CSV file.
    validate_required_columns(
        dataframe,
        OUTCOME_REQUIRED_COLUMNS,
        "Outcome",
    )

    prepared_dataframe = standardize_column_names(dataframe)

    # Convert the outcome date so later parts of the project can sort
    # and compare outcome records without converting it each time.
    prepared_dataframe["datetime"] = pd.to_datetime(
        prepared_dataframe["datetime"],
        errors="coerce",
    )

    # Convert the date of birth as well because it will be useful later
    # when comparing ages and reviewing an animal's history.
    prepared_dataframe["date_of_birth"] = pd.to_datetime(
        prepared_dataframe["date_of_birth"],
        errors="coerce",
    )

    return prepared_dataframe


def load_intake_data(
    file_path: Path | str = DEFAULT_INTAKE_PATH,
) -> pd.DataFrame:
    """
    Load and prepare the intake dataset.
    """

    # Keep the loading and preparation steps together so any part of
    # the project can get a ready-to-use intake DataFrame with one call.
    intake_dataframe = load_csv_file(
        file_path,
        "Intake",
    )

    return prepare_intake_data(intake_dataframe)


def load_outcome_data(
    file_path: Path | str = DEFAULT_OUTCOME_PATH,
) -> pd.DataFrame:
    """
    Load and prepare the outcome dataset.
    """

    # Use the same process for the outcome file so both datasets are
    # handled consistently.
    outcome_dataframe = load_csv_file(
        file_path,
        "Outcome",
    )

    return prepare_outcome_data(outcome_dataframe)


def load_prepared_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare both datasets.

    Returns:
        A tuple containing the prepared intake and outcome DataFrames.
    """

    # Load each dataset using the same preparation steps so they are
    # both ready for matching and later analysis.
    intake_dataframe = load_intake_data()
    outcome_dataframe = load_outcome_data()

    return intake_dataframe, outcome_dataframe