"""
Focused tests for reusable dashboard and location helpers.
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

from dashboard_helpers import (
    NO_LOCATION_MESSAGE,
    build_geocode_queries,
    coordinates_are_valid,
    expand_street_abbreviations,
    get_display_location,
    get_selected_candidate,
    has_usable_location,
    is_specific_address,
    normalize_location,
)


class LocationHelperTests(
    unittest.TestCase
):
    def test_has_usable_location_rejects_missing_values(
        self,
    ):
        missing_values = [
            None,
            "",
            "   ",
            pd.NA,
            float("nan"),
            "<NA>",
            "nan",
            "NAN",
            "None",
            "NONE",
            "Not available",
            "Unavailable",
            "unavailable",
            NO_LOCATION_MESSAGE,
        ]

        for value in missing_values:
            with self.subTest(
                value=value
            ):
                self.assertFalse(
                    has_usable_location(
                        value
                    )
                )

    def test_has_usable_location_accepts_real_text(
        self,
    ):
        self.assertTrue(
            has_usable_location(
                "4601 Imperial Dr in Austin (TX)"
            )
        )

    def test_display_location_preserves_original_text(
        self,
    ):
        location = (
            "4601 Imperial Dr in Austin (TX)"
        )

        self.assertEqual(
            get_display_location(
                location
            ),
            location,
        )

    def test_display_location_uses_standard_missing_message(
        self,
    ):
        self.assertEqual(
            get_display_location(
                "unavailable"
            ),
            NO_LOCATION_MESSAGE,
        )

    def test_normalize_archive_address(
        self,
    ):
        self.assertEqual(
            normalize_location(
                "4601 Imperial Dr in Austin (TX)"
            ),
            "4601 Imperial Dr, Austin, TX",
        )

    def test_normalize_parenthetical_tx_is_case_insensitive(
        self,
    ):
        self.assertEqual(
            normalize_location(
                "100 Main Rd in Del Valle ( tx )"
            ),
            "100 Main Rd, Del Valle, TX",
        )

    def test_normalize_adds_texas_when_missing(
        self,
    ):
        self.assertEqual(
            normalize_location(
                "100 Main Rd, Del Valle"
            ),
            "100 Main Rd, Del Valle, Texas",
        )

    def test_expand_common_abbreviations(
        self,
    ):
        self.assertEqual(
            expand_street_abbreviations(
                (
                    "10 Oak Dr, "
                    "20 Pine Rd, "
                    "30 Cedar Ln"
                )
            ),
            (
                "10 Oak Drive, "
                "20 Pine Road, "
                "30 Cedar Lane"
            ),
        )

    def test_build_queries_preserves_del_valle(
        self,
    ):
        queries = build_geocode_queries(
            "100 Main Rd in Del Valle (TX)"
        )

        self.assertEqual(
            queries[0],
            "100 Main Rd, Del Valle, TX",
        )

        self.assertIn(
            "100 Main Road, Del Valle, TX",
            queries,
        )

        self.assertFalse(
            any(
                "Austin" in query
                for query in queries
            )
        )

    def test_build_queries_adds_properly_ordered_austin_fallback(
        self,
    ):
        queries = build_geocode_queries(
            "123 Main Rd"
        )

        self.assertIn(
            "123 Main Rd, Austin, Texas",
            queries,
        )

        self.assertIn(
            (
                "123 Main Road, "
                "Travis County, Texas"
            ),
            queries,
        )

        self.assertNotIn(
            "123 Main Rd, Texas, Austin",
            queries,
        )

    def test_build_queries_returns_empty_for_missing_location(
        self,
    ):
        self.assertEqual(
            build_geocode_queries(
                None
            ),
            [],
        )

    def test_specific_address_requires_a_number(
        self,
    ):
        self.assertTrue(
            is_specific_address(
                "4601 Imperial Dr in Austin (TX)"
            )
        )

        self.assertFalse(
            is_specific_address(
                "Austin (TX)"
            )
        )

        self.assertFalse(
            is_specific_address(
                None
            )
        )

    def test_coordinates_are_valid(
        self,
    ):
        self.assertTrue(
            coordinates_are_valid(
                (
                    30.2672,
                    -97.7431,
                )
            )
        )

        self.assertTrue(
            coordinates_are_valid(
                [
                    "30.2672",
                    "-97.7431",
                ]
            )
        )

    def test_coordinates_reject_invalid_values(
        self,
    ):
        invalid_coordinates = [
            None,
            (),
            (
                30.0,
            ),
            (
                30.0,
                -97.0,
                1.0,
            ),
            (
                "bad",
                -97.0,
            ),
            (
                math.nan,
                -97.0,
            ),
            (
                math.inf,
                -97.0,
            ),
            (
                91.0,
                -97.0,
            ),
            (
                30.0,
                -181.0,
            ),
        ]

        for coordinates in invalid_coordinates:
            with self.subTest(
                coordinates=coordinates
            ):
                self.assertFalse(
                    coordinates_are_valid(
                        coordinates
                    )
                )


class CandidateSelectionTests(
    unittest.TestCase
):
    def test_no_selection_data_returns_none(
        self,
    ):
        self.assertIsNone(
            get_selected_candidate(
                None,
                None,
            )
        )

    def test_empty_table_returns_none(
        self,
    ):
        self.assertIsNone(
            get_selected_candidate(
                [],
                {
                    "row": 0
                },
            )
        )

    def test_invalid_row_safely_uses_first_candidate(
        self,
    ):
        view_data = [
            {
                "animal_id": "A1"
            },
            {
                "animal_id": "A2"
            },
        ]

        candidate = get_selected_candidate(
            view_data,
            {
                "row": 99
            },
        )

        self.assertEqual(
            candidate[
                "animal_id"
            ],
            "A1",
        )

    def test_non_dictionary_row_returns_none(
        self,
    ):
        self.assertIsNone(
            get_selected_candidate(
                [
                    "bad row"
                ],
                {
                    "row": 0
                },
            )
        )


if __name__ == "__main__":
    unittest.main()