import argparse
import json
import os
import time
from typing import Any

import streamlit as st

from osuniverse.utils import (
    WEIGHTS,
    calculate_stats,
    find_json_files,
    format_number,
    format_timestamp,
    load_scored_run,
    save_scored_run,
)


# Add caching to expensive operations
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_find_json_files(results_dir: str):
    return find_json_files(results_dir)


@st.cache_data(ttl=300)
def cached_calculate_stats(json_files: list[str]):
    return calculate_stats(json_files)


# Cache basic file metadata to avoid reloading files for the dropdown
@st.cache_data(ttl=300)
def get_file_metadata(json_files: list[str]) -> list[tuple[int, str, dict[str, Any]]]:
    """Load basic metadata from all files for the dropdown menu."""
    # Check if we have a cached version in session state
    cache_key = "_state_file_metadata_" + str(hash(tuple(json_files)))
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    file_options: list[tuple[int, str, dict[str, Any]]] = []
    for idx, file_path in enumerate(json_files):
        try:
            # Load just the basic info we need
            with open(file_path, "r") as f:
                data = json.load(f)
            category = data.get("category", "unknown")
            level = data.get("level", "unknown")
            ai_score = data.get("ai_score", -1)
            human_score = data.get("human_score", -1)
            name = os.path.basename(file_path)
            formatted_name = f"{category} | {level} | {name}"
            file_options.append(
                (
                    idx,
                    formatted_name,
                    {
                        "category": category,
                        "level": level,
                        "ai_score": ai_score,
                        "human_score": human_score,
                    },
                )
            )
        except Exception:
            # If we can't load the file, just show the filename
            file_options.append((idx, os.path.basename(file_path), {}))

    # Save to session state
    st.session_state.file_metadata_cache = file_options
    return file_options


# First, let's modify the update_file_metadata_in_cache function to make refreshing optional
def update_file_metadata_in_cache(
    json_files: list[str], index: int, refresh_list: bool = True
):
    """Force metadata reload by clearing all relevant caches."""
    # Only clear caches if refresh_list is True
    if refresh_list:
        # Clear the cache for get_file_metadata
        get_file_metadata.clear()  # type: ignore

        # Clear the file_metadata_cache from session state to force reload
        if "file_metadata_cache" in st.session_state:
            del st.session_state["file_metadata_cache"]

        # Clear any other cache keys related to file metadata
        for key in list(st.session_state.keys()):
            if key.startswith("_state_file_metadata_") or key.startswith(  # type: ignore
                "cache_file_metadata_"
            ):
                del st.session_state[key]

        # Set a flag to force dropdown refresh
        st.session_state.force_dropdown_refresh = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scored TestCaseRun Viewer")
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Directory to scan for JSON files",
    )
    # When running with streamlit, need to get rid of streamlit's own args
    streamlit_args = ["--server.port", "--server.address", "--server.headless"]
    argv: list[str] = [
        arg
        for arg in os.sys.argv[1:]  # type: ignore
        if not any(arg.startswith(s) for s in streamlit_args)  # type: ignore
    ]
    return parser.parse_args(argv)


def clear_session_data():
    """Clear only file-specific data from session state."""
    keys_to_clear = ["scored_run", "is_successful", "human_comment"]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def navigate_to_file(new_index: int):
    """Navigate to a specific file index and clear session data."""
    # Update to new index
    st.session_state.current_file_idx = new_index

    # Clear related session data
    clear_session_data()

    # Force immediate rerun for ALL navigation actions
    st.rerun()


def main():
    args = parse_args()
    st.set_page_config(layout="wide")

    # Initialize session states to prevent reruns
    if "current_file_idx" not in st.session_state:
        st.session_state.current_file_idx = 0

    if "needs_rerun" not in st.session_state:
        st.session_state.needs_rerun = False

    if "force_dropdown_refresh" not in st.session_state:
        st.session_state.force_dropdown_refresh = False

    # Store filter state for comparison
    if "previous_filter_state" not in st.session_state:
        st.session_state.previous_filter_state = False

    # Store current sort and filter settings
    if "current_sort_by" not in st.session_state:
        st.session_state.current_sort_by = "Category"

    # Replace individual filter flags with a single filter option
    if "current_filter" not in st.session_state:
        st.session_state.current_filter = "All"

    results_dir = args.dir or "results"

    # Main container with two columns - file list and content
    file_list_col, main_content_col = st.columns([1, 4])

    with file_list_col:
        st.markdown("### Test Files")

        # Directory input at top of file list
        results_dir = (
            st.text_input(
                "Results Directory",
                value=results_dir,
                help="Directory to scan for JSON files",
            )
            or "results"
        )

        # Move reload button to the immediate top position
        if st.button(
            "🔄 Reload File List", key="reload_file_list_top", use_container_width=True
        ):
            # Clear caches and force reload
            cached_find_json_files.clear()  # type: ignore
            cached_calculate_stats.clear()  # type: ignore
            get_file_metadata.clear()  # type: ignore
            st.session_state.needs_rerun = True
            st.rerun()

        # Add sorting and filtering controls
        sort_col, filter_col = st.columns(2)

        # Detect if sort/filter changes to force reload
        previous_sort_by = st.session_state.get("current_sort_by", "Category")
        previous_filter = st.session_state.get("current_filter", "All")

        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                options=["Category", "Level", "Name", "Score"],
                index=["Category", "Level", "Name", "Score"].index(previous_sort_by),
                key="sort_by",
            )

        with filter_col:
            # Replace checkboxes with a dropdown
            filter_option = st.selectbox(
                "Filter",
                options=["All", "Unreviewed only", "Reviewed only", "Disagreed only"],
                index=[
                    "All",
                    "Unreviewed only",
                    "Reviewed only",
                    "Disagreed only",
                ].index(previous_filter),
                key="filter_option",
            )

        # Apply changes immediately if settings changed
        if sort_by != previous_sort_by or filter_option != previous_filter:
            st.session_state.current_sort_by = sort_by

            # Store filter state
            filter_changed = filter_option != previous_filter
            st.session_state.current_filter = filter_option

            if filter_changed:
                clear_session_data()

            # Force refresh
            st.session_state.needs_rerun = True
            st.rerun()

        # Find JSON files and metadata
        json_files = cached_find_json_files(results_dir)
        if not json_files:
            st.error(f"No JSON files found in {results_dir}")
            return

        # Get the metadata for all files
        file_metadata = get_file_metadata(json_files)

        # Apply filtering based on selected filter option
        if filter_option == "Unreviewed only":  # Show only unreviewed files
            filtered_metadata = [
                (idx, name, meta)
                for idx, name, meta in file_metadata
                if meta.get("human_score", -1) == -1
            ]
        elif filter_option == "Reviewed only":  # Show only reviewed files
            filtered_metadata = [
                (idx, name, meta)
                for idx, name, meta in file_metadata
                if meta.get("human_score", -1) != -1
            ]
        elif (
            filter_option == "Disagreed only"
        ):  # Show only files where human and AI disagree
            filtered_metadata = [
                (idx, name, meta)
                for idx, name, meta in file_metadata
                if (
                    meta.get("human_score", -1) != -1  # Human reviewed
                    and meta.get("ai_score", -1) != -1  # AI reviewed
                    and (  # Disagreement: one passed, one failed
                        (
                            meta.get("human_score", -1) >= 1.0
                            and meta.get("ai_score", -1) < 1.0
                        )
                        or (
                            meta.get("human_score", -1) < 1.0
                            and meta.get("ai_score", -1) >= 1.0
                        )
                    )
                )
            ]
        else:  # "All" - Show all files
            filtered_metadata = file_metadata

        # Apply sorting based on selection
        if sort_by == "Category":  # Use the current value directly
            sorted_metadata = sorted(
                filtered_metadata, key=lambda x: x[2].get("category", "")
            )
        elif sort_by == "Level":
            # Sort by level importance (using WEIGHTS) from small to big
            sorted_metadata = sorted(
                filtered_metadata,
                key=lambda x: WEIGHTS.get(x[2].get("level", ""), 0),
                # Removed reverse=True to sort from small to big weight
            )
        elif sort_by == "Score":
            # Sort by human score first, then AI score
            sorted_metadata = sorted(
                filtered_metadata,
                key=lambda x: (
                    x[2].get("human_score", -1) != -1,  # Unreviewed last
                    x[2].get("human_score", -1),  # Then by human score
                    x[2].get("ai_score", -1),  # Then by AI score
                ),
                reverse=True,  # Higher scores first
            )
        else:  # Name is the default
            sorted_metadata = sorted(
                filtered_metadata, key=lambda x: os.path.basename(json_files[x[0]])
            )

        # Create mapping from original indices to sorted indices
        index_mapping = {meta[0]: i for i, meta in enumerate(sorted_metadata)}

        # If current index is filtered out, select the first available file
        current_idx_filtered_out = False

        if filter_option == "Unreviewed only":
            # Check if current file is reviewed
            current_idx_filtered_out = any(
                idx == st.session_state.current_file_idx
                and meta.get("human_score", -1) != -1
                for idx, _, meta in file_metadata
            )
        elif filter_option == "Reviewed only":
            # Check if current file is unreviewed
            current_idx_filtered_out = any(
                idx == st.session_state.current_file_idx
                and meta.get("human_score", -1) == -1
                for idx, _, meta in file_metadata
            )
        elif filter_option == "Disagreed only":
            # Check if current file doesn't have a disagreement
            current_idx_filtered_out = not any(
                idx == st.session_state.current_file_idx
                and meta.get("human_score", -1) != -1
                and meta.get("ai_score", -1) != -1
                and (
                    (
                        meta.get("human_score", -1) >= 1.0
                        and meta.get("ai_score", -1) < 1.0
                    )
                    or (
                        meta.get("human_score", -1) < 1.0
                        and meta.get("ai_score", -1) >= 1.0
                    )
                )
                for idx, _, meta in file_metadata
            )

        if current_idx_filtered_out and sorted_metadata:
            st.session_state.current_file_idx = sorted_metadata[0][0]

        # Generate a key that changes when we want to force a refresh
        dropdown_key = f"file_selector_{hash(tuple(json_files))}"
        if st.session_state.force_dropdown_refresh:
            # Add timestamp to force refresh
            dropdown_key += f"_{int(time.time())}"
            # Reset the flag
            st.session_state.force_dropdown_refresh = False

        # Use radio buttons with sorted and filtered metadata
        if sorted_metadata:
            selected_idx_in_list = st.radio(
                "Select test file",
                range(len(sorted_metadata)),
                format_func=lambda x: sorted_metadata[x][
                    1  # type: ignore
                ],  # Display name from sorted list
                index=index_mapping.get(st.session_state.current_file_idx, 0)
                if st.session_state.current_file_idx in [m[0] for m in sorted_metadata]
                else 0,
                label_visibility="collapsed",
                key=dropdown_key,  # Dynamic key forces component refresh
            )

            # Get the actual file index from the sorted metadata
            selected_file_idx = sorted_metadata[selected_idx_in_list][0]

            if selected_file_idx != st.session_state.current_file_idx:
                navigate_to_file(selected_file_idx)
        else:
            st.info("No files match the current filter.")

    # Main content area
    with main_content_col:
        # Create tabs for Overview and Details
        overview_tab, details_tab = st.tabs(["Run Overview", "Tests Details"])

        with overview_tab:
            stats = cached_calculate_stats(json_files)

            st.markdown("### General Statistics")
            general_data: list[dict[str, Any]] = []
            general_data.append(
                {
                    "Total Tests": stats["total"],
                    "Passed Tests": stats["passed"],
                    "Success Rate": f"{stats['success_rate']:.2f}%",
                    "Weighted Success Rate": f"{stats['weighted_success_rate']:.2f}%",
                    "Avg Duration (s)": f"{int(float(stats['total_duration']) / stats['total'])}",  # type: ignore
                    "Total Duration (s)": f"{int(stats['total_duration'])}",  # type: ignore
                }
            )
            st.dataframe(general_data, hide_index=True, use_container_width=True)  # type: ignore

            token_data: list[dict[str, Any]] = []
            token_data.append(
                {
                    "Avg Input Tokens": format_number(
                        int(float(stats["total_input_tokens"]) / stats["total"])  # type: ignore
                    ),
                    "Total Input Tokens": format_number(
                        int(stats["total_input_tokens"])  # type: ignore
                    ),
                    "Avg Output Tokens": format_number(
                        int(float(stats["total_output_tokens"]) / stats["total"])  # type: ignore
                    ),
                    "Total Output Tokens": format_number(
                        int(stats["total_output_tokens"])  # type: ignore
                    ),
                    "Total Eval Input Tokens": format_number(
                        int(stats["total_validation_input_tokens"])  # type: ignore
                    ),
                    "Total Eval Output Tokens": format_number(
                        int(stats["total_validation_output_tokens"])  # type: ignore
                    ),
                }
            )
            st.dataframe(token_data, hide_index=True, use_container_width=True)  # type: ignore

            # Add new section for category-level matrix
            st.markdown("### Category-Level Statistics")

            # Get unique levels and categories
            all_levels = sorted(stats["by_levels"].keys(), key=lambda x: WEIGHTS[x])  # type: ignore
            all_categories = sorted(stats["by_categories"].keys())  # type: ignore

            # Create matrix data
            matrix_data: list[dict[str, Any]] = []
            category_weighted_success_rate: dict[str, float] = {}
            for category in all_categories:
                row_data = {"Category": category}
                weighted_passed: float = 0
                weighted_total: float = 0
                for level in all_levels:
                    cat_level_data: dict[str, int] = (  # type: ignore
                        stats["by_categories_by_level"].get(category, {}).get(level, {})  # type: ignore
                    )
                    passed = cat_level_data.get("passed", 0)  # type: ignore
                    amount = cat_level_data.get("amount", 0)  # type: ignore
                    if amount > 0:
                        value = f"{passed}/{amount} ({passed / amount * 100:.2f}%)"
                    else:
                        value = "N/A"
                    weighted_passed += passed * WEIGHTS[level]  # type: ignore
                    weighted_total += amount * WEIGHTS[level]  # type: ignore
                    row_data[level] = value
                if weighted_total > 0:
                    row_data["Weighted Success Rate"] = (
                        f"{weighted_passed / weighted_total * 100:.2f}%"
                    )
                    category_weighted_success_rate[category] = (
                        weighted_passed / weighted_total * 100
                    )
                else:
                    row_data["Weighted Success Rate"] = "N/A"
                    category_weighted_success_rate[category] = 0
                matrix_data.append(row_data)
            row_data = {"Category": "Total for level"}
            for level, level_stats in sorted(stats["by_levels"].items()):  # type: ignore
                row_data[level] = (
                    f"{level_stats['passed']}/{level_stats['amount']} ({level_stats['passed'] / level_stats['amount'] * 100:.2f}%)"
                )
            row_data["Weighted Success Rate"] = f"{stats['weighted_success_rate']:.2f}%"
            matrix_data.append(row_data)

            if matrix_data:
                st.dataframe(  # type: ignore
                    matrix_data,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No category-level matrix data available")

            # Add tables for levels and categories after the main stats
            st.markdown("### Statistics by Level")

            # Create a DataFrame for levels
            level_data: list[dict[str, Any]] = []
            for level, level_stats in sorted(stats["by_levels"].items()):  # type: ignore
                success_rate = (
                    (level_stats["passed"] / level_stats["amount"] * 100)
                    if level_stats["amount"] > 0
                    else 0
                )
                level_data.append(
                    {
                        "Level": level,
                        "Total": level_stats["amount"],
                        "Passed": level_stats["passed"],
                        "Success Rate": f"{success_rate:.2f}%",
                        "Avg Duration (s)": f"{int(float(level_stats['duration']) / level_stats['amount'])}",
                        "Avg Input Tokens": format_number(
                            int(
                                float(level_stats["input_tokens"])
                                / level_stats["amount"]
                            )
                        ),
                        "Avg Output Tokens": format_number(
                            int(
                                float(level_stats["output_tokens"])
                                / level_stats["amount"]
                            )
                        ),
                        "Total Duration (s)": f"{int(level_stats['duration'])}",
                        "Total In Tokens": format_number(
                            int(level_stats["input_tokens"])
                        ),
                        "Total Out Tokens": format_number(
                            int(level_stats["output_tokens"])
                        ),
                        "Total Eval In Tokens": format_number(
                            int(level_stats["validation_input_tokens"])
                        ),
                        "Total Eval Out Tokens": format_number(
                            int(level_stats["validation_output_tokens"])
                        ),
                    }
                )

            level_data = sorted(level_data, key=lambda x: WEIGHTS[x["Level"]])

            if level_data:
                st.dataframe(  # type: ignore
                    level_data,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No level statistics available")

            st.markdown("### Statistics by Category")

            # Create a DataFrame for categories
            category_data: list[dict[str, Any]] = []
            for category, category_stats in sorted(stats["by_categories"].items()):  # type: ignore
                success_rate = (
                    (category_stats["passed"] / category_stats["amount"] * 100)
                    if category_stats["amount"] > 0
                    else 0
                )
                category_data.append(
                    {
                        "Category": category,
                        "Total": category_stats["amount"],
                        "Passed": category_stats["passed"],
                        "Success Rate": f"{success_rate:.2f}%",
                        "Weighted Success Rate": f"{category_weighted_success_rate[category]:.2f}%",
                        "Avg Duration (s)": f"{int(float(category_stats['duration']) / category_stats['amount'])}",
                        "Avg In Tokens": format_number(
                            int(
                                float(category_stats["input_tokens"])
                                / category_stats["amount"]
                            )
                        ),
                        "Avg Out Tokens": format_number(
                            int(
                                float(category_stats["output_tokens"])
                                / category_stats["amount"]
                            )
                        ),
                        "Total Duration (s)": f"{int(category_stats['duration'])}",
                        "Total In Tokens": format_number(
                            int(category_stats["input_tokens"])
                        ),
                        "Total Out Tokens": format_number(
                            int(category_stats["output_tokens"])
                        ),
                        "Total Eval In Tokens": format_number(
                            int(category_stats["validation_input_tokens"])
                        ),
                        "Total Eval Out Tokens": format_number(
                            int(category_stats["validation_output_tokens"])
                        ),
                    }
                )

            if category_data:
                st.dataframe(  # type: ignore
                    category_data,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No category statistics available")

            st.markdown("### Validation Statistics")
            validation_data: list[dict[str, Any]] = []
            validation_data.append(
                {
                    "Total Tests": stats["total"],
                    "Loaded": stats["loaded"],
                    "AI Passed": stats["ai_passed"],
                    "AI Errors": stats["ai_errors"],
                    "AI Success Rate": f"{stats['ai_success_rate']:.2f}%",
                    "Human Passed": stats["human_passed"],
                    "Human Failed": stats["human_failed"],
                    "Human Unreviewed": stats["human_unreviewed"],
                    "Human Success Rate": f"{stats['human_success_rate']:.2f}%",
                    "Disagreements": stats["disagreements"],
                    "Disagreement Rate": f"{stats['disagreement_rate']:.2f}%",
                }
            )
            st.dataframe(  # type: ignore
                validation_data,
                hide_index=True,
                use_container_width=True,
            )

        with details_tab:
            # Reset current_file_idx if directory changes
            if (
                "last_dir" not in st.session_state
                or st.session_state.last_dir != results_dir
            ):
                st.session_state.current_file_idx = 0
                st.session_state.last_dir = results_dir
                clear_session_data()

            current_file = json_files[st.session_state.current_file_idx]

            # Create a reverse mapping from original indices to positions in the sorted_metadata list
            original_to_sorted_position = {
                meta[0]: i for i, meta in enumerate(sorted_metadata)
            }

            # Navigation buttons at top of content area
            prev_col, next_col, reload_col = st.columns([1, 1, 5])

            with prev_col:
                if st.button("⬅️ Previous Test"):
                    # Find current position in the sorted/filtered list
                    if st.session_state.current_file_idx in original_to_sorted_position:
                        current_pos = original_to_sorted_position[
                            st.session_state.current_file_idx
                        ]
                        # Get the previous item in the sorted/filtered list
                        prev_pos = (current_pos - 1) % len(sorted_metadata)
                        # Navigate to the original index of that item
                        navigate_to_file(sorted_metadata[prev_pos][0])
                    else:
                        # Current file not in filtered list, go to last item
                        navigate_to_file(sorted_metadata[-1][0])

            with next_col:
                if st.button("Next Test ➡️"):
                    # Find current position in the sorted/filtered list
                    if st.session_state.current_file_idx in original_to_sorted_position:
                        current_pos = original_to_sorted_position[
                            st.session_state.current_file_idx
                        ]
                        # Get the next item in the sorted/filtered list
                        next_pos = (current_pos + 1) % len(sorted_metadata)
                        # Navigate to the original index of that item
                        navigate_to_file(sorted_metadata[next_pos][0])
                    else:
                        # Current file not in filtered list, go to first item
                        navigate_to_file(sorted_metadata[0][0])

            with reload_col:
                # Display current file name
                current_metadata = file_metadata[st.session_state.current_file_idx]
                st.markdown(f"**Current file:** {current_metadata[1]}")

            # Load the current file
            if "scored_run" not in st.session_state:
                try:
                    run = load_scored_run(current_file)
                    st.session_state["scored_run"] = run
                except Exception as e:
                    st.error(f"Error loading file {current_file}: {str(e)}")
                    return

            run = st.session_state["scored_run"]

            # Create two columns for the main content
            left_col, _, right_col = st.columns([3, 1, 7])

            with right_col:
                st.subheader("Test Information")

                # Convert data to markdown table
                markdown_table = ""

                # Convert JSON to table format
                info_data = [
                    ("ID", run.id),
                    ("Name", run.name),
                    ("Category", run.category),
                    ("Level", run.level),
                    ("Task", run.task),
                    ("Setup Command", run.setup_cmd),
                    ("Desktop Image", run.desktop_image),
                    ("Agent YAML", run.agent_yaml),
                    ("Agent Model", run.agent_model),
                    ("Status", run.status),
                    ("Max Steps", run.max_steps),
                    ("Input Tokens", format_number(run.input_tokens)),
                    ("Output Tokens", format_number(run.output_tokens)),
                    ("Eval Input Tokens", format_number(run.validation_input_tokens)),
                    ("Eval Output Tokens", format_number(run.validation_output_tokens)),
                    (
                        "Duration",
                        f"{run.trajectory[-1].timestamp - run.trajectory[0].timestamp:.1f}s",
                    ),
                ]

                if run.checks and len(run.checks) > 0:
                    for check in run.checks:
                        value = check.to_dict()["value"]
                        if check.CHECK_TYPE == "command_output":
                            value = check.to_dict()["command"] + " 🔹 " + value
                        info_data.append(
                            (
                                "Check 🔹 " + check.CHECK_TYPE + " 🔹",
                                value,
                            )
                        )

                if (
                    run.command_output_check_results
                    and len(run.command_output_check_results) > 0
                ):
                    for check in run.command_output_check_results:
                        info_data.append(
                            (
                                "Output for 🔹 " + check.command + " 🔹",
                                check.output,
                            )
                        )

                # Create markdown table
                markdown_table = "| Field | Value |\n|---|---|\n"
                for field, value in info_data:
                    # Ensure proper markdown table formatting by replacing pipes and newlines
                    safe_value = str(value).replace("|", "\\|").replace("\n", "<br>")
                    markdown_table += f"| **{field}** | {safe_value} |\n"

                st.markdown(markdown_table)

            with left_col:
                st.subheader("Test Results")

                # AI Results (read-only)
                st.markdown("#### AI Evaluation")
                st.text(f"Score: {run.ai_score} {run.ai_score >= 1.0 and '✅' or '❌'}")
                st.markdown("**Comment:**")
                st.markdown(run.ai_comment)

                # Human Review section
                st.markdown("#### Human Review")

                # Display current human review status
                human_status_icon = (
                    "✅"
                    if run.human_score >= 1.0
                    else "❌"
                    if run.human_score == 0
                    else "❓"
                )
                human_status_text = (
                    "Passed"
                    if run.human_score >= 1.0
                    else "Failed"
                    if run.human_score == 0
                    else "Not Reviewed"
                )
                st.markdown(
                    f"**Current Status: {human_status_icon} {human_status_text}**"
                )

                # Reset comment field when switching files
                if "human_comment" not in st.session_state:
                    st.session_state.human_comment = (
                        run.human_comment if run.human_comment else ""
                    )

                if st.button("✅ Mark as Passed", use_container_width=True):
                    run.human_score = 1.0
                    save_scored_run(current_file, run)

                    # Clear session data but don't update metadata cache
                    st.session_state.pop("scored_run", None)

                    # Pass refresh_list=False to prevent file list refresh
                    update_file_metadata_in_cache(
                        json_files,
                        st.session_state.current_file_idx,
                        refresh_list=False,
                    )
                    st.rerun()

                if st.button("❌ Mark as Failed", use_container_width=True):
                    run.human_score = 0.0
                    save_scored_run(current_file, run)

                    # Clear session data but don't update metadata cache
                    st.session_state.pop("scored_run", None)

                    # Pass refresh_list=False to prevent file list refresh
                    update_file_metadata_in_cache(
                        json_files,
                        st.session_state.current_file_idx,
                        refresh_list=False,
                    )
                    st.rerun()

            # Trajectory section below both columns
            st.markdown("---")  # Add a separator

            # Show the final result first if available
            if run.result:
                st.markdown("#### Final Result")
                left_col, right_col = st.columns([1, 2])
                with left_col:
                    st.markdown(
                        f"**Timestamp:** {format_timestamp(run.result.timestamp)}"
                    )
                    st.markdown(f"**Action:** {run.result.action}")

                with right_col:
                    if run.result.screenshot and run.result.screenshot.startswith(
                        "data:image"
                    ):
                        st.image(
                            run.result.screenshot,
                            caption="Final screenshot",
                            use_container_width=True,
                        )
            st.markdown("---")

            # Trajectory steps
            st.markdown("#### Execution Steps")
            for i, step in enumerate(reversed(run.trajectory), start=1):
                with st.container():
                    left_col, right_col = st.columns([1, 2])
                    with left_col:
                        st.markdown(f"##### Step {len(run.trajectory) - i + 1}")
                        st.markdown(
                            f"**Timestamp:** {format_timestamp(step.timestamp)}"
                        )
                        st.markdown(f"**Action:** {step.action}")
                        if step.thought:
                            st.markdown(f"**Thought:** {step.thought}")

                    with right_col:
                        if step.screenshot and step.screenshot.startswith("data:image"):
                            st.image(
                                step.screenshot,
                                caption=f"Screenshot for step {len(run.trajectory) - i + 1}",
                                use_container_width=True,
                            )
                    st.divider()

    # Handle rerun at the end of the function if needed
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.rerun()


if __name__ == "__main__":
    main()
