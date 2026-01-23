# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tool trajectory metrics implementation."""

import json
from typing import Any, Dict, List, Tuple


def _get_tool_calls(instance: dict) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]]
]:
    """Extracts reference and predicted tool calls from an instance.

    Args:
        instance: The instance dictionary containing 'reference_trajectory' and 'intermediate_events'.

    Returns:
        A tuple containing:
            - reference_tool_calls: List of reference tool calls.
            - tool_calls: List of predicted tool calls extracted from intermediate events.
    """
    if isinstance(instance, str):
        instance = json.loads(instance)
    intermediate_events = instance.get("intermediate_events", [])
    reference_tool_calls = instance.get("reference_trajectory", [])
    tool_calls = []
    for event in intermediate_events:
        if (
            "content" not in event
            or "parts" not in event["content"]
        ):
            continue
        for part in event["content"]["parts"]:
            function_call = part.get("function_call")
            if not function_call:
                continue
            tool_name = function_call["name"]
            tool_input = function_call["args"]
            tool_calls.append(
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input
                }
            )
    return (reference_tool_calls, tool_calls)


def _get_tool_match(
    tool_name: str,
    reference_tool_name: str,
    tool_input: dict[str, Any],
    reference_tool_input: dict[str, Any],
) -> bool:
    """Checks if a predicted tool call matches a reference tool call.

    Args:
        tool_name: The name of the predicted tool.
        reference_tool_name: The name of the reference tool.
        tool_input: The arguments of the predicted tool.
        reference_tool_input: The arguments of the reference tool.

    Returns:
        True if the tool names match and inputs match (or reference input is None), False otherwise.
    """
    if (
        tool_name == reference_tool_name
        and (tool_input == reference_tool_input or reference_tool_input is None)
    ):
        return True
    else:
        return False


def trajectory_exact_match_func(instance: dict) -> float:
    """Calculates the exact match score for tool trajectory.

    Requires precise match of tool calls in the exact order.

    Args:
        instance: The instance dictionary.

    Returns:
        The score (1.0 for exact match, 0.0 otherwise).
    """
    score = 1.0
    reference_tool_calls, tool_calls = _get_tool_calls(instance)
    if len(reference_tool_calls) != len(tool_calls):
        score = 0.0
    else:
        for tool_call, ref_tool_call in zip(
            tool_calls, reference_tool_calls
        ):
            tool_call_name = tool_call["tool_name"]
            ref_tool_call_name = ref_tool_call["tool_name"]
            tool_call_args = tool_call["tool_input"]
            ref_tool_call_args = ref_tool_call["tool_input"]
            if not _get_tool_match(
                tool_call_name,
                ref_tool_call_name,
                tool_call_args,
                ref_tool_call_args
            ):
                score = 0.0
    return score


def trajectory_in_order_match_func(instance: dict) -> float:
    """Calculates the in-order match score for tool trajectory.

    Checks if predicted tool calls match reference tool calls in the correct relative order,
    ignoring extra intervening calls.

    Args:
        instance: The instance dictionary.

    Returns:
        The score (1.0 for exact match, 0.0 otherwise).
    """
    score = 1.0
    reference_tool_calls, tool_calls = _get_tool_calls(instance)
    reference_tools = [
        tool_call["tool_name"] for tool_call in reference_tool_calls
    ]
    similar_tools_calls = []
    for tool_call in tool_calls:
        if tool_call["tool_name"] in reference_tools:
            similar_tools_calls.append(tool_call)
    for tool_call, ref_tool_call in zip(
        similar_tools_calls, reference_tool_calls
    ):
        tool_call_name = tool_call["tool_name"]
        ref_tool_call_name = ref_tool_call["tool_name"]
        tool_call_args = tool_call["tool_input"]
        ref_tool_call_args = ref_tool_call["tool_input"]
        if not _get_tool_match(
            tool_call_name,
            ref_tool_call_name,
            tool_call_args,
            ref_tool_call_args
        ):
            score = 0.0
    return score


def _get_matches_count(
    tool_calls: List[Dict[str, Any]],
    reference_tool_calls: List[Dict[str, Any]]
) -> int:
    """Calculates the number of matches between predicted and reference tool calls.

    Matches are determined using _get_tool_match and are greedy (first available match is taken).

    Args:
        tool_calls: List of predicted tool calls.
        reference_tool_calls: List of reference tool calls.

    Returns:
        The count of matched tool calls.
    """
    matches = 0
    # Create a copy of indices to track which reference calls have been matched
    unmatched_ref_indices = list(range(len(reference_tool_calls)))

    for tool_call in tool_calls:
        match_found = False
        for ref_idx in unmatched_ref_indices:
            ref_tool_call = reference_tool_calls[ref_idx]
            if _get_tool_match(
                tool_call["tool_name"],
                ref_tool_call["tool_name"],
                tool_call["tool_input"],
                ref_tool_call["tool_input"],
            ):
                match_found = True
                unmatched_ref_indices.remove(ref_idx)
                break
        if match_found:
            matches += 1
    return matches


def trajectory_any_order_match_func(instance: dict) -> float:
    """Calculates the any-order match score.

    Checks if the set of predicted tool calls exactly matches the set of reference tool calls,
    regardless of order.

    Args:
        instance: The instance dictionary.

    Returns:
        The score (1.0 for exact match, 0.0 otherwise).
    """
    reference_tool_calls, tool_calls = _get_tool_calls(instance)
    score = 0.0
    if len(reference_tool_calls) == len(tool_calls):
        matches = _get_matches_count(tool_calls, reference_tool_calls)
        if matches == len(reference_tool_calls):
            score = 1.0
    return score


def trajectory_precision_func(instance: dict) -> float:
    """Calculates the precision of predicted tool calls.

    Precision = (Number of matches) / (Total predicted calls).

    Args:
        instance: The instance dictionary.

    Returns:
        The precision score.
    """
    reference_tool_calls, tool_calls = _get_tool_calls(instance)
    # If no tools were predicted, precision is 0.0 unless reference was also empty
    # If reference is empty and tool_calls is empty -> 1.0
    if not tool_calls:
        return 1.0 if not reference_tool_calls else 0.0

    matches = _get_matches_count(tool_calls, reference_tool_calls)
    score = matches / len(tool_calls)
    return score


def trajectory_recall_func(instance: dict) -> float:
    """Calculates the recall of predicted tool calls.

    Recall = (Number of matches) / (Total reference calls).

    Args:
        instance: The instance dictionary.

    Returns:
        A the recall score.
    """
    reference_tool_calls, tool_calls = _get_tool_calls(instance)
    # If no tools in reference, recall is 1.0 (trivial success)
    if not reference_tool_calls:
        return 1.0

    matches = _get_matches_count(tool_calls, reference_tool_calls)
    score = matches / len(reference_tool_calls)
    return score
