"""Versioned motion prompts for passive Physics→WM model inputs.

This changes model conditioning only. The frozen task, physical reference and
independent evaluator remain the source of task meaning and scoring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .agentic_contracts import AgentTask
from .lingbot_world2_adapter import validate_event_prompt


PROFILE = "motion_events_v2"
MINIMAL_PROFILE = "event_minimal_v3"
SOURCE_ABLATION_SHA256 = "a2ef36d78cbdac2f0982da10b8f25b1ef5a6b399f3050a41e4621dca484ad37d"


def compile_motion_prompt(task: AgentTask) -> str:
    """Express the declared trigger, visible motion, and target event in order."""
    if task.embodiment is not None or len(task.operation_program) != 1:
        raise ValueError("motion event prompt needs one passive operation")
    operation = task.operation_program[0]
    if operation.operation != "release" or len(operation.subjects) != 1:
        raise ValueError("motion event profile does not cover this passive operation")
    subject_id = operation.subjects[0]
    objects = {obj.object_id: obj for obj in task.objects}
    subject = objects[subject_id]
    contacts = [predicate for predicate in task.success_predicates
                if predicate.kind == "contact" and predicate.subject == subject_id
                and predicate.object in objects]
    if len(contacts) != 1:
        raise ValueError("motion event profile needs one declared contact target")
    target = objects[contacts[0].object]
    subject_text = subject.visual_description.strip().rstrip(". ")
    target_text = target.visual_description.strip().rstrip(". ")
    for article in ("a ", "an ", "the "):
        if subject_text.casefold().startswith(article):
            subject_text = subject_text[len(article):]
        if target_text.casefold().startswith(article):
            target_text = target_text[len(article):]
    if not subject_text or not target_text:
        raise ValueError("motion event profile needs visible object descriptions")
    initial = task.initial_state.strip().rstrip(". ")
    if not initial or "\n" in initial:
        raise ValueError("motion event profile needs one explicit initial state")
    prompt = (
        f"A {subject_text} starts above a {target_text}. "
        f"{initial}. "
        f"At the start, the {subject.category.replace('_', ' ')} is released. "
        "It immediately falls downward under gravity, visibly approaches "
        f"the {target.category.replace('_', ' ')}, and contacts its surface. "
        "After impact, it makes a brief settling motion."
    )
    return validate_event_prompt(prompt)


def compile_minimal_prompt(task: AgentTask) -> str:
    """Front-load a short, ordered event, matching the prompt-only ablation."""
    # Reuse the same declared release/contact admission as the longer profile.
    compile_motion_prompt(task)
    subject_id = task.operation_program[0].subjects[0]
    subject = next(item for item in task.objects if item.object_id == subject_id)
    target_id = next(item.object for item in task.success_predicates
                     if item.kind == "contact" and item.subject == subject_id)
    target = next(item for item in task.objects if item.object_id == target_id)
    appearance = re.split(r"\bwith\b", subject.visual_description, maxsplit=1,
                          flags=re.IGNORECASE)[0].strip().rstrip(". ,")
    appearance = re.sub(r"^(?:a|an|the)\s+", "", appearance, flags=re.IGNORECASE)
    if len(appearance.split()) > 7:
        appearance = subject.category.replace("_", " ")
    height = re.search(r"\b(\d+(?:\.\d+)?)\s*(m|cm)\s+above\b",
                       task.initial_state, flags=re.IGNORECASE)
    above = (f"{height.group(1)} {height.group(2)} above"
             if height else "above")
    destination = target.category.replace("_", " ")
    prompt = (
        f"A {appearance} is released from {above} the {destination}. "
        f"It immediately falls visibly downward under gravity, contacts the "
        f"{destination}, and settles on it."
    )
    return validate_event_prompt(prompt)


def select_prompt(task: AgentTask, model_profile: str, prompt_profile: str) -> tuple[str, dict]:
    if model_profile != "lingbot_world2_14b_causal_fast":
        raise ValueError("motion event profile requires the registered World2 model")
    if prompt_profile == "frozen_task":
        prompt = task.model_prompts.get(model_profile)
        if not prompt:
            raise ValueError("World2 task needs a frozen model prompt")
        prompt = validate_event_prompt(prompt)
    elif prompt_profile == PROFILE:
        prompt = compile_motion_prompt(task)
    elif prompt_profile == MINIMAL_PROFILE:
        prompt = compile_minimal_prompt(task)
    else:
        raise ValueError("unknown Physics→WM prompt profile")
    receipt = {
        "schema_version": "worldolympiad_physics_prompt_receipt.v1",
        "profile": prompt_profile,
        "task_sha256": task.content_hash(),
        "model_profile": model_profile,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "source_ablation_sha256": (SOURCE_ABLATION_SHA256
                                   if prompt_profile in {PROFILE, MINIMAL_PROFILE} else None),
    }
    return prompt, receipt


def verify_prompt_receipt(path: Path, task: AgentTask, model_profile: str,
                          prompt_profile: str) -> str:
    prompt, expected = select_prompt(task, model_profile, prompt_profile)
    if json.loads(path.read_text()) != expected:
        raise ValueError("published Physics→WM prompt receipt differs from selected profile")
    return prompt
