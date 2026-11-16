"""Policy interface module for robot evaluation.

Provides interfaces for policies that need structured goals rather than
natural language task descriptions.

This module is OPTIONAL - the core gen+eval system is text-based.
These interfaces exist for policies that cannot interpret natural language.

The unified PolicyInterfaceAgent handles the entire pipeline:
1. Parses task → goal predicates + preconditions
2. Finds objects matching categories in scene
3. Verifies preconditions using state/vision tools
4. Returns ranked valid (target, reference) bindings
5. Computes exact poses for each binding (via PredicateResolver)
"""

from scenecode.robot_eval.policy_interface.policy_agent import (
    ObjectBinding,
    PolicyInterfaceAgent,
    PolicyInterfaceOutput,
)
from scenecode.robot_eval.policy_interface.predicate_resolver import (
    ExactPosePredicate,
    PredicateResolver,
    ResolverResult,
)

__all__ = [
    "ExactPosePredicate",
    "ObjectBinding",
    "PolicyInterfaceAgent",
    "PolicyInterfaceOutput",
    "PredicateResolver",
    "ResolverResult",
]
