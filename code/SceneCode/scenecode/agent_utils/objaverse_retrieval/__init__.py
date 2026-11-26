"""Objaverse (ObjectThor) object library retrieval system.

Provides access to ~50K pre-scaled 3D objects from the ObjectThor dataset
using CLIP-based semantic search.
"""

from scenecode.agent_utils.objaverse_retrieval.config import ObjaverseConfig
from scenecode.agent_utils.objaverse_retrieval.retrieval import ObjaverseRetriever

__all__ = [
    "ObjaverseConfig",
    "ObjaverseRetriever",
]
