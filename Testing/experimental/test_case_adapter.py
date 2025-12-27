from __future__ import annotations
from typing import Any, Dict

def to_llm_test_case(payload: Dict[str, Any]):
    from deepeval.test_case import LLMTestCase
    return LLMTestCase(**payload)

def to_conversational_test_case(payload: Dict[str, Any]):
    """
    Gap (intentional): implement Conversation Trace → ConversationalTestCase mapping.
    Your schema defines ConversationalTestCase and Turn structure. :contentReference[oaicite:7]{index=7}
    """
    raise NotImplementedError("ConversationalTestCase adapter not implemented in v1.")

def to_arena_test_case(payload: Dict[str, Any]):
    """
    Gap (intentional): implement ArenaTestCase mapping.
    """
    raise NotImplementedError("ArenaTestCase adapter not implemented in v1.")
