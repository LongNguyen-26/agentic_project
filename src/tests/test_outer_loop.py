import unittest
from unittest import mock
from devday_agent.agent.nodes import outer_loop
from devday_agent.agent.state import OuterState

class OuterLoopResilienceTests(unittest.TestCase):
    def test_submit_failure_retains_state_for_retry(self):
        """Scenario 1: failed submit must keep current_task for retry."""
        fake_client = mock.Mock()
        # Simulate first submit attempt failure.
        fake_client.submit_task_result.return_value = False
        
        state: OuterState = {
            "session_id": "session-123",
            "access_token": "token",
            "current_task": {"id": "task-1"},
            "task_result": {"answers": ["A"], "thought_log": "test", "used_tools": []},
            "should_continue": True,
            "error": None,
            "planning_hints": ""
        }
        
        with mock.patch.object(outer_loop, "_get_client", return_value=fake_client):
            next_state = outer_loop.submit_node(state)
            
        # Empty update means current_task is preserved in existing state.
        self.assertEqual(next_state, {})
        self.assertEqual(fake_client.submit_task_result.call_count, 1)

    def test_submit_success_clears_state(self):
        """Successful submit should clear current_task and task_result."""
        fake_client = mock.Mock()
        fake_client.submit_task_result.return_value = True
        
        state: OuterState = {
            "session_id": "session-123",
            "access_token": "token",
            "current_task": {"id": "task-1"},
            "task_result": {"answers": ["A"], "thought_log": "test", "used_tools": []},
            "should_continue": True,
            "error": None,
            "planning_hints": ""
        }
        
        with mock.patch.object(outer_loop, "_get_client", return_value=fake_client):
            next_state = outer_loop.submit_node(state)
            
        # Ensure successful submit cleanup contract is preserved.
        self.assertIsNone(next_state.get("current_task"))
        self.assertIsNone(next_state.get("task_result"))