import unittest
from unittest import mock
from agent.nodes import outer_loop
from agent.state import OuterState

class OuterLoopResilienceTests(unittest.TestCase):
    def test_submit_failure_retains_state_for_retry(self):
        """Kịch bản 1: Nộp bài thất bại thì không được xóa current_task."""
        fake_client = mock.Mock()
        # Giả lập lần nộp đầu tiên thất bại
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
            
        # Kiểm tra xem state trả về có phải là dict rỗng (nghĩa là không ghi đè xóa current_task) hay không
        self.assertEqual(next_state, {})
        self.assertEqual(fake_client.submit_task_result.call_count, 1)

    def test_submit_success_clears_state(self):
        """Nộp bài thành công thì phải dọn dẹp current_task."""
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
            
        # Kiểm tra xem state đã bị xóa đúng chuẩn chưa
        self.assertIsNone(next_state.get("current_task"))
        self.assertIsNone(next_state.get("task_result"))