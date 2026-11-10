import tempfile
import unittest
import json
import threading
import httpx
import http.client
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from app import Handler, Harness, ensure_workspace


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.harness = Harness(self.root)

    def tearDown(self):
        self.harness.client.close()
        self.temp.cleanup()

    def test_create_file_returns_actual_unified_diff(self):
        result, change = self.harness.action(
            {"type": "write_file", "path": "src/hello.py", "content": "print('hello')\n"},
            "test-session",
        )
        self.assertEqual(result, "Created src/hello.py.")
        self.assertEqual(self.root.joinpath("src/hello.py").read_text(encoding="utf-8"), "print('hello')\n")
        self.assertTrue(change["created"])
        self.assertEqual(change["added"], 1)
        self.assertIn("+print('hello')", change["diff"])

    def test_agent_workspace_is_created_and_isolated_from_project_root(self):
        project_file = self.root / "project.py"
        project_file.write_text("outside workspace\n", encoding="utf-8")
        workspace = ensure_workspace(self.root)
        harness = Harness(workspace)
        self.addCleanup(harness.client.close)

        result, _ = harness.action(
            {"type": "write_file", "path": "inside.py", "content": "inside workspace\n"},
            "test-session",
        )

        self.assertEqual(result, "Created inside.py.")
        self.assertTrue((workspace / "inside.py").is_file())
        self.assertEqual(project_file.read_text(encoding="utf-8"), "outside workspace\n")
        self.assertEqual([item["path"] for item in harness.files()], ["inside.py"])
        with self.assertRaisesRegex(ValueError, "stay inside the project directory"):
            harness.project_path("../project.py")

    def test_existing_file_requires_read_and_tracks_changes(self):
        target = self.root / "hello.py"
        target.write_text("before\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Read the existing file"):
            self.harness.action(
                {"type": "write_file", "path": "hello.py", "content": "after\n"}, "test-session"
            )
        self.harness.action({"type": "read_file", "path": "hello.py"}, "test-session")
        result, change = self.harness.action(
            {"type": "write_file", "path": "hello.py", "content": "after\n"}, "test-session"
        )
        self.assertEqual(result, "Updated hello.py.")
        self.assertEqual(change["added"], 1)
        self.assertEqual(change["removed"], 1)
        self.assertIn("-before", change["diff"])
        self.assertIn("+after", change["diff"])

    def test_rejects_paths_outside_project_and_secrets(self):
        for path in ("../outside.txt", ".env", "credentials/token"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.harness.project_path(path)

    def test_lists_real_workspace_files_and_excludes_secrets(self):
        (self.root / "app.py").write_text("print('hello')", encoding="utf-8")
        (self.root / ".env").write_text("secret", encoding="utf-8")
        self.assertEqual([item["path"] for item in self.harness.files()], ["app.py"])

    def test_python_syntax_check_reports_actual_error(self):
        (self.root / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        with self.assertRaises(SyntaxError):
            self.harness.action({"type": "check_python", "path": "broken.py"}, "test-session")

    def test_empty_test_discovery_is_not_reported_as_passed(self):
        process_result = SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="Ran 0 tests in 0.000s\n\nOK",
        )
        with patch("app.subprocess.run", return_value=process_result):
            result, _ = self.harness.action({"type": "run_tests"}, "test-session")
        self.assertTrue(result.startswith("Tests not verified:"))

    def test_failed_model_tool_is_retried_before_completion_is_reported(self):
        responses = iter([
            {
                "message": "The file was created successfully.",
                "done": True,
                "actions": [{"type": "write_file"}],
            },
            {
                "message": "The missing arguments are corrected; I will write the file and verify it.",
                "done": True,
                "actions": [{
                    "type": "write_file", "path": "recovered.py",
                    "content": "print('recovered')\n",
                }],
            },
            {"message": "The file was written and the tests passed.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            answer = json.dumps(next(responses))
            yield SimpleNamespace(message=SimpleNamespace(content=answer))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(
                self.harness,
                "action",
                side_effect=[
                    ValueError("write_file requires string path and content fields."),
                    ("Created recovered.py.", {
                        "path": "recovered.py", "created": True, "added": 1,
                        "removed": 0, "diff": ["+print('recovered')"],
                    }),
                    ("Tests passed. 9 tests.", None),
                ],
            ):
                self.harness.respond("test-session", "Create a file", events.append, "qwen3.5:9b")

        self.assertNotIn(
            "The file was created successfully.",
            [event.get("text") for event in events],
        )
        self.assertTrue(any(event["type"] == "tool_completed" and not event["success"] for event in events))
        self.assertTrue(any(event["type"] == "status" and "recover" in event["text"] for event in events))
        self.assertIn("Tests passed. 9 tests.", [event.get("result") for event in events])
        self.assertEqual(events[-1]["type"], "complete")

    def test_invalid_model_response_is_retried_with_actionable_error(self):
        responses = iter([
            {"message": 123, "done": True, "actions": []},
            {"message": "Recovered response.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat) as chat:
            self.harness.respond("test-session", "Give a short answer.", events.append, "qwen3.5:9b")

        self.assertEqual(chat.call_count, 2)
        self.assertTrue(any(
            event["type"] == "status" and "message must be a string" in event["text"]
            for event in events
        ))
        self.assertIn("Recovered response.", [event.get("text") for event in events])
        self.assertEqual(events[-1]["type"], "complete")

    def test_model_cannot_finish_inspection_and_edit_task_after_listing_files(self):
        responses = iter([
            {
                "message": "Inspection and implementation are complete.",
                "done": True,
                "actions": [{"type": "list_files"}],
            },
            {
                "message": "I will inspect the existing code.",
                "done": False,
                "actions": [{"type": "read_file", "path": "app.py"}],
            },
            {
                "message": "The helper is implemented.",
                "done": True,
                "actions": [{
                    "type": "write_file", "path": "helper.py",
                    "content": "def helper():\n    return True\n",
                }],
            },
            {"message": "All tests passed.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))

        action_results = [
            ("Project files:\napp.py", None),
            ("Existing source", None),
            ("Created helper.py.", {
                "path": "helper.py", "created": True, "added": 2,
                "removed": 0, "diff": ["+def helper():", "+    return True"],
            }),
            ("Tests passed. 9 tests.", None),
        ]
        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(self.harness, "action", side_effect=action_results) as execute:
                self.harness.respond(
                    "test-session",
                    "Inspect this project and implement a helper.",
                    events.append,
                    "qwen3.5:9b",
                )

        self.assertNotIn(
            "Inspection and implementation are complete.",
            [event.get("text") for event in events],
        )
        self.assertEqual(
            [call.args[0]["type"] for call in execute.call_args_list],
            ["list_files", "read_file", "write_file", "run_tests"],
        )
        self.assertTrue(any(
            event["type"] == "status" and "not complete" in event["text"]
            for event in events
        ))
        self.assertEqual(events[-1]["type"], "complete")

    def test_explicit_run_tests_request_executes_tool_before_model_finishes(self):
        responses = iter([
            {
                "message": "No test command found in the project.",
                "done": True,
                "actions": [],
            },
            {"message": "Tests passed.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(self.harness, "action", return_value=("Tests passed. 5 tests.", None)):
                self.harness.respond(
                    "test-session", "Run the complete test suite.", events.append, "qwen3.5:9b"
                )

        self.assertTrue(any(
            event["type"] == "tool_started" and event["tool"] == "run_tests"
            and event.get("origin") == "user_request"
            for event in events
        ))
        self.assertNotIn(
            "No test command found in the project.",
            [event.get("text") for event in events],
        )
        self.assertIn("Tests passed. 5 tests.", [event.get("result") for event in events])
        self.assertEqual(events[-1]["type"], "complete")

    def test_explicit_python_checks_execute_even_if_model_skips_them(self):
        responses = iter([
            {"message": "Both files are valid.", "done": True, "actions": []},
            {"message": "Both Python files passed their syntax checks.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(
                self.harness,
                "action",
                side_effect=[
                    ("Python syntax OK: one.py", None),
                    ("Python syntax OK: two.py", None),
                ],
            ) as execute:
                self.harness.respond(
                    "test-session",
                    "Run check_python on one.py and two.py.",
                    events.append,
                    "qwen3.5:9b",
                )

        self.assertEqual(
            [call.args[0]["path"] for call in execute.call_args_list],
            ["one.py", "two.py"],
        )
        self.assertTrue(all(
            event.get("success", True)
            for event in events if event["type"] == "tool_completed"
        ))
        self.assertEqual(events[-1]["type"], "complete")

    def test_file_edits_trigger_real_test_verification(self):
        responses = iter([
            {
                "message": "The file was created.",
                "done": True,
                "actions": [{
                    "type": "write_file", "path": "new.py", "content": "print('hello')\n",
                }],
            },
            {"message": "Tests passed.", "done": True, "actions": []},
        ])

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(responses))))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(
                self.harness,
                "action",
                side_effect=[
                    ("Created new.py.", {"path": "new.py", "created": True, "added": 1,
                                         "removed": 0, "diff": ["+print('hello')"]}),
                    ("Tests passed. 7 tests.", None),
                ],
            ) as execute:
                self.harness.respond("test-session", "Create new.py.", events.append, "qwen3.5:9b")

        self.assertEqual(
            [call.args[0]["type"] for call in execute.call_args_list],
            ["write_file", "run_tests"],
        )
        self.assertTrue(any(
            event["type"] == "tool_started" and event["tool"] == "run_tests"
            and event.get("origin") == "verification"
            for event in events
        ))
        self.assertEqual(events[-1]["type"], "complete")

    def test_repeated_identical_tool_failure_stops_task(self):
        response = {
            "message": "",
            "done": False,
            "actions": [{"type": "read_file", "path": "../outside.py"}],
        }

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat) as chat:
            self.harness.respond("test-session", "Inspect this project.", events.append, "qwen3.5:9b")

        self.assertEqual(chat.call_count, 2)
        self.assertTrue(any(
            event.get("text", "").startswith("Stopped after the same action failed")
            for event in events
        ))
        self.assertEqual(events[-1]["type"], "complete")

    def test_consecutive_empty_turns_stop_task(self):
        response = {"message": "I will continue.", "done": False, "actions": []}

        def fake_chat(**kwargs):
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat) as chat:
            self.harness.respond("test-session", "Continue.", events.append, "qwen3.5:9b")

        self.assertEqual(chat.call_count, 3)
        self.assertTrue(any(
            event.get("text", "").startswith("Stopped after 3 consecutive turns")
            for event in events
        ))
        self.assertEqual(events[-1]["type"], "complete")

    def test_cancelled_task_stops_before_running_model_action(self):
        cancel_event = threading.Event()
        response = {
            "message": "",
            "done": False,
            "actions": [{"type": "write_file", "path": "should_not_exist.py", "content": ""}],
        }

        def fake_chat(**kwargs):
            cancel_event.set()
            yield SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            with patch.object(self.harness, "action") as execute:
                self.harness.respond(
                    "test-session", "Create a file.", events.append, "qwen3.5:9b", cancel_event,
                )

        execute.assert_not_called()
        self.assertIn("Task cancelled.", [event.get("text") for event in events])
        self.assertEqual(events[-1]["type"], "complete")

    def test_cancelled_task_reports_cancellation_when_stream_times_out(self):
        cancel_event = threading.Event()

        def fake_chat(**kwargs):
            cancel_event.set()
            raise httpx.ReadTimeout("No response data received.")

        events = []
        with patch.object(self.harness.client, "chat", side_effect=fake_chat):
            self.harness.respond(
                "test-session", "Do work.", events.append, "qwen3.5:9b", cancel_event,
            )

        self.assertIn("Task cancelled.", [event.get("text") for event in events])
        self.assertEqual(events[-1]["type"], "complete")

    def test_task_deadline_stops_before_calling_model(self):
        self.harness.max_task_seconds = 0
        events = []
        with patch.object(self.harness.client, "chat") as chat:
            self.harness.respond("test-session", "Do work.", events.append, "qwen3.5:9b")

        chat.assert_not_called()
        self.assertIn(
            "Task stopped after reaching its time limit.",
            [event.get("text") for event in events],
        )
        self.assertEqual(events[-1]["type"], "complete")

    def test_running_task_can_be_cancelled_and_releases_session(self):
        cancel_event = self.harness.register_task("test-session")
        with self.assertRaisesRegex(ValueError, "already running"):
            self.harness.register_task("test-session")

        self.assertTrue(self.harness.cancel_task("test-session"))
        self.assertTrue(cancel_event.is_set())
        self.harness.finish_task("test-session", cancel_event)
        self.assertFalse(self.harness.cancel_task("test-session"))

    def test_cancel_api_sets_the_active_task_event(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.harness = self.harness
        cancel_event = self.harness.register_task("test-session")
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request(
                "POST",
                "/api/cancel",
                body=json.dumps({"session_id": "test-session"}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            result = json.loads(response.read())

            self.assertEqual(response.status, 200)
            self.assertTrue(result["cancelled"])
            self.assertTrue(cancel_event.is_set())
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            server_thread.join()
            self.harness.finish_task("test-session", cancel_event)


if __name__ == "__main__":
    unittest.main()
