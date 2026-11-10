import tempfile
import unittest
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from app import Harness


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.harness = Harness(self.root)

    def tearDown(self):
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

    def test_failed_model_tool_is_retried_before_completion_is_reported(self):
        responses = iter([
            {
                "message": "The file was created successfully.",
                "done": True,
                "actions": [{"type": "write_file"}],
            },
            {
                "message": "I couldn't write the file because the required path and content were missing.",
                "done": True,
                "actions": [],
            },
        ])

        def fake_chat(**kwargs):
            answer = json.dumps(next(responses))
            yield SimpleNamespace(message=SimpleNamespace(content=answer))

        events = []
        with patch("app.ollama.chat", side_effect=fake_chat):
            self.harness.respond("test-session", "Create a file", events.append, "qwen3.5:9b")

        self.assertNotIn(
            "The file was created successfully.",
            [event.get("text") for event in events],
        )
        self.assertTrue(any(event["type"] == "tool_completed" and not event["success"] for event in events))
        self.assertTrue(any(event["type"] == "status" and "recover" in event["text"] for event in events))
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
        with patch("app.ollama.chat", side_effect=fake_chat):
            with patch.object(self.harness, "action", return_value=("Tests passed. 5 tests.", None)):
                self.harness.respond(
                    "test-session", "Use run_tests to run tests.", events.append, "qwen3.5:9b"
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
        with patch("app.ollama.chat", side_effect=fake_chat):
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
        with patch("app.ollama.chat", side_effect=fake_chat):
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


if __name__ == "__main__":
    unittest.main()
