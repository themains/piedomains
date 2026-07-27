#!/usr/bin/env python

"""Tests for structured JSON logging and bound context."""

import json
import logging
import unittest
from io import StringIO

from piedomains import piedomains_logging as plog


class TestJsonFormatter(unittest.TestCase):
    """JSON output carries message plus any extra= fields."""

    def _emit(self, **extra):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(plog.JsonFormatter())
        handler.addFilter(plog._ContextFilter())
        logger = logging.getLogger("piedomains.test.json")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.error("navigation timed out", extra=extra)
        return json.loads(stream.getvalue().strip())

    def test_core_fields_present(self):
        record = self._emit()
        self.assertEqual(record["level"], "ERROR")
        self.assertEqual(record["msg"], "navigation timed out")
        self.assertIn("ts", record)
        self.assertIn("logger", record)

    def test_extra_fields_are_promoted_to_top_level(self):
        record = self._emit(domain="foo.com", stage="fetch", error_code="timeout")
        self.assertEqual(record["domain"], "foo.com")
        self.assertEqual(record["stage"], "fetch")
        self.assertEqual(record["error_code"], "timeout")

    def test_output_is_one_line_per_record(self):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(plog.JsonFormatter())
        logger = logging.getLogger("piedomains.test.json.lines")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.info("one")
        logger.info("two")
        lines = [x for x in stream.getvalue().splitlines() if x.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            json.loads(line)

    def test_bound_context_appears_on_every_record(self):
        plog.bind_context(run_id="abc123")
        try:
            record = self._emit(domain="foo.com")
            self.assertEqual(record["run_id"], "abc123")
        finally:
            plog.clear_context()

    def test_clear_context_removes_bound_fields(self):
        plog.bind_context(run_id="abc123")
        plog.clear_context()
        self.assertNotIn("run_id", self._emit())

    def test_binding_none_unbinds_a_single_field(self):
        plog.bind_context(run_id="abc123")
        plog.bind_context(run_id=None)
        try:
            self.assertNotIn("run_id", self._emit())
        finally:
            plog.clear_context()

    def test_exception_info_is_captured(self):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(plog.JsonFormatter())
        logger = logging.getLogger("piedomains.test.json.exc")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
        record = json.loads(stream.getvalue().strip())
        self.assertEqual(record["exc_type"], "ValueError")
        self.assertIn("boom", record["exc"])


class TestConfigureLoggingFormat(unittest.TestCase):
    """The json format is selectable and text remains the default."""

    def test_json_format_is_accepted(self):
        plog.configure_logging(console_format="json", force_reconfigure=True)
        logger = logging.getLogger("piedomains")
        formatters = [h.formatter for h in logger.handlers]
        self.assertTrue(any(isinstance(f, plog.JsonFormatter) for f in formatters))

    def test_default_format_stays_text(self):
        plog.configure_logging(force_reconfigure=True)
        logger = logging.getLogger("piedomains")
        formatters = [h.formatter for h in logger.handlers]
        self.assertFalse(any(isinstance(f, plog.JsonFormatter) for f in formatters))

    def test_env_var_selects_json(self):
        import os

        os.environ["PIEDOMAINS_LOG_FORMAT"] = "json"
        try:
            plog.configure_logging(force_reconfigure=True)
            logger = logging.getLogger("piedomains")
            formatters = [h.formatter for h in logger.handlers]
            self.assertTrue(any(isinstance(f, plog.JsonFormatter) for f in formatters))
        finally:
            del os.environ["PIEDOMAINS_LOG_FORMAT"]
            plog.configure_logging(force_reconfigure=True)

    def test_invalid_format_still_raises(self):
        with self.assertRaises(ValueError):
            plog.configure_logging(console_format="nonsense", force_reconfigure=True)


if __name__ == "__main__":
    unittest.main()
