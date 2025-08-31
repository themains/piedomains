"""
Test configuration management.
"""

import unittest
from unittest.mock import patch
import os
from piedomains.config import Config, get_config, set_config, configure


class TestConfig(unittest.TestCase):
    """Test configuration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original config
        self.original_config = get_config()

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original config
        set_config(self.original_config)

    def test_default_config_values(self):
        """Test default configuration values."""
        config = Config()

        self.assertEqual(config.get('http_timeout'), 10)
        self.assertEqual(config.get('webdriver_timeout'), 30)
        self.assertEqual(config.get('page_load_timeout'), 30)
        self.assertEqual(config.get('max_retries'), 3)
        self.assertEqual(config.get('retry_delay'), 1)
        self.assertEqual(config.get('screenshot_wait_time'), 5)
        self.assertEqual(config.get('batch_size'), 50)
        self.assertEqual(config.get('parallel_workers'), 4)

    def test_config_properties(self):
        """Test configuration property access."""
        config = Config()

        self.assertEqual(config.http_timeout, 10)
        self.assertEqual(config.webdriver_timeout, 30)
        self.assertEqual(config.page_load_timeout, 30)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 1)
        self.assertEqual(config.screenshot_wait_time, 5)
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.parallel_workers, 4)

    def test_config_override(self):
        """Test configuration override with dictionary."""
        override_config = {
            'http_timeout': 20,
            'batch_size': 100,
            'custom_setting': 'test_value'
        }

        config = Config(override_config)

        self.assertEqual(config.get('http_timeout'), 20)
        self.assertEqual(config.get('batch_size'), 100)
        self.assertEqual(config.get('custom_setting'), 'test_value')
        # Default values should still be present
        self.assertEqual(config.get('webdriver_timeout'), 30)

    def test_config_set_and_get(self):
        """Test setting and getting configuration values."""
        config = Config()

        config.set('test_key', 'test_value')
        self.assertEqual(config.get('test_key'), 'test_value')

        # Test with default value
        self.assertEqual(config.get('nonexistent_key', 'default'), 'default')

    def test_config_update(self):
        """Test updating configuration with multiple values."""
        config = Config()

        updates = {
            'http_timeout': 15,
            'new_setting': 'new_value'
        }

        config.update(updates)

        self.assertEqual(config.get('http_timeout'), 15)
        self.assertEqual(config.get('new_setting'), 'new_value')

    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config({'test_key': 'test_value'})
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['test_key'], 'test_value')
        self.assertIn('http_timeout', config_dict)

    @patch.dict(os.environ, {
        'PIEDOMAINS_HTTP_TIMEOUT': '25',
        'PIEDOMAINS_MAX_RETRIES': '5',
        'PIEDOMAINS_BATCH_SIZE': '75',
        'PIEDOMAINS_USER_AGENT': 'Custom Agent'
    })
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        config = Config()

        self.assertEqual(config.get('http_timeout'), 25)
        self.assertEqual(config.get('max_retries'), 5)
        self.assertEqual(config.get('batch_size'), 75)
        self.assertEqual(config.get('user_agent'), 'Custom Agent')

    @patch.dict(os.environ, {
        'PIEDOMAINS_HTTP_TIMEOUT': 'invalid_number'
    })
    def test_invalid_environment_variable(self):
        """Test handling of invalid environment variable values."""
        with patch('builtins.print') as mock_print:
            config = Config()
            # Should fall back to default value
            self.assertEqual(config.get('http_timeout'), 10)
            # Should print warning
            mock_print.assert_called()

    def test_global_config_functions(self):
        """Test global configuration functions."""
        # Test get_config returns same instance
        config1 = get_config()
        config2 = get_config()
        self.assertIs(config1, config2)

        # Test set_config
        new_config = Config({'test_global': 'test_value'})
        set_config(new_config)
        retrieved_config = get_config()
        self.assertIs(retrieved_config, new_config)
        self.assertEqual(retrieved_config.get('test_global'), 'test_value')

        # Test configure function
        configure(test_configure='configure_value', http_timeout=99)
        self.assertEqual(get_config().get('test_configure'), 'configure_value')
        self.assertEqual(get_config().get('http_timeout'), 99)

    def test_image_size_property(self):
        """Test image size property returns tuple."""
        config = Config()
        image_size = config.image_size

        self.assertIsInstance(image_size, tuple)
        self.assertEqual(len(image_size), 2)
        self.assertEqual(image_size, (254, 254))

    def test_webdriver_window_size_property(self):
        """Test webdriver window size property."""
        config = Config()
        window_size = config.webdriver_window_size

        self.assertIsInstance(window_size, str)
        self.assertEqual(window_size, "1280,1024")

    def test_user_agent_property(self):
        """Test user agent property."""
        config = Config()
        user_agent = config.user_agent

        self.assertIsInstance(user_agent, str)
        self.assertIn("Mozilla", user_agent)


if __name__ == "__main__":
    unittest.main()

