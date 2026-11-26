import socket
import unittest

from unittest.mock import MagicMock, patch

from scenecode.utils.network_utils import find_available_port, is_port_available


class TestNetworkUtils(unittest.TestCase):
    """Test network utility functions."""

    @patch("socket.socket")
    def test_is_port_available_success(self, mock_socket_class):
        """Test is_port_available when port is available."""
        mock_socket = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket
        mock_socket.bind.return_value = None

        result = is_port_available("localhost", 8080)

        self.assertTrue(result)
        mock_socket.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mock_socket.bind.assert_called_once_with(("localhost", 8080))

    @patch("socket.socket")
    def test_is_port_available_occupied(self, mock_socket_class):
        """Test is_port_available when port is occupied."""
        mock_socket = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket
        mock_socket.bind.side_effect = OSError("Address already in use")

        result = is_port_available("localhost", 8080)

        self.assertFalse(result)
        mock_socket.bind.assert_called_once_with(("localhost", 8080))

    @patch("socket.socket")
    def test_is_port_available_other_os_error(self, mock_socket_class):
        """Test is_port_available with other OSError."""
        mock_socket = MagicMock()
        mock_socket_class.return_value.__enter__.return_value = mock_socket
        mock_socket.bind.side_effect = OSError("Permission denied")

        result = is_port_available("localhost", 80)

        self.assertFalse(result)

    def test_is_port_available_socket_creation(self):
        """Test that is_port_available creates socket with correct parameters."""
        with patch("socket.socket") as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket_class.return_value.__enter__.return_value = mock_socket

            is_port_available("127.0.0.1", 9000)

            mock_socket_class.assert_called_once_with(
                socket.AF_INET, socket.SOCK_STREAM
            )
            mock_socket.setsockopt.assert_called_once_with(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )

    @patch("scenecode.utils.network_utils.random.shuffle")
    @patch("scenecode.utils.network_utils.is_port_available")
    def test_find_available_port_first_available(
        self, mock_is_port_available, mock_shuffle
    ):
        """Test find_available_port when first port in range is available."""
        mock_shuffle.return_value = None
        mock_is_port_available.return_value = True

        result = find_available_port("localhost", (8000, 8002))

        self.assertEqual(result, 8000)
        mock_is_port_available.assert_called_once_with("localhost", 8000)

    @patch("scenecode.utils.network_utils.random.shuffle")
    @patch("scenecode.utils.network_utils.is_port_available")
    def test_find_available_port_second_available(
        self, mock_is_port_available, mock_shuffle
    ):
        """Test find_available_port when second port in range is available."""
        mock_shuffle.return_value = None
        mock_is_port_available.side_effect = [False, True]

        result = find_available_port("localhost", (8000, 8002))

        self.assertEqual(result, 8001)
        self.assertEqual(mock_is_port_available.call_count, 2)
        mock_is_port_available.assert_any_call("localhost", 8000)
        mock_is_port_available.assert_any_call("localhost", 8001)

    @patch("scenecode.utils.network_utils.is_port_available")
    def test_find_available_port_none_available(self, mock_is_port_available):
        """Test find_available_port when no ports are available."""
        mock_is_port_available.return_value = False

        result = find_available_port("localhost", (8000, 8002))

        self.assertIsNone(result)
        self.assertEqual(mock_is_port_available.call_count, 3)  # 8000, 8001, 8002

    @patch("scenecode.utils.network_utils.is_port_available")
    def test_find_available_port_single_port_range(self, mock_is_port_available):
        """Test find_available_port with single port range."""
        mock_is_port_available.return_value = True

        result = find_available_port("127.0.0.1", (9000, 9000))

        self.assertEqual(result, 9000)
        mock_is_port_available.assert_called_once_with("127.0.0.1", 9000)

    @patch("scenecode.utils.network_utils.random.shuffle")
    @patch("scenecode.utils.network_utils.is_port_available")
    def test_find_available_port_range_order(
        self, mock_is_port_available, mock_shuffle
    ):
        """Test find_available_port checks ports in ascending order."""
        mock_shuffle.return_value = None
        call_order = []

        def track_calls(host, port):
            call_order.append(port)
            return port == 8002  # Only port 8002 is available

        mock_is_port_available.side_effect = track_calls

        result = find_available_port("localhost", (8000, 8003))

        self.assertEqual(result, 8002)
        self.assertEqual(call_order, [8000, 8001, 8002])

    def test_find_available_port_invalid_range(self):
        """Test find_available_port with invalid range (start > end)."""
        with patch("scenecode.utils.network_utils.is_port_available") as mock_check:
            result = find_available_port("localhost", (8002, 8000))

            self.assertIsNone(result)
            mock_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
