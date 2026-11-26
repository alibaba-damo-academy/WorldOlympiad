import io
import msgpack
import numpy as np
import pytest
from wo.models.gr00t_client import encode, decode, N16Client


def test_n16_wire_is_upstream_npy_not_msgpack_numpy():
    array = np.arange(21, dtype=np.float32).reshape(1, 3, 7)
    raw = msgpack.unpackb(encode({'state.joint_position': array}), raw=False)
    value = raw['state.joint_position']
    assert value['__ndarray_class__'] is True
    np.testing.assert_array_equal(np.load(io.BytesIO(value['as_npy']), allow_pickle=False), array)
    np.testing.assert_array_equal(decode(encode({'x': array}))['x'], array)
    with pytest.raises(TypeError, match='object dtype'):
        encode(np.array([object()], dtype=object))


def test_request_envelope_and_response_without_server():
    class Socket:
        def send(self, data):
            request = decode(data)
            assert request['endpoint'] == 'get_action'
            assert request['data']['options'] is None
            np.testing.assert_array_equal(request['data']['observation']['x'], [1, 2])
        def recv(self):
            return encode([{'action.joint_position': np.zeros((1, 32, 7))}, {}])
    client = object.__new__(N16Client)
    client.socket = Socket()
    action, info = client.get_action({'x': np.array([1, 2])})
    assert action['action.joint_position'].shape == (1, 32, 7)
    assert info == {}
