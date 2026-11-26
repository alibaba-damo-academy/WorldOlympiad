"""Lightweight N1.6 wire client (NOT N1.7's msgpack_numpy protocol).

Matches gr00t.policy.server_client on branch n1d6, without importing torch or
transformers into the native simulator. Numpy payloads are non-pickled .npy.
"""
import io
import numpy as np
import msgpack


def encode(data):
    def default(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('object dtype is forbidden on the policy wire')
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            return {'__ndarray_class__': True, 'as_npy': stream.getvalue()}
        raise TypeError(f'unsupported wire type: {type(value)}')
    return msgpack.packb(data, default=default)


def decode(data):
    def hook(value):
        if value.get('__ndarray_class__'):
            return np.load(io.BytesIO(value['as_npy']), allow_pickle=False)
        return value
    return msgpack.unpackb(data, object_hook=hook, raw=False)


class N16Client:
    def __init__(self, host='localhost', port=5555, timeout_ms=120000):
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f'tcp://{host}:{port}')

    def call(self, endpoint, data=None):
        request = {'endpoint': endpoint}
        if data is not None:
            request['data'] = data
        self.socket.send(encode(request))
        message = self.socket.recv()
        if message == b'ERROR':
            raise RuntimeError('GR00T N1.6 server error')
        response = decode(message)
        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def get_action(self, observation):
        return tuple(self.call('get_action', {'observation': observation, 'options': None}))

    def close(self):
        self.socket.close()
        self.context.term()
