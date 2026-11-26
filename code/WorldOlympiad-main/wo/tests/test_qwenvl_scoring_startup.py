import io
import json
from scripts import score_robotics_cases


class _LiveService:
    def poll(self):
        return None


class _HealthResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_qwenvl_startup_waits_for_matching_live_health_without_generation_warmup(monkeypatch, tmp_path):
    elapsed = [0.0]
    commands = []
    service = _LiveService()
    monkeypatch.setattr(score_robotics_cases.subprocess, 'Popen',
                        lambda command, **kwargs: (commands.append(command), service)[1])
    monkeypatch.setattr(score_robotics_cases.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(score_robotics_cases.time, 'sleep', lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    weights = tmp_path / 'weights'

    def health(url, timeout):
        assert url == 'http://127.0.0.1:8008/health'
        assert timeout == 2
        model = str(weights / 'QwenVL') if elapsed[0] >= 302 else 'another-model'
        return _HealthResponse(json.dumps({'status': 'ok', 'model': model}).encode())

    monkeypatch.setattr(score_robotics_cases.urllib.request, 'urlopen', health)
    started, log = score_robotics_cases._start_qwenvl(
        tmp_path, tmp_path, weights, tmp_path / 'python', '2', {}, tmp_path / 'qwenvl.log')
    try:
        assert started is service
        assert elapsed[0] >= 302
        assert elapsed[0] < score_robotics_cases.QWENVL_STARTUP_TIMEOUT_S
        assert '--warmup' not in commands[0]
    finally:
        log.close()


def test_qwenvl_startup_budget_exceeds_prior_five_minute_cutoff():
    assert score_robotics_cases.QWENVL_STARTUP_TIMEOUT_S > 300
