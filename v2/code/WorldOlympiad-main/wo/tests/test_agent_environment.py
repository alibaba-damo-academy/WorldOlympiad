from wo.agent.run import load_agent_environment


def test_direct_cli_loads_existing_profile_without_overwriting_explicit_environment(tmp_path,monkeypatch):
    path=tmp_path/'runtime.env'
    path.write_text('OPENAI_API_KEY=fixture-key\nOPENAI_BASE_URL=https://fixture.invalid/v1\nWO_AGENT_MODEL=gpt-5.6-sol\n')
    monkeypatch.delenv('OPENAI_BASE_URL',raising=False)
    monkeypatch.setenv('OPENAI_API_KEY','explicit-fixture-key')
    monkeypatch.setenv('WO_AGENT_MODEL','gpt-5.6-sol')
    load_agent_environment(path)
    import os
    assert os.environ['OPENAI_API_KEY']=='explicit-fixture-key'
    assert os.environ['OPENAI_BASE_URL']=='https://fixture.invalid/v1'
    assert os.environ['WO_AGENT_MODEL']=='gpt-5.6-sol'


def test_missing_profile_does_not_change_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','explicit-fixture-key')
    load_agent_environment(tmp_path/'missing.env')
    import os
    assert os.environ['OPENAI_API_KEY']=='explicit-fixture-key'
