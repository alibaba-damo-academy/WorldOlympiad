from wo.agent.stages import STAGES


def test_score_stage_skips_native_scorer_for_not_applicable_case():
    command = STAGES['score'].build({'run': 'bench_v1', 'case': 'case_a', 'model': 'cosmos3'}, 1)
    text = ' '.join(command)
    assert 'worldolympiad_case.json' in text
    assert 'then' in text and 'fi' in text
