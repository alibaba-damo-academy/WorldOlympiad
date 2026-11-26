import json,sqlite3
import pytest
from wo.agent import tools as T,jobs

@pytest.mark.parametrize('status',['queued','running'])
@pytest.mark.parametrize('layout',[None,'droid','scenecode'])
def test_reference_output_is_unique_per_run_case_even_when_optional_args_differ(tmp_path,monkeypatch,status,layout):
 con=sqlite3.connect(':memory:',isolation_level=None);con.row_factory=sqlite3.Row;con.executescript(jobs.SCHEMA)
 jid=con.execute('insert into jobs(kind,args,status,created,timeout_s) values(?,?,?,0,100)',('reference',json.dumps({'run':'nine','case':'mallet'}),status)).lastrowid
 monkeypatch.setattr(T,'run_dir',lambda run:tmp_path)
 monkeypatch.setattr(T.J,'connect',lambda:con)
 monkeypatch.setattr(T.J,'submit',lambda *a,**kw:pytest.fail('duplicate reference entered queue'))
 args={'run':'nine','case':'mallet'}
 if layout is not None:args['layout']=layout
 assert T.submit('reference',json.dumps(args))==f'reused {status} job #{jid} (reference)'
 assert con.execute('select count(*) from jobs').fetchone()[0]==1
 con.close()

@pytest.mark.parametrize('different', ['run','case'])
def test_distinct_reference_output_is_not_deduplicated(tmp_path,monkeypatch,different):
 con=sqlite3.connect(':memory:',isolation_level=None);con.row_factory=sqlite3.Row;con.executescript(jobs.SCHEMA)
 con.execute('insert into jobs(kind,args,status,created,timeout_s) values(?,?,?,0,100)',('reference',json.dumps({'run':'nine','case':'mallet'}),'running'))
 monkeypatch.setattr(T,'run_dir',lambda run:tmp_path)
 monkeypatch.setattr(T.J,'connect',lambda:con)
 calls=[]
 monkeypatch.setattr(T.J,'submit',lambda *a,**kw:calls.append((a,kw)) or 44)
 args={'run':'nine','case':'mallet'};args[different]='different'
 assert T.submit('reference',json.dumps(args))=='queued job #44 (reference)'
 assert len(calls)==1
 con.close()
