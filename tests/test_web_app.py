import copy
import hashlib
import io
import json

import cv2
import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from facade_mcp.web_app import create_app, editable_report, extract_frames, encode


@pytest.fixture
def workspace(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        yield client,app.state.repo


def draft():
    return {'title':'A栋报告','summary_html':'原始概述','project':{'building_name':'A栋'},'defects':[{'id':'d1','number':1,'type':'T','facade':'待确认','description_html':'锈渍','advice_html':'现场复核','area_pixels':25,'evidence':{},'source_type':'T'}]}


def seed(client,repo):
    task=client.post('/api/tasks').json()
    repo.save(task['id'],draft(),0,'生成报告')
    return '/api/tasks/'+task['id'],task['id']


def test_versions_conflict_restore_and_integrity(workspace):
    c,repo=workspace;base,tid=seed(c,repo)
    r=c.put(base+'/report',json={'report':{'title':'人工修订'},'base_revision':1})
    assert r.status_code==200 and r.json()['revision']==2
    assert c.put(base+'/report',json={'report':{'title':'过时覆盖'},'base_revision':1}).status_code==409
    assert c.post(base+'/versions/1',json={'base_revision':2}).json()['revision']==3
    assert c.get(base).json()['report']['title']=='A栋报告'
    assert c.get(base+'/versions/2').json()['report']['title']=='人工修订'
    versions=c.get(base+'/versions').json()
    assert [r['version'] for r in versions]==[3,2,1]
    previous=''
    for row in reversed(versions):
        doc=c.get(base+'/versions/'+str(row['version'])).json()['report']
        assert row['parent_digest']==previous
        assert row['digest']==hashlib.sha256(encode([tid,row['version'],row['created'],row['reason'],previous,doc]).encode()).hexdigest()
        previous=row['digest']


def test_report_protects_evidence_and_sanitizes_markup():
    original=draft();original['defects'][0]['evidence']={'original':'/immutable.png'}
    patch=copy.deepcopy(original);row=patch['defects'][0]
    row.update(evidence={'original':'https://evil.test'},area_pixels=900,source_type='C',type='E',advice_html='<b onclick="bad()">复核</b><img src=x onerror=bad()>')
    patch['defects'].append({'id':'new','type':'C'})
    result=editable_report(original,patch)
    assert len(result['defects'])==1
    result=result['defects'][0]
    assert result['evidence']=={'original':'/immutable.png'} and result['area_pixels']==25 and result['source_type']=='T'
    assert result['type']=='E' and result['advice_html']=='<b>复核</b>'


def test_unknown_defect_can_be_edited_without_reclassifying():
    doc=draft();doc['defects'][0]['type']='UNKNOWN'
    patch=copy.deepcopy(doc);patch['title']='只改标题'
    assert editable_report(doc,patch)['defects'][0]['type']=='UNKNOWN'


def test_upload_isolation_and_input_validation(workspace):
    c,repo=workspace;task=c.post('/api/tasks').json();base='/api/tasks/'+task['id']
    stream=io.BytesIO();Image.new('RGB',(8,8),'white').save(stream,'PNG')
    r=c.post(base+'/uploads',content=stream.getvalue(),headers={'X-File-Name':'../../example.png'})
    assert r.status_code==201
    asset=r.json()['assets'][0]
    assert asset['name']=='example.png' and 'path' not in asset
    assert c.get(base+'/assets/'+asset['id']).content==stream.getvalue()
    other=c.post('/api/tasks').json()
    assert c.get('/api/tasks/'+other['id']+'/assets/'+asset['id']).status_code==404
    assert c.post(base+'/uploads',content=b'not a png',headers={'X-File-Name':'bad.png'}).status_code==400
    assert c.post(base+'/uploads',content=b'<script/>',headers={'X-File-Name':'bad.html'}).status_code==400
    assert c.post(base+'/uploads',content=b'',headers={'X-File-Name':'bad.txt'}).status_code==400
    assert len(c.get(base).json()['assets'])==1
    assert c.post('/api/tasks',headers={'Origin':'https://evil.test'}).status_code==403
    assert c.put(base+'/settings',json={'max_frames':'bad'}).status_code==400
    assert c.put(base+'/settings',json={'prompt_mode':'text','segmenter':'mobile_sam'}).status_code==400


def test_video_sampling_keeps_source_time(tmp_path):
    path=tmp_path/'sample.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,(32,32))
    assert writer.isOpened()
    for frame in range(20):writer.write(np.full((32,32,3),frame*10,dtype=np.uint8))
    writer.release()
    result=extract_frames(path,tmp_path/'frames',{'frame_interval':1,'max_frames':2})
    assert [r['timestamp_seconds'] for r in result]==[0,2]
    assert all(r['source_video']=='sample.avi' for r in result)


def test_exports_escape_and_preserve_source(workspace):
    c,repo=workspace;base,tid=seed(c,repo)
    doc=draft();doc['title']='<script>alert(1)</script>'
    repo.save(tid,doc,1,'测试')
    response=c.get(base+'/export')
    assert response.status_code==200
    assert '<script>' not in response.text and '&lt;script&gt;' in response.text
    assert 'attachment' in response.headers['content-disposition']
    assert c.get(base+'/export?format=json').json()==doc


def test_restart_marks_interrupted_job(tmp_path):
    app=create_app(tmp_path);repo=app.state.repo;task=repo.create();repo.claim(task['id'])
    restarted=create_app(tmp_path)
    assert restarted.state.repo.get(task['id'])['status']=='failed'


def wait_job(client,base):
    import time
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        task=client.get(base).json()
        if task['status']!='running': return task
        time.sleep(.02)
    pytest.fail('后台任务未按时结束')


def test_prompt_edit_and_failed_instruction_preserve_report(workspace):
    c,repo=workspace;base,tid=seed(c,repo)
    assert c.post(base+'/chat',json={'prompt':'第1条改为白华','base_revision':1}).status_code==202
    task=wait_job(c,base)
    assert task['status']=='ready' and task['revision']==2
    assert task['report']['defects'][0]['type']=='E'
    assert task['report']['defects'][0]['source_type']=='T'
    c.post(base+'/chat',json={'prompt':'随意润色整份报告','base_revision':2})
    failed=wait_job(c,base)
    assert failed['status']=='failed' and failed['revision']==2
    assert failed['report']==task['report']


def test_online_missing_configuration_never_discards_report(workspace,monkeypatch):
    c,repo=workspace;base,tid=seed(c,repo)
    for name in ['FACADE_LLM_MODEL','FACADE_LLM_API_KEY','OPENAI_API_KEY']:monkeypatch.delenv(name,raising=False)
    c.put(base+'/settings',json={'online_assistant':True})
    c.post(base+'/chat',json={'prompt':'精简概述','base_revision':1})
    task=wait_job(c,base)
    assert task['status']=='failed' and task['revision']==1
    assert '未配置' in task['error']


def test_busy_task_rejects_edits_and_duplicate_runs(workspace):
    c,repo=workspace;base,tid=seed(c,repo);repo.claim(tid)
    assert c.put(base+'/report',json={'report':{'title':'覆盖'},'base_revision':1}).status_code==409
    assert c.put(base+'/settings',json={'building_name':'覆盖'}).status_code==409
    assert c.post(base+'/chat',json={'prompt':'标题改为测试'}).status_code==409
    assert c.post(base+'/versions/1',json={'base_revision':1}).status_code==409
