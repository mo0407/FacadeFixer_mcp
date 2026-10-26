import asyncio
import copy
from pathlib import Path
import pytest
from PIL import Image
from starlette.testclient import TestClient
from facade_mcp.web_app import create_app, Repository, clean_markup, LABELS
from facade_mcp.web_corrections import Corrections,evidence_geometry
from facade_mcp import web_report

@pytest.fixture
def context(tmp_path):
    app=create_app(tmp_path);repo=app.state.repo;task=repo.create();tid=task['id']
    source=tmp_path/'original.png';Image.new('RGB',(100,80),'gray').save(source)
    repo.update(tid,assets=[{'id':'original','kind':'evidence','path':str(source),'name':'source.png'}])
    row={'id':'d1','number':1,'type':'T','facade':'待确认','source_type':'T','source_instance':'1','bbox_xyxy':[10,10,50,50],'width':100,'height':80,'evidence':{'original':f'/api/tasks/{tid}/assets/original'},'area_pixels':20,'description_html':'待复核','advice_html':'待复核'}
    doc=web_report.normalize({'title':'测试报告','summary_html':'工程概况','project':{'building_name':'测试楼'},'defects':[row]})
    repo.save(tid,doc,0,'原始报告')
    def register(task_id,artifact):return '/api/tasks/'+task_id+'/assets/'+Path(artifact['path']).name
    manager=Corrections(repo,None,register)
    with TestClient(app) as client:yield client,repo,manager,tid,row

def data(row,**kwargs):return {'defect_id':row['id'],'base_revision':1,'bbox_xyxy':[12,12,55,55],**kwargs}

def test_queue_validation_task_isolation_and_history(context):
    c,repo,manager,tid,row=context;base='/api/tasks/'+tid+'/corrections'
    assert c.post(base,json=data(row,bbox_xyxy=[-1,0,100,80])).status_code==400
    assert c.post(base,json=data(row,positive_points=[[100,20]])).status_code==400
    assert c.post(base,json=data(row,positive_points=[[True,20]])).status_code==400
    assert c.post(base,json=data(row,bbox_xyxy=None,note='只写备注')).status_code==400
    assert c.post(base,json=data(row,base_revision=0)).status_code==409
    assert c.post(base,json=data(row)).status_code==200
    assert c.post(base,json=data(row,bbox_xyxy=[15,15,60,60])).status_code==200
    rows=manager.list(tid);assert [r['status'] for r in rows]==['superseded','pending']
    other=repo.create()
    assert c.post('/api/tasks/'+other['id']+'/corrections/'+rows[-1]['id']+'/cancel').status_code==409
    assert c.post(base+'/'+rows[-1]['id']+'/cancel').status_code==200
    assert manager.list(tid)[-1]['status']=='cancelled'

def test_batch_calls_bbox_then_points_and_preserves_old_version(context,tmp_path):
    c,repo,manager,tid,row=context
    manager.queue(repo.get(tid),data(row,positive_points=[[20,20]],negative_points=[[50,50]]))
    before=repo.get(tid)['report'];calls=[]
    mask=tmp_path/'mask.png';Image.new('L',(100,80),255).save(mask)
    overlay=tmp_path/'overlay.png';Image.new('RGB',(100,80),'blue').save(overlay)
    async def call(name,args):
        calls.append((name,args))
        return {'instances':[{'instance_id':'1','bbox_xyxy':[12,12,55,55],'area_pixels':80,'mask':{'path':str(mask)},'overlay':{'path':str(overlay)}}],'annotation_manifest_path':str(tmp_path/'manifest.json'),'annotation_manifest':{'path':str(tmp_path/'manifest.json')}}
    asyncio.run(manager.apply(repo.get(tid),call))
    assert [c[0] for c in calls]==['facade_models_segment','facade_models_refine']
    assert calls[0][1]['defects'][0]['bbox_xyxy']==[12,12,55,55]
    assert calls[1][1]['positive_points']==[[20,20]]
    after=repo.get(tid);assert after['revision']==2
    updated=after['report']['defects'][0]
    assert updated['area_pixels']==80 and updated['evidence']['original']==row['evidence']['original']
    assert c.get('/api/tasks/'+tid+'/versions/1').json()['report']==before
    assert manager.list(tid)[0]['status']=='applied'

def test_failed_and_stale_mark_does_not_overwrite_report(context):
    c,repo,manager,tid,row=context;manager.queue(repo.get(tid),data(row))
    async def fail(name,args):raise RuntimeError('模型不可用')
    asyncio.run(manager.apply(repo.get(tid),fail))
    assert repo.get(tid)['revision']==1 and manager.list(tid)[0]['status']=='failed'
    doc=repo.get(tid)['report'];doc['defects'][0]['type']='E';repo.save(tid,doc,1,'人工改类')
    calls=[]
    async def forbidden(name,args):calls.append(name)
    asyncio.run(manager.apply(repo.get(tid),forbidden))
    assert calls==[] and repo.get(tid)['revision']==2
    assert '来源证据或类别' in manager.list(tid)[0]['error']

def test_report_structure_and_four_distinct_roles(context):
    c,repo,manager,tid,row=context
    view=c.get('/api/tasks/'+tid+'/report-view?edit=1').json()['html']
    for heading in ['报告签署页','企业声明',*web_report.CHAPTERS]:assert heading in view
    for caption in ['缺陷在大图中的位置','缺陷局部特写','缺陷掩码（二值分割）','缺陷掩码与原图叠加']:assert caption in view
    assert view.count('未提供此类证据')==4
    assert 'data-field="section:conclusion"' in view
    exported=c.get('/api/tasks/'+tid+'/export').text
    assert 'contenteditable' not in exported and 'data-annotate' not in exported
    assert '严重缺陷' not in exported and '未签署' in exported

def test_geometry_retains_stored_pixels_and_removes_exif(context,tmp_path):
    c,repo,manager,tid,row=context
    im=Image.new('RGB',(100,80),'white');exif=Image.Exif();exif[274]=6
    source=tmp_path/'rotated.jpg';im.save(source,exif=exif)
    registered={}
    def record(tid,artifact):registered[Path(artifact['path']).name]=artifact['path'];return '/test/'+Path(artifact['path']).name
    evidence_geometry(repo,tid,row,source,record)
    assert row['width']==100 and row['height']==80
    with Image.open(registered['stored-pixels.png']) as plain:assert plain.size==(100,80) and plain.getexif().get(274,1)==1
    with Image.open(registered['crop.jpg']) as crop:assert crop.size==(40,40)
    assert row['evidence']['location']!=row['evidence']['crop']
