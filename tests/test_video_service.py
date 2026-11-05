import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from fastmcp import Client

from facade_mcp.storage import ArtifactStore
from facade_mcp.video_service import VideoService
from facade_mcp.video_server import create_server


@pytest.fixture
def fixture(tmp_path):
    store=ArtifactStore(tmp_path/'jobs',[tmp_path]);s=VideoService(store)
    path=tmp_path/'video.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
    assert writer.isOpened()
    texture=np.random.default_rng(17).integers(0,256,(300,320,3),dtype=np.uint8)
    for i in range(15):writer.write(texture[i:i+240])
    writer.release()
    return s,store,path


def test_extract_pair_and_motion_are_traced(fixture):
    s,store,path=fixture
    inspect=s.inspect(str(path));assert inspect['metadata']['frame_count']==15
    assert inspect['metadata']['radiometric_data'] is None
    a=s.extract(str(path),sample_count=3);b=s.extract(str(path),'thermal',3)
    assert [f['frame_index'] for f in a['frames']]==[0,7,14]
    assert Image.open(a['frames'][0]['image']['path']).size==(320,240)
    pairs=s.pair(a['manifest']['uri'],b['manifest']['uri'])
    assert len(pairs['pairs'])==3 and not pairs['pairs'][0]['pixel_registered']
    shifted=s.pair(a['manifest']['uri'],b['manifest']['uri'],99,.1)
    assert not shifted['pairs'] and len(shifted['unmatched_visible'])==3
    tracked=s.track(str(path),a['manifest']['uri'],stride=1,tracking_width=320)
    assert tracked['status']=='completed',tracked
    assert abs(tracked['transforms'][-1][1][2]-14)<2
    overview=s.overview(a['manifest']['uri'],tracked['transforms'],tracked['canvas_width'],tracked['canvas_height'],
                        [{'frame_id':a['frames'][0]['frame_id'],'defect_id':'D1','bbox_xyxy':[20,20,40,40]}])
    assert overview['status']=='completed',overview
    assert len(overview['markers'])==1


def make_defect(s,path):
    a=s.extract(str(path),sample_count=2)
    return {'defect_id':'D1','title':'<script>bad</script>','observation':'candidate',
            'evidence':[{'evidence_id':'E1','modality':'visible','image_path':a['frames'][0]['image']['uri'],
                         'bbox_xyxy':[10,10,60,60],'seconds':0}]}


def test_curate_versions_masks_html_and_tampering(fixture,tmp_path):
    s,store,path=fixture;d=make_defect(s,path)
    mask=np.zeros((240,320),np.uint8);mask[20:35,20:35]=255;mp=tmp_path/'mask.png';Image.fromarray(mask).save(mp)
    d['evidence'][0].update(mask_path=str(mp),mask_method='test fixture',mask_semantics='visible_candidate')
    one=s.curate({'building_id':'B1'},[d],'reviewer','first review')
    assert one['version']==1
    two=s.curate({'building_id':'B1'},[d],'reviewer','correct interpretation',one['manifest']['uri'])
    assert two['version']==2 and one['manifest']['path']!=two['manifest']['path']
    doc=json.loads(Path(two['manifest']['path']).read_text())
    assert doc['parent']['sha256']==one['manifest']['sha256']
    assert doc['defects'][0]['evidence'][0]['area_pixels']==225
    rendered=s.render_html(two['manifest']['uri']);text=Path(rendered['html']['path']).read_text(encoding='utf-8')
    assert '&lt;script&gt;' in text and '<script>bad' not in text and 'data:image/jpeg;base64,' in text
    assert 'evidence_location' in doc['defects'][0]['evidence'][0]['location']['role']
    # Assets cannot silently mutate beneath a reviewed report.
    Path(doc['defects'][0]['evidence'][0]['crop']['path']).write_bytes(b'corrupt')
    assert s.render_html(two['manifest']['uri'])['status']=='failed'


def test_invalid_inputs_are_failed_jobs(fixture,tmp_path):
    s,store,path=fixture
    r=s.extract(str(path),timestamps=[float('nan')]);assert r['status']=='failed'
    assert store.job_status(r['job_id'])['status']=='failed'
    d=make_defect(s,path)
    assert s.curate({},[d,d],'x','reason')['status']=='failed'
    d['evidence'][0]['bbox_xyxy']=[0,0,999,999]
    assert s.curate({},[d],'x','reason')['status']=='failed'
    outside=tmp_path.parent/'not-allowed.mp4'
    assert s.inspect(str(outside))['status']=='failed'


def test_thermal_semantics_not_diagnosis(fixture,tmp_path):
    s,_,path=fixture;d=make_defect(s,path);m=np.zeros((240,320),np.uint8);m[20:30,20:30]=255
    p=tmp_path/'thermal-mask.png';Image.fromarray(m).save(p)
    d['evidence'][0].update(modality='thermal',mask_path=str(p),mask_method='colour threshold',mask_semantics='confirmed_void')
    assert s.curate({},[d],'x','reason')['status']=='failed'
    d['evidence'][0]['mask_semantics']='pseudocolor_candidate'
    assert s.curate({},[d],'x','reason')['status']=='completed'


def test_mcp_surface_and_real_stdio(fixture):
    s,store,path=fixture
    async def run():
        async with Client(create_server(store)) as client:
            names={t.name for t in await client.list_tools()}
            assert len(names)==9 and 'video_track_transforms' in names
            result=await client.call_tool('video_inspect',{'video_path':str(path)})
            assert not result.is_error
        config={'mcpServers':{'video':{'command':sys.executable,'args':['-m','facade_mcp.video_server'],
                 'env':{'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src'),'FACADE_WORKSPACE':str(store.root),
                        'FACADE_ALLOWED_INPUT_ROOTS':str(path.parent),'FASTMCP_CHECK_FOR_UPDATES':'off'}}}}
        async with Client(config) as client:
            result=await client.call_tool('video_inspect',{'video_path':str(path)})
            assert not result.is_error
            assert 'completed' in str(result)
    asyncio.run(run())
