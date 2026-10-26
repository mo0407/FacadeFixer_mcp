"""Persistent original-pixel annotation queue; explicit, versioned batch inference."""
import copy
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from PIL import Image,ImageDraw
from starlette.exceptions import HTTPException

def fingerprint(row):
    return hashlib.sha256(json.dumps({k:row.get(k) for k in ('id','type','bbox_xyxy','evidence')},sort_keys=True).encode()).hexdigest()

def evidence_geometry(repo,task_id,row,source,register):
    """Make actual full-image box and crop, never repeat a picture to fill slots."""
    folder=repo.root/'annotations'/task_id/uuid4().hex;folder.mkdir(parents=True)
    with Image.open(source) as im:
        im=im.convert('RGB');row['width'],row['height']=im.size
        plain=folder/'stored-pixels.png';im.save(plain,exif=b'')
        row['annotation_image']=register(task_id,{'path':str(plain),'sha256':hashlib.sha256(plain.read_bytes()).hexdigest()})
        box=row.get('bbox_xyxy')
        if not box:return
        x1,y1,x2,y2=map(int,box)
        crop=folder/'crop.jpg';im.crop((x1,y1,x2,y2)).save(crop)
        draw=ImageDraw.Draw(im);draw.rectangle((x1,y1,max(x1,x2-1),max(y1,y2-1)),outline='#3563e9',width=max(2,round(min(im.size)/200)))
        location=folder/'location.jpg';im.save(location)
    for key,path in [('location',location),('crop',crop)]:
        row['evidence'][key]=register(task_id,{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

class Corrections:
    def __init__(self,repo,store,register):
        self.repo,self.store,self.register=repo,store,register
        with repo.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS corrections(id TEXT PRIMARY KEY,task_id TEXT,defect_id TEXT,created TEXT,status TEXT,payload TEXT,source_fingerprint TEXT,base_revision INTEGER,result_revision INTEGER,error TEXT)')
            con.execute("UPDATE corrections SET status='failed',error='服务重启，重新提交本批即可重试' WHERE status='applying'")

    def list(self,tid):
        with self.repo.connect() as con: rows=con.execute('SELECT * FROM corrections WHERE task_id=? ORDER BY created',(tid,)).fetchall()
        return [{**dict(row),'payload':json.loads(row['payload'])} for row in rows]

    def queue(self,task,data):
        from .web_app import now,encode
        if task['status']=='running':raise HTTPException(409,'任务正在处理')
        if data.get('base_revision')!=task['revision']:raise HTTPException(409,'报告已有新版本，请重新打开标注')
        row=next((r for r in (task['report'] or {}).get('defects',[]) if r['id']==data.get('defect_id')),None)
        if not row:raise HTTPException(404,'缺陷不存在')
        if not row.get('width') or not row.get('bbox_xyxy'):raise HTTPException(400,'缺少原图尺寸或检测框，无法标注')
        width,height=row['width'],row['height']
        def coord(values,size):
            return isinstance(values,list) and len(values)==size and all(isinstance(n,(int,float)) and not isinstance(n,bool) and math.isfinite(n) for n in values)
        box=data.get('bbox_xyxy')
        if box is not None and (not coord(box,4) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)):
            raise HTTPException(400,'框必须位于原图内，采用原始像素 xyxy 坐标')
        payload={'bbox_xyxy':box,'note':str(data.get('note',''))[:2000]}
        for key in ('positive_points','negative_points'):
            points=data.get(key,[])
            if not isinstance(points,list) or len(points)>100 or any(not coord(p,2) or not(0<=p[0]<width and 0<=p[1]<height) for p in points):raise HTTPException(400,'标注点必须位于原图内，每类最多100个')
            payload[key]=points
        if not box and not payload['positive_points'] and not payload['negative_points']:raise HTTPException(400,'请重画框或标记至少一个 Mask 点；文字备注不能单独驱动模型修改')
        profile=task['settings']['segmenter']
        if (payload['positive_points'] or payload['negative_points']) and profile not in {'mobile_sam','sam3'}:raise HTTPException(400,'正负点修改需要 MobileSAM 或 SAM3，请先调整模型')
        cid=uuid4().hex
        with self.repo.connect() as con:
            con.execute("UPDATE corrections SET status='superseded' WHERE task_id=? AND defect_id=? AND status IN ('pending','failed')",(task['id'],row['id']))
            con.execute('INSERT INTO corrections VALUES(?,?,?,?,?,?,?,?,?,?)',(cid,task['id'],row['id'],now(),'pending',encode(payload),fingerprint(row),task['revision'],None,None))
        return cid

    def cancel(self,tid,cid):
        if self.repo.get(tid)['status']=='running':raise HTTPException(409,'任务正在处理')
        with self.repo.connect() as con:
            count=con.execute("UPDATE corrections SET status='cancelled' WHERE id=? AND task_id=? AND status IN ('pending','failed')",(cid,tid)).rowcount
        if not count:raise HTTPException(409,'该标注不能撤回')

    def update(self,cid,status,error=None,revision=None):
        with self.repo.connect() as con:con.execute('UPDATE corrections SET status=?,error=?,result_revision=? WHERE id=?',(status,error,revision,cid))

    async def apply(self,task,call):
        """call is the existing MCP client; all pixel modifications stay in model tools."""
        from .web_app import encode
        document=copy.deepcopy(task['report']);completed=[];failed=[]
        items=[r for r in self.list(task['id']) if r['status'] in {'pending','failed'}]
        for item in items:
            self.update(item['id'],'applying')
            try:
                row=next((r for r in document['defects'] if r['id']==item['defect_id']),None)
                if row is None or fingerprint(row)!=item['source_fingerprint']:raise ValueError('来源证据或类别已经变化，请重新标注后提交')
                payload=item['payload'];profile=task['settings']['segmenter']
                if (payload['positive_points'] or payload['negative_points']) and profile not in {'mobile_sam','sam3'}:raise ValueError('当前分割模型不支持点提示')
                asset=next(a for a in self.repo.get(task['id'])['assets'] if a['id']==row['evidence']['original'].rsplit('/',1)[-1])
                source=asset['path'];box=payload['bbox_xyxy'] or row['bbox_xyxy']
                target={'instance_id':str(row['source_instance']),'class_code':row['type'],'bbox_xyxy':box}
                self.repo.message(task['id'],'status',f'正在统一修改第 {row["number"]} 项缺陷…')
                result=await call('facade_models_segment',{'image_path':source,'defects':[target],'model_profile':profile,'prompt_mode':'bbox'})
                if payload['positive_points'] or payload['negative_points']:
                    result=await call('facade_models_refine',{'image_path':source,'annotation_manifest_path':result['annotation_manifest_path'],'instance_id':str(row['source_instance']),'positive_points':payload['positive_points'],'negative_points':payload['negative_points'],'model_profile':profile})
                instance=next(r for r in result['instances'] if str(r['instance_id'])==str(row['source_instance']))
                updated=copy.deepcopy(row);updated['bbox_xyxy']=instance['bbox_xyxy'];updated['area_pixels']=instance['area_pixels']
                updated['evidence']['mask']=self.register(task['id'],instance['mask'])
                updated['evidence']['overlay']=self.register(task['id'],instance.get('overlay') or result['overlay'])
                updated['annotation_manifest']=result['annotation_manifest'];updated['last_correction_id']=item['id']
                evidence_geometry(self.repo,task['id'],updated,source,self.register)
                document['defects'][document['defects'].index(row)]=updated
                completed.append(item['id'])
            except Exception as error:
                self.update(item['id'],'failed',str(error)[:800]);failed.append(item['id'])
        if completed:
            try:
                revision=self.repo.save(task['id'],document,task['revision'],f'统一修改 bbox / Mask · {len(completed)} 项 · 本地用户')
            except Exception:
                for cid in completed:self.update(cid,'failed','保存冲突，原报告未覆盖，请刷新后重试')
                raise
            for cid in completed:self.update(cid,'applied',revision=revision)
        else:revision=task['revision']
        self.repo.update(task['id'],status='ready' if not failed else 'failed',error=f'{len(failed)} 项修改未完成，请在待修改列表查看原因' if failed else None)
        self.repo.message(task['id'],'assistant',f'统一修改完成：{len(completed)} 项已生成新证据，{len(failed)} 项未完成。当前报告 V{revision}，仍需人工复核。')
