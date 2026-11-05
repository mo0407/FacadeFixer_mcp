"""Local video evidence tools. No network, temperature inference or automatic diagnosis."""
from __future__ import annotations

import base64
import functools
import html
import io
import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .storage import ArtifactStore, sha256, utc_now


def operation(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        job, folder = self.store.create_job(fn.__name__, {})
        try:
            arguments=inspect.signature(fn).bind(self,job,folder,*args,**kwargs)
            arguments.apply_defaults()
            record=self.store.job_status(job)
            record['inputs']={k:v for k,v in arguments.arguments.items() if k not in ('self','job','folder')}
            self.store.write_json(folder/'job.json',record)
            value = fn(self, job, folder, *args, **kwargs)
            result = {"status": "completed", "job_id": job, **value}
            self.store.complete_job(job, result)
            return result
        except Exception as exc:
            self.store.fail_job(job, str(exc))
            return {"status": "failed", "job_id": job, "error": str(exc)}
    return wrapped


class VideoService:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def _json(self, job, folder, name, data):
        path = folder / name
        self.store.write_json(path, data)
        return self.store.artifact(path, name.removesuffix('.json'), job)

    def _read(self, path, schema):
        data = json.loads(self.store.resolve_input(path).read_text(encoding='utf-8'))
        if data.get('schema') != schema:
            raise ValueError(f'Expected {schema}')
        return data

    def _image(self, value):
        with Image.open(self.store.resolve_input(value)) as im:
            return im.convert('RGB')

    @staticmethod
    def _metadata(path):
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError('Unsupported or unreadable video')
            return {'fps': fps, 'frame_count': count,
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'duration_seconds': count / fps}
        finally:
            cap.release()

    @operation
    def inspect(self, job, folder, video_path: str, modality: str = 'visible'):
        if modality not in ('visible', 'thermal'):
            raise ValueError('modality must be visible or thermal')
        path = self.store.resolve_input(video_path)
        data = {'schema': 'facade-video/v1', 'source_sha256': sha256(path),
                'source_path': str(path), 'modality': modality, **self._metadata(path),
                'metadata_scope': 'OpenCV video stream properties only',
                'gps': None, 'radiometric_data': None,
                'note': 'GPS/temperature metadata not inspected; null does not mean absent.'}
        return {'metadata': data, 'manifest': self._json(job, folder, 'video.json', data)}

    @operation
    def extract(self, job, folder, video_path: str, modality: str = 'visible',
                sample_count: int = 16, timestamps: list[float] | None = None):
        if modality not in ('visible', 'thermal') or not 2 <= sample_count <= 200:
            raise ValueError('Invalid modality or sample_count (2..200)')
        path = self.store.resolve_input(video_path)
        before = sha256(path)
        info = self._metadata(path)
        if timestamps is not None:
            if not timestamps or len(timestamps) > 200 or any(not math.isfinite(t) or t < 0 or t >= info['duration_seconds'] for t in timestamps):
                raise ValueError('timestamps must contain 1..200 finite in-range seconds')
            indices = sorted(set(round(t * info['fps']) for t in timestamps))
        else:
            indices = sorted(set(np.linspace(0, info['frame_count'] - 1, sample_count).round().astype(int).tolist()))
        if any(i >= info['frame_count'] for i in indices):
            raise ValueError('Timestamp rounds beyond last frame')
        cap = cv2.VideoCapture(str(path)); frames = []
        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, im = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode frame {index}')
                actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if actual != index:
                    raise ValueError('Decoder did not seek to requested frame')
                output = folder / f'frame-{index:08d}.png'
                Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).save(output)
                frames.append({'frame_id': f'{before[:16]}-{index}', 'frame_index': index,
                               'seconds': index / info['fps'], 'modality': modality,
                               'image': self.store.artifact(output, 'original_frame', job),
                               'width': im.shape[1], 'height': im.shape[0],
                               'sharpness_laplacian': float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())})
        finally:
            cap.release()
        if sha256(path) != before:
            raise ValueError('Source video changed during extraction')
        data = {'schema': 'facade-frames/v1', 'source_sha256': before, 'modality': modality,
                'metadata': info, 'frames': frames,
                'quality_note': 'Sharpness is a relative image score, not defect certainty or an automatic selection.'}
        return {'frames': frames, 'manifest': self._json(job, folder, 'frames.json', data)}

    @operation
    def pair(self, job, folder, visible_manifest: str, thermal_manifest: str,
             offset_seconds: float = 0, tolerance_seconds: float = .25):
        v = self._read(visible_manifest, 'facade-frames/v1')
        t = self._read(thermal_manifest, 'facade-frames/v1')
        if v['modality'] != 'visible' or t['modality'] != 'thermal':
            raise ValueError('Expected visible then thermal manifests')
        if not math.isfinite(offset_seconds) or not math.isfinite(tolerance_seconds) or not 0 <= tolerance_seconds <= 60:
            raise ValueError('Invalid offset/tolerance')
        # Explicit convention: thermal timestamp = visible timestamp + offset.
        edges = sorted((abs(b['seconds'] - a['seconds'] - offset_seconds), a['frame_id'], b['frame_id']) for a in v['frames'] for b in t['frames'])
        used_v, used_t, pairs = set(), set(), []
        for delta, a, b in edges:
            if delta <= tolerance_seconds and a not in used_v and b not in used_t:
                pairs.append({'visible_frame_id': a, 'thermal_frame_id': b, 'time_error_seconds': delta,
                              'status': 'time_candidate', 'pixel_registered': False})
                used_v.add(a); used_t.add(b)
        data = {'schema': 'facade-pairs/v1', 'pairs': pairs, 'offset_seconds': offset_seconds,
                'unmatched_visible': [f['frame_id'] for f in v['frames'] if f['frame_id'] not in used_v],
                'unmatched_thermal': [f['frame_id'] for f in t['frames'] if f['frame_id'] not in used_t],
                'note': 'Greedy nearest-time one-to-one candidates only; verify landmarks. No image registration or defect merging.'}
        return {**data, 'manifest': self._json(job, folder, 'pairs.json', data)}

    @operation
    def track(self, job, folder, video_path: str, frame_manifest: str, stride: int = 3, tracking_width: int = 640):
        data=self._read(frame_manifest,'facade-frames/v1');path=self.store.resolve_input(video_path)
        if sha256(path)!=data['source_sha256']:raise ValueError('Video does not match extracted frames')
        if not 1<=stride<=10 or not 320<=tracking_width<=1280:raise ValueError('stride 1..10; tracking_width 320..1280')
        info=self._metadata(path)
        if info['frame_count']>18000:raise ValueError('Split long videos before tracking (max 18000 frames)')
        scale=min(1.,tracking_width/info['width']);w=round(info['width']*scale);h=round(info['height']*scale)
        sx=w/info['width'];sy=h/info['height'];targets={f['frame_index'] for f in data['frames']}
        cap=cv2.VideoCapture(str(path));prev=None;T=np.eye(3);found={};checks=[];n=0
        try:
            while True:
                ok,raw=cap.read()
                if not ok:break
                if n%stride and n not in targets:n+=1;continue
                gray=cv2.cvtColor(cv2.resize(raw,(w,h)),cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    roi=np.zeros((h,w),np.uint8);roi[int(h*.05):int(h*.95),int(w*.3):int(w*.85)]=255
                    p=cv2.goodFeaturesToTrack(prev,700,.015,7,mask=roi)
                    if p is None or len(p)<20:raise ValueError(f'Insufficient texture at frame {n}')
                    q,st,_=cv2.calcOpticalFlowPyrLK(prev,gray,p,None,winSize=(25,25),maxLevel=3)
                    back,sb,_=cv2.calcOpticalFlowPyrLK(gray,prev,q,None,winSize=(25,25),maxLevel=3)
                    good=(st.ravel()>0)&(sb.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<.8)
                    a=q[good,0];b=p[good,0]
                    if len(a)<20:raise ValueError(f'Insufficient tracks at frame {n}')
                    A,inside=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
                    if A is None or inside.sum()<20:raise ValueError(f'Unstable motion at frame {n}')
                    delta=np.median((b-a)[inside.ravel()>0],axis=0)
                    residual=float(np.median(np.linalg.norm((b-a)[inside.ravel()>0]-delta,axis=1)))
                    if residual>3 or np.linalg.norm(delta)>max(w,h)*.2:raise ValueError(f'Excessive motion/parallax at frame {n}')
                    step=np.eye(3);step[:2,2]=delta;T=T@step;checks.append({'frame':n,'inliers':int(inside.sum()),'translation_residual_pixels':residual})
                if n in targets:found[n]=T.copy()
                prev=gray;n+=1
        finally:cap.release()
        if set(found)!=targets:raise ValueError('Video ended before all requested frames')
        if sha256(path)!=data['source_sha256']:raise ValueError('Video changed during tracking')
        corners=np.float32([[[0,0],[w,0],[w,h],[0,h]]]);pts=np.concatenate([cv2.transform(corners,m[:2])[0] for m in found.values()]);lo=np.floor(pts.min(0));hi=np.ceil(pts.max(0));size=(hi-lo).astype(int)
        if int(size[0])*int(size[1])>24_000_000:raise ValueError('Overview exceeds 24 MP; split the video')
        offset=np.eye(3);offset[:2,2]=-lo;scaling=np.diag([sx,sy,1])
        matrices=[(offset@found[f['frame_index']]@scaling).tolist() for f in data['frames']]
        record={'schema':'facade-video-transforms/v1','transforms':matrices,'frame_ids':[f['frame_id'] for f in data['frames']],
                'canvas_width':int(size[0]),'canvas_height':int(size[1]),'checks':checks,
                'quality_status':'needs_visual_review','method':'dense forward-backward LK; RANSAC filtering; median translation',
                'limitations':'Translation-only baseline. Not metric, perspective rectified or cross-modal registration. Inspect seams before use.'}
        return {**record,'manifest':self._json(job,folder,'transforms.json',record)}

    @operation
    def overview(self, job, folder, frame_manifest: str, transforms: list[list[list[float]]],
                 canvas_width: int, canvas_height: int, markers: list[dict] | None = None):
        """Render reviewed transforms; registration is an explicit caller input, never guessed."""
        data = self._read(frame_manifest, 'facade-frames/v1'); frames = data['frames']
        if len(transforms) != len(frames) or not 1 <= canvas_width <= 20000 or not 1 <= canvas_height <= 30000 or canvas_width * canvas_height > 24_000_000:
            raise ValueError('Invalid transform count or canvas (max 24 MP)')
        canvas = np.full((canvas_height, canvas_width, 3), 244, np.uint8)
        best = np.zeros(canvas.shape[:2], np.float32); mapped = {}; maps = {}
        for frame, matrix in zip(frames, transforms):
            M = np.asarray(matrix, dtype=float)
            if M.shape != (3, 3) or not np.isfinite(M).all() or abs(np.linalg.det(M)) < 1e-10:
                raise ValueError('Need finite invertible 3x3 original-pixel-to-canvas transforms')
            im = np.asarray(self._image(frame['image']['uri'])); h, w = im.shape[:2]
            corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]],float)
            denominators = corners @ M[2]
            if np.min(denominators) * np.max(denominators) <= 0:
                raise ValueError('Transform crosses projective horizon')
            weight = np.broadcast_to((.1 + .9 * (1 - np.abs(np.arange(h) - (h-1)/2) / max(h/2,1)))[:,None], (h,w)).astype(np.float32)
            ww = cv2.warpPerspective(weight, M, (canvas_width,canvas_height))
            warped = cv2.warpPerspective(im, M, (canvas_width,canvas_height))
            keep = ww > best; canvas[keep] = warped[keep]; best[keep] = ww[keep]
            maps[frame['frame_id']] = M; mapped[frame['frame_id']] = (w,h)
        if not np.any(best):
            raise ValueError('No image pixels intersect canvas')
        image = Image.fromarray(canvas); output = folder/'overview.png'; image.save(output)
        draw = ImageDraw.Draw(image); projected = []
        for marker in markers or []:
            fid = marker['frame_id']; w,h = mapped[fid]; box = self._box(marker['bbox_xyxy'],w,h)
            x1,y1,x2,y2 = box; p = cv2.perspectiveTransform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]), maps[fid])[0]
            if not np.isfinite(p).all():
                raise ValueError('Invalid projected marker')
            draw.line([tuple(x) for x in p]+[tuple(p[0])], fill='red', width=3)
            label = str(marker['defect_id']); draw.text(tuple(p[0]), label, fill='red')
            projected.append({'defect_id': label, 'frame_id': fid, 'polygon': p.tolist()})
        annotated = folder/'annotated.png'; image.save(annotated)
        record = {'schema':'facade-overview/v1','coordinate_space':'original_pixels_to_canvas',
                  'modality':data['modality'],'transforms':transforms,'markers':projected,
                  'quality_status':'needs_visual_review','pixel_registered_across_modalities':False,
                  'note':'Provided transforms only. No automatic calibration; source colours preserved, not temperature measurement.',
                  'image':self.store.artifact(output,'overview',job), 'annotated':self.store.artifact(annotated,'overview_annotated',job)}
        return {**record,'manifest':self._json(job,folder,'overview.json',record)}

    @staticmethod
    def _box(value, w, h):
        if len(value) != 4 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or int(x)!=x for x in value):
            raise ValueError('bbox requires four finite integer original-pixel coordinates')
        x1,y1,x2,y2 = map(int,value)
        if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
            raise ValueError('bbox outside original image')
        return [x1,y1,x2,y2]

    @operation
    def curate(self, job, folder, project: dict, defects: list[dict], actor: str, reason: str,
               parent_manifest: str | None = None, overviews: list[dict] | None = None):
        if not actor.strip() or not reason.strip() or not 1 <= len(defects) <= 200:
            raise ValueError('actor, reason, and 1..200 defects required')
        parent = self._read(parent_manifest,'facade-video-review/v1') if parent_manifest else None
        if parent and parent['project'] != project:
            raise ValueError('Revision must retain project identity')
        ids = set(); evidence_ids = set(); rows = []
        for i, d in enumerate(defects):
            did = str(d['defect_id'])
            if not did or did in ids:
                raise ValueError('Unique nonempty defect_id required')
            ids.add(did)
            if d.get('review_status','candidate') not in ('candidate','reviewed','excluded'):
                raise ValueError('Unknown review_status')
            evidences = d.get('evidence',[])
            if not 1 <= len(evidences) <= 8:
                raise ValueError('Need 1..8 representative evidence images per defect')
            snapshots = []
            for j,e in enumerate(evidences):
                eid = str(e['evidence_id'])
                if not eid or eid in evidence_ids:
                    raise ValueError('An evidence_id cannot belong to multiple defects; merge explicitly')
                evidence_ids.add(eid)
                if e['modality'] not in ('visible','thermal'):
                    raise ValueError('Unknown modality')
                im = self._image(e['image_path']); box = self._box(e['bbox_xyxy'],*im.size)
                seconds = e.get('seconds')
                if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
                    raise ValueError('Invalid evidence timestamp')
                prefix = f'{i:03}-{j:02}'; original=folder/f'{prefix}-original.png'; im.save(original)
                crop = folder/f'{prefix}-crop.png'; im.crop(box).save(crop)
                loc = im.copy(); dr=ImageDraw.Draw(loc); dr.rectangle(box,outline='red',width=max(2,im.width//500)); dr.text((box[0],max(0,box[1]-14)),did,fill='red')
                loc.thumbnail((1440,1000)); location=folder/f'{prefix}-location.jpg';loc.save(location,quality=92)
                row={'evidence_id':eid,'modality':e['modality'],'seconds':seconds,'bbox_xyxy':box,
                     'frame_id':e.get('frame_id'),'source_sha256':sha256(self.store.resolve_input(e['image_path'])),
                     'original':self.store.artifact(original,'evidence_original',job),
                     'crop':self.store.artifact(crop,'evidence_crop',job),'location':self.store.artifact(location,'evidence_location',job)}
                if e.get('mask_path'):
                    if not e.get('mask_method') or not e.get('mask_semantics'):
                        raise ValueError('mask_method and mask_semantics are required')
                    if e['modality']=='thermal' and e['mask_semantics']!='pseudocolor_candidate':
                        raise ValueError('Thermal video masks must be pseudocolor_candidate, not diagnosed voids')
                    with Image.open(self.store.resolve_input(e['mask_path'])) as m: a=np.asarray(m)
                    if a.shape!=(im.height,im.width) or not set(np.unique(a)).issubset({0,255}):
                        raise ValueError('Mask must be original-sized binary 0/255')
                    fg=a>0; allowed=np.zeros_like(fg);x1,y1,x2,y2=box;allowed[y1:y2,x1:x2]=True
                    if np.any(fg & ~allowed) or not fg.any():
                        raise ValueError('Mask empty or outside evidence bbox')
                    mp=folder/f'{prefix}-mask.png';Image.fromarray(a).save(mp)
                    rgb=np.array(im);rgb[fg]=np.rint(rgb[fg]*.5+np.array([255,60,50])*.5).astype(np.uint8)
                    op=folder/f'{prefix}-overlay.png';Image.fromarray(rgb).crop(box).save(op)
                    row.update(mask=self.store.artifact(mp,'mask',job),overlay=self.store.artifact(op,'mask_overlay',job),
                               area_pixels=int(fg.sum()),mask_method=e['mask_method'],mask_semantics=e['mask_semantics'])
                snapshots.append(row)
            rows.append({'defect_id':did,'title':str(d.get('title',did)),'observation':str(d.get('observation','')),
                         'interpretation':str(d.get('interpretation','')),'recommendation':str(d.get('recommendation','')),
                         'review_status':d.get('review_status','candidate'),'evidence':snapshots})
        ovs=[]
        for i,o in enumerate(overviews or []):
            if i>=2: raise ValueError('At most two overviews')
            im=self._image(o['image_path']);p=folder/f'overview-{i}.png';im.save(p)
            ovs.append({'label':str(o['label']),'image':self.store.artifact(p,'overview',job),'quality_status':'needs_visual_review'})
        data={'schema':'facade-video-review/v1','project':project,'defects':rows,'overviews':ovs,
              'version':parent['version']+1 if parent else 1,'parent':{'sha256':sha256(self.store.resolve_input(parent_manifest)),'path':str(self.store.resolve_input(parent_manifest))} if parent else None,
              'audit':{'actor':actor,'reason':reason,'timestamp':utc_now(),'identity_verified':False},
              'formal_ready':False,'note':'Explicit supplied grouping; no automatic physical-defect deduplication or diagnosis. Revisions are full snapshots.'}
        return {'defect_count':len(rows),'version':data['version'],'manifest':self._json(job,folder,'review.json',data)}

    def _uri(self, artifact):
        im=self._image(artifact['uri']);buf=io.BytesIO();im.save(buf,format='JPEG',quality=92)
        return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()

    @operation
    def render_html(self, job, folder, review_manifest: str):
        data=self._read(review_manifest,'facade-video-review/v1');esc=lambda s:html.escape(str(s))
        parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>视频分析</title><style>body{font:16px/1.7 sans-serif;background:#f1f4f8;color:#1c2c43;max-width:1200px;margin:auto;padding:24px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;object-fit:contain;max-height:440px}.overview img{max-height:none}.location{max-width:720px}figure{margin:0}figcaption{font-size:13px;color:#65758a}@media(max-width:600px){.grid{grid-template-columns:1fr}}</style><h1>视频分析</h1>',f'<p>版本 {data["version"]} · 初步图像复核，非正式鉴定报告。热像颜色不代表可靠温度或确诊空鼓。</p>']
        if data['overviews']:
            parts.append('<section><h2>立面总览</h2><div class="grid overview">')
            for o in data['overviews']:parts.append(f'<figure><img src="{self._uri(o["image"])}"><figcaption>{esc(o["label"])}</figcaption></figure>')
            parts.append('</div><p>仅为位置参考，可能存在拼接接缝；不同模态不表示已经像素配准。</p></section>')
        for d in data['defects']:
            parts.append(f'<section><h2>{esc(d["defect_id"])} {esc(d["title"])}</h2><p>{esc(d["review_status"])}</p>')
            for e in d['evidence']:
                parts.append(f'<p>{esc(e["modality"])} · {esc(e["seconds"])} 秒</p><img class="location" src="{self._uri(e["location"])}"><div class="grid">')
                for key,label in [('crop','原图局部'),('overlay','候选 Mask 叠加')]:
                    if key in e:parts.append(f'<figure><img src="{self._uri(e[key])}"><figcaption>{label}</figcaption></figure>')
                parts.append('</div>')
            for key,label in [('observation','观察'),('interpretation','判断'),('recommendation','建议')]:parts.append(f'<p><b>{label}：</b>{esc(d[key])}</p>')
            parts.append('</section>')
        path=folder/'report.html';path.write_text(''.join(parts)+'</html>',encoding='utf-8')
        return {'html':self.store.artifact(path,'embedded_report',job),'formal_ready':False}

    @operation
    def render_pdf(self, job, folder, review_manifest: str, font_path: str):
        """Optional reportlab dependency; caller provides a permitted CJK TrueType font."""
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import ParagraphStyle
        data=self._read(review_manifest,'facade-video-review/v1');font=self.store.resolve_input(font_path)
        font_name='F'+sha256(font)[:12];pdfmetrics.registerFont(TTFont(font_name,str(font)))
        path=folder/'report.pdf';c=canvas.Canvas(str(path),pagesize=(595,842));c.setTitle('视频分析')
        style=ParagraphStyle('body',fontName=font_name,fontSize=10,leading=16,wordWrap='CJK');page_no=1;y=790
        def footer():
            c.setFont(font_name,9);c.drawString(35,22,f'视频分析 | 待人工确认 | {page_no}')
        def newpage():
            nonlocal page_no,y
            footer();c.showPage();page_no+=1;y=790
        def para(value):
            nonlocal y
            p=Paragraph(html.escape(str(value)),style);_,height=p.wrap(525,10000)
            if height>700:raise ValueError('Text too long for PDF; shorten narrative')
            if y-height<45:newpage()
            p.drawOn(c,35,y-height);y-=height+10
        def pic(a,maxheight=200):
            nonlocal y
            from reportlab.lib.utils import ImageReader
            im=self._image(a['uri']);ratio=min(525/im.width,maxheight/im.height);w,h=im.width*ratio,im.height*ratio
            if y-h<45:newpage()
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=92);buf.seek(0);c.drawImage(ImageReader(buf),35+(525-w)/2,y-h,w,h);y-=h+14
        para('视频分析 - 初步图像复核');para(f'版本 {data["version"]}；不是现场鉴定。热像仅用于异常候选复核。')
        for o in data['overviews']:para(o['label']);pic(o['image'],560)
        for d in data['defects']:
            newpage();c.bookmarkPage(d['defect_id']);para(d['defect_id']+' '+d['title'])
            for e in d['evidence']:
                para(f'{e["modality"]} / {e["seconds"]} 秒');pic(e['location'],155);pic(e['crop'],130)
                if 'overlay' in e:pic(e['overlay'],130)
            for k in ('observation','interpretation','recommendation'):para(d[k])
        footer();c.save()
        return {'pdf':self.store.artifact(path,'report_pdf',job),'page_count':page_no,'visual_review_required':True,'formal_ready':False}
