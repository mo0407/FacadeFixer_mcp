"""Editable rendition of FacadeFixer_report/report_web, with explicit draft states."""
from __future__ import annotations
import copy
import html
from pathlib import Path

TEMPLATE = 'FacadeFixer_report/report_web@0c679ad'
SECTIONS = {
    'client':'待补充','inspector':'待补充','author':'待补充','reviewer':'待补充','approver':'待补充',
    'statement':'1、本报告仅对委托检测范围内容负责。<br>2、未经编写、审核、批准确认，不作为正式签署文件。<br>3、委托方提供资料的真实可靠性由委托方负责。<br>4、报告异议可向报告出具方书面提出。',
    'purpose':'了解建筑外墙可见缺陷情况，为现场复核、维修和后续巡检提供依据。',
    'scope':'待补充委托范围及未覆盖、不可达部位。',
    'method':'采用上传图片及视频抽帧进行可见缺陷检测与分割，结果为模型候选，需人工复核。',
    'standards':'待补充并核验适用标准及版本。',
    'materials':'待补充委托文件、设计图纸、现场记录等资料。',
    'instrument_name':'待补充','instrument_model':'待补充','instrument_id':'待补充',
    'conclusion':'尚未形成正式鉴定结论。模型估算、未采集、不可达和待确认信息不作为正式量测或专家签署结论。',
    'recommendations':'建议现场核验候选缺陷及采集覆盖范围，完善责任信息和证据后再形成正式报告。',
}
ROW_TEXT = ('captured_at','gps_latitude','gps_longitude','capture_height','pedestrian_risk','severity','severity_reason_html')
CHAPTERS = ('一、工程概况','二、鉴定目的、对象、内容及方法','三、鉴定依据及材料','四、主要检测仪器','五、现场勘验及检测情况','六、鉴定结论','七、建议与说明')

def normalize(doc):
    doc=copy.deepcopy(doc)
    doc['template']=TEMPLATE
    doc['sections']={**SECTIONS,**doc.get('sections',{})}
    for row in doc.get('defects',[]):
        for key in ROW_TEXT: row.setdefault(key,'待复核' if key in {'severity','pedestrian_risk'} else '待补充')
    return doc

def render(doc, revision, labels, clean, *, editable=False, url_for=lambda x:x):
    """One renderer for editor, preview and self-contained exported HTML."""
    doc=normalize(doc);project=doc.get('project',{});rows=doc.get('defects',[]);sections=doc['sections']
    esc=lambda x:html.escape(str(x if x is not None else '待补充'),quote=True)
    def field(value,key,row_id='',plain=False):
        content=esc(value) if plain else clean(value)
        if not editable:return content
        return f'<div class="editable" contenteditable="true" role="textbox" aria-label="{esc(key)}" data-field="{esc(key)}" data-id="{esc(row_id)}">{content}</div>'
    def section(key): return field(sections[key],'section:'+key)
    def page(name,body):return f'<section class="report-page" id="{name}">{body}</section>'
    def table(body):return '<table class="template-table">'+body+'</table>'
    def tr(*cells):return '<tr>'+''.join('<td>'+str(c)+'</td>' for c in cells)+'</tr>'
    def h(index):return '<h2>'+CHAPTERS[index]+'</h2>'
    result=[page('cover','<div class="report-cover"><h1>'+field(doc['title'],'title',plain=True)+'</h1><p class="cover-subtitle">外墙检测鉴定报告</p><p>基于无人机航拍与 AI 智能识别技术</p><p>检测期次：'+esc(project.get('period') or '待补充')+f'<br>报告版本：V{revision}<br>草稿 · 未签署 / 待复核</p></div>')]
    result.append(page('responsibility','<h2>报告签署页</h2>'+table(tr('委托单位',section('client'),'工程名称',esc(project.get('building_name')))+tr('工程地点',esc(project.get('location') or '待补充'),'项目类型','外墙外立面检测鉴定')+tr('现场检测',section('inspector'),'报告编写',section('author'))+tr('报告审核',section('reviewer'),'报告批准',section('approver'))+tr('冻结时间','未冻结','报告编号',esc(doc.get('report_number','待补充'))))+'<p class="draft-note">填写责任人不等于完成签署，本版本不生成签章或冻结证明。</p>'))
    result.append(page('statement','<h2>企业声明</h2>'+section('statement')))
    result.append(page('contents','<h2>目　录</h2>'+''.join(f'<p><a href="#chapter-{i}">{name}</a></p>' for i,name in enumerate(CHAPTERS))))
    result.append(page('chapter-0',h(0)+field(doc.get('summary_html','待补充'),'summary_html')))
    result.append(page('chapter-1',h(1)+'<h3>2.1 鉴定目的</h3>'+section('purpose')+'<h3>2.2 鉴定对象</h3><p>'+esc(project.get('building_name','待补充'))+' 外墙外立面。</p><h3>2.3 鉴定内容</h3>'+section('scope')+'<h3>2.4 鉴定方法</h3>'+section('method')+'<h3>2.5 鉴定标准</h3>'+section('standards')))
    result.append(page('chapter-2',h(2)+section('materials')))
    result.append(page('chapter-3',h(3)+table('<tr><th>序号</th><th>仪器名称</th><th>型号</th><th>编号</th></tr>'+tr('1',section('instrument_name'),section('instrument_model'),section('instrument_id')))))
    counts={}
    for row in rows:counts[row['type']]=counts.get(row['type'],0)+1
    stats=table('<tr><th>类别编号</th><th>缺陷类型</th><th>缺陷数量（处）</th></tr>'+(''.join(tr(esc(k),esc(labels.get(k,k)),v) for k,v in counts.items()) or tr('—','未记录候选缺陷',0)))
    summary=table('<tr><th>序号</th><th>缺陷类型</th><th>楼栋</th><th>立面</th><th>严重程度</th><th>处理建议</th></tr>'+''.join(tr(row['number'],esc(labels.get(row['type'],row['type'])),esc(project.get('building_name')),esc(row['facade']),esc(row['severity']),clean(row['advice_html'])) for row in rows))
    result.append(page('chapter-4',h(4)+f'<p>本次记录 {len(rows)} 项候选缺陷。候选数量不代表已确认缺陷数量。</p><h3>5.1 缺陷类别统计</h3>'+stats+'<h3>5.2 建筑物外墙外立面缺陷情况</h3>'+summary))
    for row in rows:
        rid=row['id'];number=row['number'];evidence=row.get('evidence',{})
        f=lambda key:field(row.get(key,'待补充'),key,rid,plain=not key.endswith('_html'))
        kind=esc(labels.get(row['type'],row['type']))
        if editable:
            kind=f'<select aria-label="第{number}条缺陷类型" data-row="{esc(rid)}" data-key="type">'+''.join(f'<option value="{esc(k)}" {"selected" if row["type"]==k else ""}>{esc(v)}</option>' for k,v in ({row['type']:'待复核',**labels} if row['type'] not in labels else labels).items())+'</select>'
        body=f'<div class="detail-title">缺陷 {number} — {esc(labels.get(row["type"],row["type"]))}</div>'
        body+=table(tr('原始图像',esc(row.get('image_name') or '待补充'),'拍摄时间',f('captured_at'))+tr('GPS纬度',f('gps_latitude'),'GPS经度',f('gps_longitude'))+tr('拍摄高度',f('capture_height'),'行人风险',f('pedestrian_risk'))+tr('所在立面',f('facade'),'像素面积',esc(row.get('area_pixels') if row.get('area_pixels') is not None else '未测量')+' px')+f'<tr><td>缺陷类型</td><td colspan="3">{kind}<p>判断依据：</p>{f("description_html")}</td></tr><tr><td>严重程度</td><td colspan="3">{f("severity")}<p>严重程度依据：</p>{f("severity_reason_html")}</td></tr><tr><td>处理建议</td><td colspan="3">{f("advice_html")}</td></tr>')
        pictures=[]
        for index,(role,caption) in enumerate([('location','缺陷在大图中的位置'),('crop','缺陷局部特写'),('mask','缺陷掩码（二值分割）'),('overlay','缺陷掩码与原图叠加')]):
            url=evidence.get(role)
            content=f'<img src="{esc(url_for(url))}" alt="{caption}">' if url else '<div class="evidence-missing">未提供此类证据</div>'
            if editable and url:content=f'<button type="button" class="evidence-open" data-annotate="{esc(rid)}" title="点击或按 A 快捷标注">{content}</button>'
            pictures.append(f'<figure>{content}<figcaption>图 {number}-{chr(97+index)} — {caption}</figcaption></figure>')
        body+='<div class="four-evidence">'+''.join(pictures)+'</div>'
        if editable:body+=f'<button class="quiet annotate-row" data-annotate="{esc(rid)}">标注 bbox / Mask · A</button>'
        body+='<p class="draft-note">模型候选 / 待人工复核。物理面积、严重程度和行人风险不得由像素面积直接推定。</p>'
        result.append(page('defect-'+str(number),f'<div data-defect="{esc(rid)}">'+body+'</div>'))
    result.append(page('chapter-5',h(5)+section('conclusion')))
    result.append(page('chapter-6',h(6)+section('recommendations')))
    return ''.join(result)

def stylesheet():return (Path(__file__).parent/'web_static/report.css').read_text(encoding='utf-8')
