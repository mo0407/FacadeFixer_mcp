"""Editable HTML pages for Canva import; no flattened UI screenshots."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
tokens = {"colors": {"primary": "#3563E9", "text": "#172033", "muted": "#778196", "canvas": "#FFFFFF", "sidebar": "#F7F8FA", "border": "#E5E9F0", "selection": "#EDF2FF"}, "font": "Microsoft YaHei, sans-serif", "spacing": [4,8,12,16,24,32,48], "radii": [8,12,16,24], "desktop": {"width":1440,"height":900,"sidebar":220,"chat":490,"editor":730}}
ROOT.joinpath('tokens.json').write_text(json.dumps(tokens,ensure_ascii=False,indent=2),encoding='utf-8')
def box(x,y,w,h,text='',bg='white',size=16,color='#172033',border=True):
    return f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:{bg};color:{color};font-size:{size}px;border:{"1px solid #E5E9F0" if border else "none"};border-radius:12px;padding:16px;box-sizing:border-box;line-height:1.6">{text}</div>'
def text(x,y,w,h,label,size=16,color='#172033'):
    return box(x,y,w,h,label,'transparent',size,color,False)
def shell(title):
    return box(0,0,220,900,'','#F7F8FA',border=False)+text(12,12,205,60,'▥ FacadeFixer',23)+box(16,88,188,52,'＋ 新建报告','#3563E9',16,'white')+text(8,160,200,45,'最近任务',13,'#778196')+box(12,210,196,50,'A栋外墙巡检','#EDF2FF',15,'#3563E9',False)+text(12,270,196,50,'园区北楼复查',15)+text(12,820,196,50,'本地工作区　⚙',14)+text(244,12,650,60,title,20)
pages=[]
pages.append(('01 上传与新建',shell('新建报告')+text(440,180,760,80,'上传素材，开始巡检报告',36)+text(415,260,800,50,'添加视频、图片和建筑资料，用一句话描述你的需求',18,'#778196')+box(380,360,900,250)+box(400,380,240,60,'南立面照片 · 12 张','#F7F8FA')+box(652,380,260,60,'巡检视频.mp4 · 02:36','#F7F8FA')+text(400,455,830,85,'请根据这些素材生成外墙巡检报告，按立面整理缺陷，并附上标注图片。',19)+box(400,548,140,44,'＋ 添加素材')+box(555,548,130,44,'建筑信息')+box(1210,548,50,44,'↑','#3563E9',18,'white')+text(540,650,620,45,'A栋办公楼 · 2026年10月巡检　✎',15)+text(410,735,850,50,'缺少资料时助手会继续询问；报告生成后支持直接编辑和对话修改。',14,'#778196')))
editor=shell('A栋外墙巡检　· 已保存')+box(245,100,440,110,'请根据上传的视频和照片，生成A栋外墙巡检报告。','#F7F8FA')+text(245,240,440,100,'报告草稿已生成。你可以点击右侧正文直接修改，或继续告诉我修改要求。')+box(245,345,440,80,'▤ A栋外墙巡检报告 · V1<br>可编辑报告')+box(245,470,440,85,'将第3条缺陷改为白华，并补充处理建议。','#F7F8FA')+text(245,575,440,90,'已更新第3条缺陷。<br><span style="color:#3563E9">V2 · 查看修改</span>')+box(245,765,440,100,'描述修改要求，或先选中报告内容…')+box(710,0,730,900,'','#F7F8FA',border=False)+box(730,15,85,42,'编辑','#3563E9',15,'white')+text(824,12,90,42,'预览')+text(1080,12,160,42,'修改记录')+box(1280,15,130,42,'导出 HTML','#3563E9',15,'white')+box(730,75,680,52,'正文 ▾　 B　 I　 U　 ≡　 ↶　 ↷　　　 已自动保存')+box(730,147,680,730)+text(758,166,620,70,'A栋外墙巡检报告　✎',30)+text(758,236,620,40,'草稿 · 待复核',14,'#778196')+box(758,300,620,65,'项目名称　A栋办公楼　　　 巡检期次　2026年10月','#F7F8FA')+text(758,390,620,48,'03　白华（泛碱）',22,'#3563E9')+box(758,450,620,80,'缺陷编号　D-003　　　 所在立面　南立面<br>缺陷类型　白华　　　　 复核状态　待复核')+box(758,550,300,120,'原始图片 · 证据区','#F1F3F7',16,'#778196')+box(1078,550,300,120,'Mask叠加 · 证据区','#EDF2FF',16,'#778196')+box(758,710,620,105,'处理建议<br><span style="background:#DCE7FF">建议现场复核白华范围</span>，结合含水情况排查原因。│')+box(1060,680,250,45,'✧ 让 AI 修改选中内容','#EDF2FF',15,'#3563E9')
pages.append(('02 报告编辑与对话修改',editor))
pages.append(('03 建筑信息与运行配置',shell('建筑信息 · 按需打开的抽屉')+box(800,80,600,780)+text(825,105,550,60,'建筑信息与模型配置',26)+box(825,200,550,70,'建筑名称 *　A栋办公楼')+box(825,290,550,70,'巡检期次　2026年10月')+box(825,380,550,70,'建筑位置　可填写地址或经纬度')+box(825,470,550,70,'检测模型　YOLO12m　▾')+box(825,560,550,70,'Mask 模型　MobileSAM　▾　提示方式　框')+text(825,650,550,100,'视频按固定间隔抽帧，保留来源时间。未配置在线语言模型时，提供明确的本地指令范围。',16,'#778196')+box(1110,770,265,50,'保存设置','#3563E9',16,'white')))
pages.append(('04 修改记录',shell('报告修改记录')+text(300,120,1000,70,'每次修改都有来源，可查看旧版并恢复为新版本',26)+box(300,230,1000,120,'V3　直接编辑 · 本地用户<br>处理建议：补充现场复核说明<br>查看版本　恢复为新版本')+box(300,375,1000,120,'V2　提示词修改 · 本地用户<br>“将第3条缺陷改为白华”<br>保留原始证据与模型分类来源')+box(300,520,1000,120,'V1　模型生成 · 待人工复核<br>输入素材、工具调用及报告证据可追溯')+text(300,700,1000,80,'冲突处理：报告已被其他窗口修改时，提示刷新，不覆盖新版本。',18,'#778196')))
pages.append(('05 开发规范',text(50,35,1300,80,'FacadeFixer · 蓝白轻量界面开发规范',34)+text(50,140,1250,70,'1440 × 900　侧栏 220px　对话区 490px　报告区 730px',23)+box(65,255,290,100,'Primary　#3563E9','#3563E9',20,'white')+box(385,255,290,100,'Text　#172033','#172033',20,'white')+box(705,255,290,100,'Canvas　#FFFFFF')+box(1025,255,290,100,'Sidebar　#F7F8FA','#F7F8FA')+text(50,415,1300,90,'字号 14 / 16 / 20 / 30　间距 4 / 8 / 12 / 16 / 24 / 32 / 48　圆角 8 / 12 / 16 / 24',22)+text(50,540,1300,100,'组件：历史会话、附件卡片、提示词输入框、资料抽屉、报告编辑器、版本记录、错误提示',22)+text(50,680,1300,130,'交互：上传 → 补充资料 → 生成 → 编辑 → 导出。加载时禁止重复提交；失败明确显示原因并保留素材。<br>窄屏：折叠侧栏，对话 / 报告切换。所有内容为可编辑文字与布局元素，非截图。',20)))
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body style="margin:0;font-family:Microsoft YaHei,Arial,sans-serif;background:#e8ebf0">'+''.join(f'<section data-document-role="page" data-label="{title}" style="width:1440px;height:900px;position:relative;background:white;overflow:hidden;margin-bottom:24px">{body}</section>' for title,body in pages)+'</body></html>'
ROOT.joinpath('canva-editable.html').write_text(html,encoding='utf-8')
import re
from html import unescape
layouts=[]
for title,body in pages:
    elements=[]
    for index,match in enumerate(re.finditer(r'<div style="([^"]+)">(.*?)</div>',body,re.S)):
        styles=dict(part.split(':',1) for part in match[1].split(';') if ':' in part)
        elements.append({'id':f'element-{index+1}','text':unescape(re.sub('<[^>]+>',' ',match[2])).strip(),'bounds':{key:int(styles[prop].removesuffix('px')) for key,prop in [('x','left'),('y','top'),('width','width'),('height','height')]},'style':styles})
    layouts.append({'name':title,'width':1440,'height':900,'elements':elements})
ROOT.joinpath('layout.json').write_text(json.dumps({'pages':layouts},ensure_ascii=False,indent=2),encoding='utf-8')
