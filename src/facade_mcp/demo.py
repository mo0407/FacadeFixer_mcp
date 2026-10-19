"""A synthetic smoke fixture with supplied boxes, explicitly not detector predictions."""
from pathlib import Path
from PIL import Image, ImageDraw
from .storage import ArtifactStore


def create_demo(directory:Path) -> dict:
    directory.mkdir(parents=True,exist_ok=True)
    image = Image.new("RGB",(800,600),(207,205,195))
    draw = ImageDraw.Draw(image)
    draw.line([(170,95),(181,180),(163,270),(190,365)],fill=(35,33,32),width=6)
    draw.line([(177,180),(215,218)],fill=(40,37,35),width=3)
    draw.ellipse((420,180,580,330),fill=(167,74,30))
    path = directory/"demo_facade.png"
    image.save(path)
    annotations = {"schema":"bbox_class_only_v1","images":[{"image_path":str(path.resolve()),"defects":[
        {"id":1,"class_code":"C","class_name":"Concrete_Crack","bbox_xyxy":[120,65,245,395]},
        {"id":2,"class_code":"R","class_name":"Corrosion","bbox_xyxy":[390,150,610,360]}
    ]}]}
    annotation_path = directory/"supplied_annotations.json"
    ArtifactStore.write_json(annotation_path,annotations)
    request = {"fixture":"synthetic image with manually supplied demonstration boxes",
               "site":{"building_id":"demo-building","name":"外立面巡检演示","longitude":120.15,"latitude":30.28,"building_height_m":18,"building_radius_m":10},
               "planning":{"stand_off_m":8,"capture_heights_m":[4,10,16],"waypoint_spacing_m":10},
               "project":{"project_id":"demo-building","name":"外立面巡检演示","period":"演示期次","fixture":"synthetic demonstration, not real inspection"},
               "image_paths":[str(path.resolve())],"detection":{"method":"annotations","annotation_path":str(annotation_path.resolve())},
               "mask":{"method":"opencv"}}
    ArtifactStore.write_json(directory/"workflow.json",request)
    return request
