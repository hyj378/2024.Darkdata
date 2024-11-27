#!/usr/bin/env python
# -*- coding: utf-8 -*-
### VBB 포멧으로 작성된 MAT파일을 JSON형식으로 변환하는 코드입니다.
import os
import glob
from scipy.io import loadmat
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def convert_vbb(path, save_dir=None, image_size=None):
    assert path[-3:] == 'vbb'

    vbb_info = loadmat(path)
    # bounding box list
    objLists = vbb_info['A'][0][0][1][0]
    count=0
    for frame_id, obj in enumerate(objLists):
        # Start tree
        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder")
        folder.text = "JB_Data"
        filename = ET.SubElement(annotation, "filename")
        filename.text = f'frame_{frame_id:06d}.png'

        # Adding image size
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_size[1])
        height = ET.SubElement(size, "height")
        height.text = str(image_size[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(3) # RGB 3 channel

        if obj.shape[1] > 0:
            count += 1
            for id, (bbox, occl) in enumerate(zip(obj['pos'][0],obj['occl'][0])):
                # "p" of vbb: [x, y, w, h]
                p = bbox[0].tolist() 
                xyxy = [int(p[0] - 1), int(p[1] - 1), int(p[0]+p[2]), int(p[1]+p[3])]  # MATLAB is 1-origin
                
                # Adding Object bounding box
                object_tag = ET.SubElement(annotation, "object")
                name = ET.SubElement(object_tag, "name")
                name.text = 'crane'
                
                difficult = ET.SubElement(object_tag, "difficult")
                difficult.text = str(0)
                
                bndbox = ET.SubElement(object_tag, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(xyxy[0])
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(xyxy[1])
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(xyxy[2])
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(xyxy[3])

        # Convert to string with pretty print
        xml_str = ET.tostring(annotation, encoding="utf-8")
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml_as_str = parsed_xml.toprettyxml(indent="    ")
        
        # Remove the XML declaration
        pretty_xml_as_str = '\n'.join(pretty_xml_as_str.splitlines()[1:])
        
        # # Write to file
        output_file = os.path.join(save_dir, f'frame_{frame_id:06d}.xml')
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_xml_as_str)
    
    return count


if __name__ == '__main__':
    image_size=[1080,1920] # [height,width]
    vbbfiles = glob.glob('../data/JB_data/vbb/*.vbb')
    saveDir = '../data/JB_data/xml'
    for vbbfile in vbbfiles:
        nowDir = vbbfile.split('/')[-1].replace('_vbb.vbb', '')
        nowDir = os.path.join(saveDir, nowDir)
        if not os.path.exists(nowDir):
            os.makedirs(nowDir)
        # return the number of bounding box in this vbb file.
        count = convert_vbb(vbbfile, save_dir=nowDir, image_size=image_size)
        print(vbbfile, count)