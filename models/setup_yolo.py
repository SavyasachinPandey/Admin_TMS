#!/usr/bin/env python3
"""Download YOLO model files"""
import os
import urllib.request

def setup_yolo():
    """Download YOLO files"""
    os.makedirs('models', exist_ok=True)
    
    files = {
        'models/yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'models/coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
        'models/yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights'
    }
    
    for filepath, url in files.items():
        if not os.path.exists(filepath):
            print(f"⬇️  Downloading {os.path.basename(filepath)}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded {os.path.basename(filepath)}")
            except Exception as e:
                print(f"❌ Failed: {e}")
        else:
            print(f"✅ {os.path.basename(filepath)} exists")

if __name__ == "__main__":
    setup_yolo()
