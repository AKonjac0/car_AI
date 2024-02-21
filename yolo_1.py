import Yolo_v8 as yl
import json
import glob

with open("./clip_result_10.json", "r", encoding="utf-8") as f:
    tmp = f.read()
lst = json.loads(tmp)["test_results"]
paths = glob.glob('./test/*')
paths.sort()

for i in range(0, 30):
    print(i)
    video_path = paths[i]
    a = lst[i]
    a["closest_participants_type"] = yl.yolo(video_path)


submit_json = {
    "author": "AKonjac_",
    "time": "231116",
    "model": "clip ViT_L/14@336px",
    "test_results": []
}

submit_json["test_results"] = submit_json["test_results"] + lst
with open('clip_result_5.json', 'w', encoding='utf-8') as up:
    json.dump(submit_json, up, ensure_ascii=False)


