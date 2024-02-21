import cv2
import json
import glob
with open("./clip_result_5.json", "r", encoding="utf-8") as f:
    tmp = f.read()
lst = json.loads(tmp)["test_results"]
paths = glob.glob('./test/*')
paths.sort()

def img_to_GRAY(i, img):
    # 把图片转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取灰度图矩阵的行数和列数
    r, c = gray_img.shape[:2]
    piexs_sum = r * c  # 整个图的像素个数
    # 遍历灰度图的所有像素
    # 灰度值小于60被认为是黑
    dark_points = (gray_img < 60)
    target_array = gray_img[dark_points]
    dark_sum = target_array.size  # 偏暗的像素
    dark_prop = dark_sum / (piexs_sum)  # 偏暗像素所占比例
    # print(str(i) + " " + str(dark_prop))
    if dark_prop >= 0.50:  # 若偏暗像素所占比例超过0.6,认为为整体环境黑暗的图片
        return "night"
    elif dark_prop >= 0.40:
        return "dawn or dusk"
    else:
        return "daytime"

i = 0
for video_path in paths:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    a = lst[i]
    a["period"] = img_to_GRAY(i, frame)
    if a["period"] == "night":
        a["weather"] = "clear"
    if a["scerario"] == "parking-lot":
        a["general_obstacle"] = "speed bumper"
        a["weather"] = "unknown"
    if a["closest_participants_type"] == "bus":
        a["scerario"] = "city street"
    if a["scerario"] == "city road":
        a["scerario"] = "city street"
    if a["scerario"] == "express-way":
        a["scerario"] = "expressway"
    i += 1

submit_json = {
    "author": "AKonjac_",
    "time": "231116",
    "model": "clip ViT_L/14@336px",
    "test_results": []
}

submit_json["test_results"] = submit_json["test_results"] + lst
with open('clip_result_7.json', 'w', encoding='utf-8') as up:
    json.dump(submit_json, up, ensure_ascii=False)


'''
0 0.0635816936728395
1 0.06307532793209876
2 0.25416859567901234
3 0.2340552662037037
4 0.02212914737654321
5 0.24169849537037036
6 0.1255589313271605
7 0.8683097029320987
8 0.08415943287037037
9 0.11473138503086419
10 0.10923707561728395
11 0.00010223765432098765
12 0.03038917824074074
13 0.16921296296296295
14 0.17043065200617283
15 0.8624040316358025
16 0.15236014660493827
17 0.15334828317901233
18 0.1806621334876543
19 0.1604803240740741
20 0.1173461612654321
21 0.025165895061728394
22 0.7173369984567901
23 0.6243942901234568
24 0.8234013310185185
25 0.5586887538580247
26 0.7505025077160494
27 0.6745871913580247
28 0.7265190972222222
29 0.7652309992283951

'''
