import torch
import clip
from clip import tokenize
import glob, json
import cv2
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transforms = clip.load("ViT-L/14@336px", device=device)
# model, transforms = load_model("ViT_B_32", pretrained=True)
print("model loaded")
en_match_words = {
    "scerario": ["suburbs", "city street", "expressway", "tunnel", "parking-lot", "gas or charging stations",
                 "unknown"],
    "weather": ["clear", "cloudy", "raining", "foggy", "snowy", "unknown"],
    "period": ["daytime", "dawn or dusk", "night", "unknown"],
    "road_structure": ["normal", "crossroads", "T-junction", "ramp", "lane merging", "parking lot entrance",
                       "round about", "unknown"],
    "general_obstacle": ["nothing", "speed bumper", "traffic cone", "water horse", "stone", "manhole cover", "nothing",
                         "unknown"],
    "abnormal_condition": ["uneven", "oil or water stain", "standing water", "cracked", "nothing", "unknown"],
    "ego_car_behavior": ["slow down", "go straight", "turn right", "turn left", "stop", "U-turn", "speed up",
                         "lane change", "others"],
    "closest_participants_type": ["passenger car", "bus", "truck", "pedestrain", "policeman", "nothing", "others",
                                  "unknown"],
    "closest_participants_behavior": ["slow down", "go straight", "turn right", "turn left", "stop", "U-turn",
                                      "speed up", "lane change", "others"],
}

submit_json = {
    "author": "AKonjac_",
    "time": "231116",
    "model": "clip ViT_L/14@336px",
    "test_results": []
}

paths = glob.glob('./test/*')
paths.sort()


def get_image(Cap):

    '''
    tmp = []
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        tmp.append(frame)
        cnt += 1
    fr = int(cnt / 2)
    img = tmp[fr]
    '''

    img = Cap.read()[1]
    __image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    __image = Image.fromarray(__image)
    __image = transforms(__image).unsqueeze(0).to(device)
    return __image


for video_path in paths:
    print(video_path)
    clip_id = video_path.split('/')[-1]
    single_video_result = {
        "clip_id": clip_id,
        "scerario": "city street",
        "weather": "clear",
        "period": "daytime",
        "road_structure": "normal",
        "general_obstacle": "nothing",
        "abnormal_condition": "nothing",
        "ego_car_behavior": "go straight",
        "closest_participants_type": "passenger car",
        "closest_participants_behavior": "go straight"
    }

    for keyword in en_match_words.keys():
        if keyword not in ["weather", "period"]:
            continue
        cap = cv2.VideoCapture(video_path)
        texts = np.array(en_match_words[keyword])

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, tokenize(en_match_words[keyword]).to(device))
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        cnt = 0
        tot = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break

            cnt += 1

            if cnt == 20:
                cnt = 0
                tot += 1
                if tot >= 10:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = transforms(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_per_image, logits_per_text = model(image, tokenize(en_match_words[keyword]).to(device))
                    probs = (probs + logits_per_image.softmax(dim=-1).cpu().numpy())/2
                    print(probs)

        single_video_result[keyword] = texts[probs[0].argsort()[::-1][0]]

    submit_json["test_results"].append(single_video_result)

with open('clip_result_1.json', 'w', encoding='utf-8') as up:
    json.dump(submit_json, up, ensure_ascii=False)
