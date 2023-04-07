import os


def preidct(file_path):
    file_emotion = []
    name = file_path.split("_")
    ele = name[1].split(".")[0]
    if ele.startswith("a"):
        file_emotion.append("angry")
    elif ele.startswith("d"):
        file_emotion.append("disgust")
    elif ele.startswith("f"):
        file_emotion.append("fear")
    elif ele.startswith("h"):
        file_emotion.append("happy")
    elif ele.startswith("n"):
        file_emotion.append("neutral")
    elif ele.startswith("sa"):
        file_emotion.append("sad")
    else:
        file_emotion.append("angry")
    return file_emotion[0]
