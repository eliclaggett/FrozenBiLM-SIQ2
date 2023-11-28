import json
import pandas as pd
import pickle
import torch

DATA_DIR = '/home/eli/code/datasets/SIQ2'

split_fn = {
    'train': 'qa_train_dropped.json',
    'val': 'qa_val_dropped.json',
    'test': 'qa_test_droppeds.json',
}

with open(f"{DATA_DIR}/qa/qa_val_dropped.json") as f:
    data = [json.loads(line) for line in f]

splits = ["train", "val", "test"]
for split in splits:
    with open(f"{DATA_DIR}/qa/qa_{split}_dropped.json") as f:
        data = [json.loads(line) for line in f]
    video_id = [x["vid_name"] for x in data]
    start = [float(x["ts"].split("-")[0]) for x in data]
    end = [float(x["ts"].split("-")[1]) for x in data]
    a0 = [
        x["a0"].strip()[:-1] if x["a0"].strip()[-1] == "." else x["a0"].strip()
        for x in data
    ]
    a1 = [
        x["a1"].strip()[:-1] if x["a1"].strip()[-1] == "." else x["a1"].strip()
        for x in data
    ]
    a2 = [
        x["a2"].strip()[:-1] if x["a2"].strip()[-1] == "." else x["a2"].strip()
        for x in data
    ]
    a3 = [
        x["a3"].strip()[:-1] if x["a3"].strip()[-1] == "." else x["a3"].strip()
        for x in data
    ]
    question = [x["q"] for x in data]
    qid = [x["qid"] for x in data]
    if split != "test":
        answer_id = [x["answer_idx"] for x in data]
        df = pd.DataFrame(
            {
                "qid": qid,
                "video_id": video_id,
                "start": start,
                "end": end,
                "question": question,
                "a0": a0,
                "a1": a1,
                "a2": a2,
                "a3": a3,
                "answer_id": answer_id,
            },
            columns=[
                "qid",
                "video_id",
                "start",
                "end",
                "question",
                "a0",
                "a1",
                "a2",
                "a3",
                "answer_id",
            ],
        )
    else:
        df = pd.DataFrame(
            {
                "qid": qid,
                "video_id": video_id,
                "start": start,
                "end": end,
                "question": question,
                "a0": a0,
                "a1": a1,
                "a2": a2,
                "a3": a3
            },
            columns=[
                "qid",
                "video_id",
                "start",
                "end",
                "question",
                "a0",
                "a1",
                "a2",
                "a3"
            ],
        )
    print(len(df))
    df.to_csv(f"{DATA_DIR}/{split}.csv", index=False)
