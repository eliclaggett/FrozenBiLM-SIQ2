import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import pickle
import math


class MC_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        subtitles_path,
        features_path_video,
        features_path_face=None,
        features_path_pose=None,
        max_feats=10,
        features_dim_video=768,
        features_dim_face=35,
        features_dim_pose=768,
        tokenizer=None,
        use_context=True,
        type_map=None,
        prefix="",
        suffix="",
    ):
        self.data = pd.read_csv(csv_path)
        if subtitles_path:
            self.subs = pickle.load(open(subtitles_path, "rb"))
        else:
            self.subs = None
        self.features = th.load(features_path_video)
        if features_path_face:
            self.features_face = th.load(features_path_face)
        else:
            self.features_face = {}
        if features_path_pose:
            self.features_pose = th.load(features_path_pose)
        else:
            self.features_pose = {}
        self.max_feats = max_feats
        self.features_dim = features_dim_video
        self.features_dim_face = features_dim_face
        self.features_dim_pose = features_dim_pose
        self.mask = tokenizer.mask_token if tokenizer is not None else None
        self.use_context = use_context
        mc = 0
        while f"a{mc}" in self.data:
            mc += 1
        self.mc = mc
        self.type_map = type_map
        self.prefix = prefix
        self.suffix = suffix

    def __len__(self):
        return len(self.data)

    def _get_subtitles(self, video_id, start, end):
        # only consider subtitles that intersec with the timestamps of the video clip
        # print(self.subs[video_id])
        subs_list = [
            x["text"]
            for x in self.subs[video_id]
            if x["end"] >= start and x["start"] <= end
        ]
        return " ".join(subs_list).capitalize().strip()

    def _get_text(self, subtitles, answer, mask, question=None):
        text = (
            f"{self.prefix} Question: {question} Is it '{answer}'? {mask}{self.suffix}"
        )
        if self.use_context:
            text += f" Subtitles: {subtitles}"
        text = text.strip()
        return text

    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            # print(video_id)
            video = th.zeros(1, self.features_dim)
        else:
            if start is not None and not math.isnan(start):
                video = self.features[video_id][int(start) : int(end) + 1].float()
            else:
                video = self.features[video_id].float()
            if not len(video):
                # print(video_id, start, end)
                video = th.zeros(1, self.features_dim)
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, video_len
    
    def _get_face(self, video_id):
        if video_id not in self.features_face:
            face = th.zeros(1, self.features_dim_face)
        else:
            face = self.features_face[video_id].float()
            if not len(face):
                face = th.zeros(1, self.features_dim_face)
        if len(face) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(face[(j * len(face)) // self.max_feats])
            face = th.stack(sampled)
            face_len = self.max_feats
        elif len(face) < self.max_feats:
            face_len = len(face)
            face = th.cat(
                [face, th.zeros(self.max_feats - face_len, self.features_dim_face)], 0
            )
        else:
            face_len = self.max_feats

        return face, face_len
    
    def _get_pose(self, video_id):
        if video_id not in self.features_pose:
            pose = th.zeros(1, self.features_dim_pose)
        else:
            pose = self.features_pose[video_id].float()
            if not len(pose):
                pose = th.zeros(1, self.features_dim_pose)
        if len(pose) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(pose[(j * len(pose)) // self.max_feats])
            pose = th.stack(sampled)
            pose_len = self.max_feats
        elif len(pose) < self.max_feats:
            pose_len = len(pose)
            pose = th.cat(
                [pose, th.zeros(self.max_feats - pose_len, self.features_dim_pose)], 0
            )
        else:
            pose_len = self.max_feats

        return pose, pose_len

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]

        # get start, end
        start = self.data["start"].values[idx]
        end = self.data["end"].values[idx]

        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get subs
        if self.subs:
            subs = self._get_subtitles(video_id, start, end)
        else:
            subs = ""

        # get features
        video, video_len = self._get_video(video_id, start, end)
        face, face_len = self._get_face(video_id)
        pose, pose_len = self._get_pose(video_id)
        # face, face_len = None, None
        # pose, pose_len = None, None
        # get answer id
        answer_id = -1  # for hidden set testing
        if "answer_id" in self.data:
            answer_id = self.data["answer_id"].values[idx]

        text = []
        for i in range(self.mc):
            ai = self.data[f"a{i}"].values[idx]
            text.append(self._get_text(subs, ai, self.mask, question))

        qid = idx
        if "qid" in self.data:
            # qid = int(self.data["qid"].values[idx])
            qid = self.data["qid"].values[idx]

        return {
            "video": video,
            "video_len": video_len,
            "face": face,
            "face_len": face_len,
            "pose": pose,
            "pose_len": pose_len,
            "text": text,
            "qid": qid,
            "answer_id": answer_id,
            "type": type,
        }


def mc_collate_fn(batch):
    bs = len(batch)
    
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    
    face = th.stack([batch[i]["face"] for i in range(bs)])
    face_len = th.tensor([batch[i]["face_len"] for i in range(bs)], dtype=th.long)
    
    pose = th.stack([batch[i]["pose"] for i in range(bs)])
    pose_len = th.tensor([batch[i]["pose_len"] for i in range(bs)], dtype=th.long)
    
    text = [
        [batch[i]["text"][j] for i in range(bs)] for j in range(len(batch[0]["text"]))
    ]
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]

    return {
        "video": video,
        "video_len": video_len,
        'face': face,
        'face_len': face_len,
        'pose': pose,
        'pose_len': pose_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
    }


def build_mc_dataset(dataset_name, split, args, tokenizer):
    type_map = None
    if dataset_name == "how2qa":
        if split == "train":
            csv_path = args.how2qa_train_csv_path
        elif split == "val":
            csv_path = args.how2qa_val_csv_path
        elif split == "test":
            csv_path = args.how2qa_val_csv_path  # eval on val public
        else:
            raise NotImplementedError
        subtitles_path = args.how2qa_subtitles_path
        features_path = args.how2qa_features_path
    elif dataset_name == "tvqa":
        if split == "train":
            csv_path = args.tvqa_train_csv_path
        elif split == "val":
            csv_path = args.tvqa_val_csv_path
        elif split == "test":
            csv_path = args.tvqa_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.tvqa_subtitles_path
        features_path = args.tvqa_features_path
    elif dataset_name == "siq2":
        if split == "train":
            csv_path = args.siq2_train_csv_path
        elif split == "val":
            csv_path = args.siq2_val_csv_path
        elif split == "test":
            csv_path = args.siq2_test_csv_path
        else:
            raise NotImplementedError
        subtitles_path = args.siq2_subtitles_path
        features_path = args.siq2_features_path
    else:
        raise NotImplementedError
    return MC_Dataset(
        csv_path=csv_path,
        subtitles_path=subtitles_path,
        features_path_video=features_path,
        features_path_face=None,
        features_path_pose=None,
        max_feats=args.max_feats,
        features_dim_video=args.features_dim_video,
        features_dim_face=args.features_dim_face,
        features_dim_pose=args.features_dim_pose,
        tokenizer=tokenizer,
        use_context=args.use_context,
        prefix=args.prefix,
        suffix=args.suffix,
        type_map=type_map,
    )
