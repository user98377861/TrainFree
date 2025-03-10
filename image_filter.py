from data import BinaryDataset, ArtifactDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from tqdm import tqdm

import argparse
import torch
import time
import clip
import os
import pandas as pd 

def get_data(args, model, bs, preprocess, split='ref'):
    if split=='ref':
        df1_ref = pd.read_csv(args.df1_ref_path)
        df2_ref = pd.read_csv(args.df2_ref_path)
        dst = BinaryDataset(dataframe1=df1_ref, dataframe2=df2_ref, transform = preprocess)
    else:
        df_test = pd.read_csv(args.df_test_path)
        dst = ArtifactDataset(dataframe=df_test, transform = preprocess)
    dst_loader = DataLoader(dst, batch_size=bs, shuffle=False, num_workers=8, drop_last=False)
    
    feats = get_features(model, dst_loader)
    labels = torch.tensor(df['label'].tolist()).cuda()
    return feats, labels

def get_features(model, loader, model_name='clip'):
    features = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader)):
            images = images.cuda()
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)

    features = torch.cat(features)
    return features

def reference_set_sims_vote(feats_ref, labels_ref, feats_test, labels_test, organ='total', num_votes=3, output_dict=True):
    start_time = time.time()
    bs = 500
    splits = int(feats_test.shape[0]//bs)+1
    sims = []
    for i in range(splits):
        if i!=splits-1:
            feats_test_tp = feats_test[i*bs: (i+1)*bs]
        else:
            feats_test_tp = feats_test[i*bs: ]
        tp = feats_test_tp @ feats_ref.T
        sims.append(tp.cpu())
        del feats_test_tp, tp
        torch.cuda.empty_cache()

    sims = torch.cat(sims,dim=0)

    topk_values, topk_indices = torch.topk(sims, k=num_votes, dim=1)
    topk_labels = labels_ref[topk_indices]
    
    predicted_labels = []
    for i in range(len(feats_test)):
        sample_topk_labels = topk_labels[i].tolist()
        label_counts = Counter(sample_topk_labels)
        most_common_label, _ = label_counts.most_common(1)[0]
        predicted_labels.append(most_common_label)

    preds = torch.tensor(predicted_labels)
    labels_test_cpu = labels_test.cpu()

    # 1. multiclass
    report = classification_report(labels_test_cpu.numpy(), preds.numpy(), output_dict=output_dict)
    acc = accuracy_score(labels_test_cpu.numpy(), preds.numpy())
    tu = time.time()-start_time

    return  pd.DataFrame([{
                'acc': acc,
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                
                'vote_num':num_votes,
                'time_usage': tu,
                'organ':organ,
            }]), preds


def main(args):
    model, preprocess = clip.load('ViT-B/16')
    model.eval()
    bs = args.batch_size

    feats_ref, labels_ref = get_data(args, model, bs, preprocess, split='ref')
    feats_test, labels_test = get_data(args, model, bs, preprocess, split='test')

    result, preds = reference_set_sims_vote(feats_ref, labels_ref, feats_test, labels_test, organ='total', num_votes=3, output_dict=True)

    df_test = pd.read_csv(args.df_test_path)
    df_test['preds']=preds.numpy()
    df_test.to_csv(os.path.join(args.output_dir, 'df_res.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='process of artifact classification')
    parser.add_argument('--df1_ref_path', default='', type=str, help='pathological dataframe path of reference set')
    parser.add_argument('--df2_ref_path', default='', type=str, help='non-pathological dataframe path of reference set')
    parser.add_argument('--df_test_path', default='', type=str, help='dataframe path of test set')
    parser.add_argument('--output_dir', default='', type=str, help='dataframe path of test set')
    parser.add_argument('--batch_size', default=400, type=int, help='batch_size')
    args = parser.parse_args()

    main(args)