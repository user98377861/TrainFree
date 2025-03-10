from data import ArtifactDataset
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

def get_data(df, model, bs, preprocess):
    dst = ArtifactDataset(dataframe=df, transform = preprocess)
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
    report_multi = classification_report(labels_test_cpu.numpy(), preds.numpy(), output_dict=output_dict)
    acc_multi = accuracy_score(labels_test_cpu.numpy(), preds.numpy())
    tu = time.time()-start_time
    
    # 2. binary
    start_time = time.time()
    topk_labels_modifed = torch.where(topk_labels.cpu() != 0, torch.tensor(1), topk_labels.cpu())
    predicted_labels = []
    for i in range(len(feats_test)):
        sample_topk_labels = topk_labels_modifed[i].tolist()
        label_counts = Counter(sample_topk_labels)
        most_common_label, _ = label_counts.most_common(1)[0]
        predicted_labels.append(most_common_label)

    preds_modified = torch.tensor(predicted_labels)
    labels_test_modified = torch.where(labels_test_cpu != 0, torch.tensor(1), labels_test_cpu)

    report_binary = classification_report(labels_test_modified.numpy(), preds_modified.numpy(), target_names=['Class 0', 'Class 1'], output_dict=output_dict)
    acc_binary = accuracy_score(labels_test_modified.numpy(), preds_modified.numpy())
    tu = time.time()-start_time

    return  pd.DataFrame([{
                'acc_binary': acc_binary,
                'macro_precision_binary': report_binary['macro avg']['precision'],
                'macro_recall_binary': report_binary['macro avg']['recall'],
                'macro_f1_binary': report_binary['macro avg']['f1-score'],

                'acc_multi': acc_multi,
                'macro_precision_multi': report_multi['macro avg']['precision'],
                'macro_recall_multi': report_multi['macro avg']['recall'],
                'macro_f1_multi': report_multi['macro avg']['f1-score'],
                
                'vote_num':num_votes,
                'time_usage': tu,
                'organ':organ,
            }]), preds, preds_modified


def main(args):
    model, preprocess = clip.load('ViT-B/16')
    model.eval()
    bs = args.batch_size

    df_ref = pd.read_csv(args.df_ref_path)
    feats_ref, labels_ref = get_data(df_ref, model, bs, preprocess)

    df_test = pd.read_csv(args.df_test_path)
    feats_test, labels_test = get_data(df_test, model, bs, preprocess)

    result, preds_multi, preds_binary = reference_set_sims_vote(feats_ref, labels_ref, feats_test, labels_test, organ='total', num_votes=3, output_dict=True)
    
    df_test['preds_multi']=preds_multi.numpy()
    df_test['preds_binary']=preds_binary.numpy()
    df_test.to_csv(os.path.join(args.output_dir, 'df_res.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='process of artifact classification')
    parser.add_argument('--df_ref_path', default='', type=str, help='dataframe path of reference set')
    parser.add_argument('--df_test_path', default='', type=str, help='dataframe path of test set')
    parser.add_argument('--output_dir', default='', type=str, help='dataframe path of test set')
    parser.add_argument('--batch_size', default=400, type=int, help='batch_size')
    args = parser.parse_args()

    main(args)