from WSI_inference_OPENSLIDE_QC.wsi_colors import colors_QC7 as colors
from WSI_inference_OPENSLIDE_QC.wsi_slide_info import slide_info
import torch
import argparse
from openslide import open_slide
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import timeit
import cv2
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS = 1000000000

def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x

def keep_tile(tile, threshold=220, threshold_percent=0.75):
    # keep tile according to mean color
    tile_array = np.array(tile)
    channel_above_threshold = tile_array > threshold
    pixel_above_threshold = np.prod(channel_above_threshold, axis=-1)
    percent_background_pixels = np.sum(pixel_above_threshold) / (tile_array.shape[0] * tile_array.shape[1])
    if percent_background_pixels > threshold_percent:
        return False, None
    else:
        tile_array = tile_array[:, :, :3]
        tile_array = tile_array.reshape(-1, 3)
        average_color = np.mean(tile_array, axis=0)
        # if tile has green larger than red or blue, then it is a background tile
        if average_color[1] > average_color[0] - 10 or average_color[1] > average_color[2] - 10:
            return False, None
        # if average color is approximately gray, then it is a background tile
        if np.abs(average_color[0] - average_color[1]) < 10 and np.abs(average_color[1] - average_color[2]) < 10:
            return False, None
        return True, average_color

def get_label(mask_raw):
    # get the label of patch according to pixel ratios
    unique_values, counts = np.unique(mask_raw, return_counts=True)
    value_count_dict = dict(zip(unique_values, counts))

    other_values = {v: c for v, c in value_count_dict.items() if v not in {0, 1, 7}}

    if other_values:
        most_common_label = max(other_values.keys())
    else:
        valid_labels = {k: v for k, v in value_count_dict.items() if k in {1, 7}}
        if valid_labels:
            most_common_label = max(valid_labels, key=valid_labels.get)
        else:
            most_common_label = 0

    return most_common_label

def slide_process_single(cnt, model, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, colors,
                         ENCODER_MODEL_1, ENCODER_WEIGHTS, DEVICE, BACK_CLASS, MPP_MODEL_1, mpp, w_l0, h_l0, output_patches_dir):
    '''
    Tissue detection map is generated under MPP = 4, therefore model patch size of (512,512) corresponds to tis_det_map patch
    size of (128,128).
    '''

    model_size = (m_p_s, m_p_s)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_1, ENCODER_WEIGHTS)

    os.makedirs(output_patches_dir, exist_ok=True)
    ws, hs, labels = [],[],[]
    # Start loop
    wsi_output_path = os.path.join(output_patches_dir, f'wsi_cnt_{cnt}')
    os.makedirs(wsi_output_path, exist_ok=True)

    for he in tqdm(range(patch_n_h_l0), total=patch_n_h_l0):
        h = he * p_s + 1 if he != 0 else 0
        for wi in range(patch_n_w_l0):
            w = wi * p_s + 1 if wi != 0 else 0

            # Generate patch
            work_patch = slide.read_region((w, h), 0, (p_s, p_s))
            work_patch = work_patch.convert('RGB')
            keep, _ =  keep_tile(work_patch)
            if keep:
                # Resize to model patch size
                work_patch_resized = work_patch.resize((m_p_s, m_p_s), Image.Resampling.LANCZOS)

                image_pre = get_preprocessing(work_patch_resized, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask_raw = np.argmax(predictions, axis=0).astype('int8')
                # Save work_patch with unique filename based on prediction result
                pred_str = str(get_label(mask_raw)) # Convert mask to string for hashing
                unique_filename = f"{h}_{w}_{pred_str}.jpg" 
                patch_path = os.path.join(wsi_output_path, unique_filename)
                work_patch_resized.save(patch_path)

                unique_filename = f"{h}_{w}.png" 
                patch_path = os.path.join(wsi_output_path, unique_filename)
                mask_image = Image.fromarray(mask_raw.astype(np.uint8))
                mask_image.save(patch_path)

                ws.append(w)
                hs.append(h)
                labels.append(np.unique(mask_raw))

    df = pd.DataFrame({'w':ws, 'h':hs, 'labels':labels})

    return df

def get_model_csv(patches_dir):
    ws = []
    hs = []
    labels = []
    image_paths = []
    for subdir, _, files in tqdm(os.walk(patches_dir)):
        wsi_cnt = subdir.split('_')[-1]
        print(subdir)
        for filename in files:
            name, ext = os.path.splitext(filename)
            
            # the file ends up with .jpg is patch image file
            if ext.lower() == '.jpg':
                h = name.split('_')[0]
                w = name.split('_')[1]
                label = name.split('_')[2]
                ws.append(w)
                hs.append(h)
                labels.append(label)
                image_paths.append(os.path.join(subdir,filename))
    df = pd.DataFrame({'wsi_cnt': wsi_cnt, 'w':ws, 'h':hs, 'label':labels, 'image_path':image_paths})
    return df

def main():

    DEVICE = 'cuda'
    # MODEL(S)
    MPP_MODEL = 1
    M_P_S_MODEL = 512
    ENCODER_MODEL = 'timm-efficientnet-b0'
    ENCODER_MODEL_WEIGHTS = 'imagenet'

    # CLASSES
    BACK_CLASS = 7
    start = 10
    end = 12
    model_prim = torch.load(args.model_qc_path, map_location=DEVICE)

    # Read in slide names
    slides_df = pd.read_csv(args.slides_df_path)
    slide_names = slides_df['filepath'].values 
    
    # Start analysis loop
    df_list = []
    for cnt, path_slide in enumerate(slide_names[start:end]):
        cnt = start+cnt
        if not os.path.exists(os.path.join(args.output_patches_dir, f'wsi_cnt_{cnt}')):
            try:
                # Register start time
                start_time = timeit.default_timer()

                print("")
                print("Processing:", path_slide)

                # Open slide
                slide = open_slide(path_slide)

                # GET SLIDE INFO
                p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL, MPP_MODEL)

                '''
                Tissue detection map is generated on MPP = 10
                This map is used for on-fly control of the necessity of model inference.
                Two variants: reduced version with perfect correlation or full version scaled to working MPP of the tumor detection model
                Classes: 0 - tissue, 1 - background
                '''

                df = slide_process_single(cnt, model_prim, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                                    M_P_S_MODEL, colors, ENCODER_MODEL,
                                                    ENCODER_MODEL_WEIGHTS, DEVICE, BACK_CLASS, MPP_MODEL, mpp, w_l0, h_l0,
                                                    args.output_patches_dir)
                df_list.append(df)

            except Exception as e:
                print(f"There was some problem with the slide. The error is: {e}")

    os.makedirs(args.df_dir, exist_ok=True)
    df_all = pd.concat(df_list, axis=0)
    df_all.to_csv(os.path.join(args.df_dir,f'anns_model_{start}_{end}.csv'), index=False)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='extract patches from wsi using segmentation model')
    parser.add_argument('--model_qc_path', default='', type=str, help='grandqc model path')
    parser.add_argument('--output_patches_dir', default='', type=str, help='directory to store patches')
    parser.add_argument('--slides_df_path', default='', type=str, help='path of the dataframe that stores paths of slides')
    parser.add_argument('--df_dir', default='', type=str, help='directory of dataframes')
    args = parser.parse_args()

    main(args)

    # scan directory to get availabel patches
    output_df_path = os.path.join(args.df_dir,f'anns_model.csv')
    if not os.path.exists(output_df_path):
        df = get_model_csv(args.output_patches_dir)
        df.to_csv(output_df_path, index=False)