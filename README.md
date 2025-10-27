# CE-Reporter

### [GitHub Repository](https://github.com/CUHK-AIM-Group/CE-Reporter)


<!-- ### [arXiv]() -->
> [Towards Expert-Level Generative AI for Capsule Endoscopy Video-to-Report Generation](https://github.com/CUHK-AIM-Group/CE-Reporter) \
> [Boyun Zheng](https://scholar.google.com/citations?user=ZveKOXkAAAAJ&hl=zh-CN), [Yu Jiang](https://scholar.google.com/citations?user=P9_bgvEAAAAJ&hl=en), Dejun Fan, [Xinyu Liu](https://xinyuliu-jeffrey.github.io/), [Xiaoqing Guo](https://guo-xiaoqing.github.io/), [Romain Hardy](https://scholar.google.com/citations?user=j7tIdWcAAAAJ&hl=en), [Shengyuan Liu](https://scholar.google.com/citations?user=zP6fRqcAAAAJ), Bin Chen, Mingjie Wang, Jinni Luo, Jin Tao<sup>✉</sup>, Jiancong Hu<sup>✉</sup>, [Lichao Sun](https://lichao-sun.github.io/), [Pranav Rajpurkar](https://pranavrajpurkar.com/), [Lei Xing](https://profiles.stanford.edu/lei-xing), [Yixuan Yuan](https://www.ee.cuhk.edu.hk/~yxyuan/people/people.htm)<sup>✉</sup>

![introduction](assets/introduction.jpg)

## 📄 Introduction

Gastrointestinal (GI) diseases remain a significant global health burden due to elevated morbidity and mortality. Accurate and timely diagnosis, especially for small bowel conditions, remains challenging due to the complexity and lengthy duration (6-12 hours) of capsule endoscopy (CE) videos, which require approximately 1.5 hours of manual review. While artificial intelligence (AI) systems have enhanced CE analysis, seamlessly integrated AI-driven diagnostic report generation, a key clinical need, remains largely unexplored. Here we introduce CE Reporter, an explainable video-to-report system that transforms lengthy CE videos into evidence-supported diagnostic reports via multi-task keyframe detection and time-aware vision-language alignment. CE Reporter was trained on our established MICS-CE dataset, the first large-scale resource for automated CE reporting, which comprises 109 million frames (11,028 hours of video) and 29,338 expert-annotated keyframe-text pairs derived from 1,002 video-report pairs collected at four clinical centres using three different capsule devices. The model demonstrated superior performance across three dimensions: (1) Report quality: exceeding existing multimodal large language models in linguistic coherence and diagnostic accuracy (BLEU score: 0.573 vs. 0.523 for the best baseline; multi-lesion detection F1-score: 0.82 vs. 0.72); (2) Clinician evaluation: matching expert-level performance in correctness, completeness, and conciseness, outperforming resident clinicians in controlled comparisons while reducing reporting time by 91.9\%; (3) Clinical utility: explainable reports supported by visual keyframes and robust performance across centers and devices (F1-score: 0.72–0.81). These findings provide the first practical solution for CE video-to-report generation, highlighting the potential of AI to improve the efficiency and transparency of digestive-disease screening at scale.

## ⚙️ Setup

### Environment Setup
Create the conda environment for keyframe detection module:
```bash
conda create -n ce-reporter-kfd python=3.8 -y
conda activate ce-reporter-kfd
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install --no-deps paddlepaddle-gpu==2.6.0
conda install pillow==10.4.0
pip install -r requirements.txt
```
Note: If you encounter issues finding libcudnn.so, add the cuDNN lib directory path to the LD_LIBRARY_PATH environment variable. You can find the path by running: 
```bash
python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)". 
```
As a last resort, create a symbolic link, e.g., ln -s /xxx/miniconda3/envs/ce-reporter-s1/lib/python3.8/site-packages/nvidia/cudnn/lib/libcudnn.so.9 libcudnn.so. 

Most environment issues are related to paddlepaddle; please refer to paddlepaddle documentation (https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html) for troubleshooting. 

Then create the second conda environment (based on LLaVA (https://github.com/haotian-liu/LLaVA)):
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
conda create -n ce-reporter-llava python=3.10 -y
conda activate ce-reporter-llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn==2.6.3 --no-build-isolation
git pull
pip install -e .
pip install peft==0.12.0
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
```
If flash-attn installation fails, visit https://github.com/Dao-AILab/flash-attention/releases/tag for a suitable version.

Create the third lightweight conda environment for Llama:
```bash
conda create -n ce-reporter-llama python=3.9 -y
conda activate ce-reporter-llama
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 peft==0.14.0 trl==0.14.0 pandas==2.2.3 datasets==3.2.0 tokenizers==0.20.3 huggingface_hub modelscope
cd ../llama
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 📚 Data Preparation
The data supporting the findings of this study are divided into publicly available and restricted datasets. The datasets used for training the feature encoder are publicly accessible as follows:
- Kvasir-capsule is available at https://datasets.simula.no/kvasir-capsule
- SEE-AI is available at https://www.kaggle.com/datasets/capsuleyolo/kyucapsule
- AIIMS is available at https://misahub.in/cv2024.html
- WCE-bleeding is available at https://zenodo.org/records/10156571
- KID is available at https://mdss.uth.gr/datasets/endoscopy/kid
</details>

De-identified data from the SSH cohort, including paired video-report data and paired keyframe-text data, are subject to restricted access due to privacy and ethical considerations. These data are available for research purposes upon reasonable request to the corresponding author. Requests will be evaluated in accordance with institutional policies, ethical standards, and applicable legal requirements to ensure compliance with data privacy obligations. Responses to requests will typically be provided within one month.
Note: Our paper is currently under review. Upon acceptance of the paper, the data will be available for request. 

### The expected data structure is as follows:
```
CE-Reporter/
├── vce_data/
│   ├── video/                      # Raw video files
│   ├── frames/                     # Extracted frames
│   ├── annotation/                 # Annotation JSON files
│   ├── llava/
│   │   ├── imgs/                   # Images for LLaVA training
│   │   └── llava_annotations.json  # LLaVA training JSON
│   ├── llama/
│   │   └── llama_annotations.json  # Llama training JSON
│   └── pretrained_weight/          # Preprocessing required model weights
├── keyframe_detection/
│   └── model_weight/               # weights for keyframe detection
├── LLaVA/
│   └── CE-Reporter-llava-1.5-7b   # LLaVA weights
├── llama/
│   └── model_weight/          
│       └── CE-Reporter-llama-3.1-8b # Llama weights 
```

### Model Weights
Our model files (three parts) are open-sourced on Hugging Face:

> **Note:** These model weights are currently hosted privately while our manuscript is under review. We will make them fully public, along with the necessary code to reproduce our results, immediately upon the paper's acceptance.

| Component | Hugging Face Link |
|-----------|-------------------|
| Keyframe Detection Model | https://huggingface.co/Byzzz0301/CE-Reporter-kfd |
| LLaVA Captioning Model | https://huggingface.co/Byzzz0301/CE-Reporter-llava-1.5-7b |
| Llama Report Generation Model | https://huggingface.co/Byzzz0301/CE-Reporter-llama-3.1-8b |

## ⏳ Feature Generation

Generate frames from videos:
``` bash
conda activate ce-reporter-kfd
cd preprocess
python Video2frame.py --video_root ../vce_data/video --frame_root ../vce_data/frames
``` 

Generate features including swin_feature, EndoViT_feature (optional), and labels npy (optional):
``` bash
python extract_swin_endo_labels.py
``` 

Generate SEResnet features for inter-frame semantic similarity and create cluster.json for redundancy removal:
``` bash
python extract_se_features_and_cluster.py
``` 

## 🚀 Inference
### Keyframe Detection:
Place the downloaded first-stage model from Hugging Face into keyframe_detection/pretrained_weight.
``` bash
conda activate ce-reporter-kfd
cd keyframe_detection
python inference.py
``` 
Extracted keyframes are saved in keyframe_detection/extracted_keyframe. Generate JSON for LLaVA:
``` bash
python img_to_llava_json.py
``` 
### LLaVA:
``` bash
conda activate ce-reporter-llava
cd LLaVA/scripts
sh test.sh
``` 
### Llama:
Aggregate captions and generate summary report:
``` bash
cd llama
conda activate ce-reporter-llama
python merge_caption.py
python inference.py
``` 

## 📈 Training
### Keyframe Detection:
Augment keyframes:
``` bash
conda activate ce-reporter-kfd
cd preprocess
python augment_keyframe.py
cd ../keyframe_detection
python train.py
``` 
### LLaVA:
``` bash
conda activate ce-reporter-llava
cd LLaVA/scripts
sh finetune_lora.sh
``` 
### Llama:
Aggregate captions and generate summary report:
``` bash
conda activate ce-reporter-llama
cd llama
python finetune_lora.py
``` 

## 📏 Evaluation
- TODO: Implement evaluation metrics for report quality (BLEU, F1-score).

---
## 🎈 Acknowledgements
Some source code of ours is borrowed from [LLaVA](https://github.com/haotian-liu/LLaVA), [DeepSeek-R1-Distill-Llama-8B-LoRA](https://github.com/sunshine-JLU/deepseek-r1-distill-llama-8b-lora). Thanks for their contributions. 


<!-- ## 📜 Citation
If you find this repository/work helpful in your research, welcome to cite this paper and give a ⭐. 
```

``` -->
