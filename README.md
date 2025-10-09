# computer-vision-projects

A collection of deep learning and computer vision projects, showcasing applications of convolutional neural networks (CNNs), generative adversarial networks (GANs), and object detection architectures using **PyTorch**.

Projects by **Emilia Solari del Sol**:

| Project | Description | Key Techniques |
|----------|--------------|----------------|
| **CNN Classification** | Trained a custom Inception-inspired CNN from scratch for CIFAR-10 image classification. | Convolutional Networks, Inception Modules, AdamW, Cosine LR |
| **Faster R-CNN** | Object detection with transfer learning and configurable optimization strategies. | Transfer Learning, Feature Pyramid Network, mAP Evaluation |
| **Pix2Pix** | Paired image-to-image translation using a Conditional GAN (Edges → Shoes). | U-Net Generator, PatchGAN Discriminator, L1 Loss |
| **CycleGAN** | Unpaired image translation (Selfie ↔ Anime) with cycle-consistency learning. | Cycle-Consistency, Identity Loss, Dual GANs |

---
## Repository structure
- `cnn-classification/`
- `faster-rcnn/`
- `pix2pix/`
- `cyclegan/`

  Each folder includes:
- `README.md` — full project overview  
- `.ipynb` — Jupyter Notebook implementation  
- `/assets/` — visual outputs and performance plots  

Install deps and run locally:
```bash
pip install -r requirements.txt
jupyter notebook
