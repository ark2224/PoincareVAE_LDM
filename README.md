# **PoincareVAE_LDM**

Latent diffusion models have taken the world of computer vision by storm. With
state-of-the-art image synthesis, LDMs(such as Stable Diffusion), generate high-
fidelity images from corresponding text prompts. Typical LDMs, such as Stable
Diffusion[4], comprise of three portions:
• A Varianational Auto-encoder(VAE)
• A U-Net with a ResNet backbone
• An optional Text-Encoder
The VAE is what maps images from the pixel-space to a latent-space, upon
which the bulk of the operations pertaining to image synthesis is performed.
While the Diffusion model is currently at the state of the art, there may be room
for further improvement. This project seeks to explore one direction for further
improvement by utilizing Hyperbolic embeddings. Mathieu et al[3] showed that
continuous hierarchical representations can be captured by Poincare variational
autoencoders, and these models often provide better generalization than their
Euclidean counterparts. As images often have an implicit sense of a hierarchical
structure to them, we believe that it may be possible to improve Stable Diffusion
by ensuring that its latent space representation is embedded in hyperbolic space.
Thus, this project seeks to systematically replace Euclidean embeddings with
Hyperbolic embeddings(wherever it may be justified), to improve performance
the of Stable Diffusion.


## **Roadmap**

The starting point of the project will be to recreate the setup of Stable Diffu-
sion. For simplicity, we will use the MNIST Handwritten digit-dataset[2]. Once
we recreate the setup of stable-diffusion, we will then systematically utilize the
findings of Khrulkov et al[1](along with their code-base), to implement Hyper-
bolic layers. Finally, we will utilize Mathieu et al’s finding to assimilate the
works of Khrulkov et al into the architecture of Stable Diffusion.[1]


## **Further points of study**

This is a herculean task we have undertaken, to truly see if our hypothesis is
correct, we will need to train our models on massive amount of data. Curating
this data will then form the bulk of efforts. At the moment, we are not quite
sure how we will overcome this task, so we refuse to point to any one particular
dataset. Thus, the further points of study will be decided by the meetings with
have with Professor Turkcan.


## **References**

1. Valentin Khrulkov et al. “Hyperbolic Image Embeddings”. In: The IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). June
2020.

2. Yann LeCun and Corinna Cortes. “MNIST handwritten digit database”.
In: (2010). url: http://yann.lecun.com/exdb/mnist/.

3. Emile Mathieu et al. “Continuous Hierarchical Representations with Poincar ́e
Variational Auto-Encoders”. In: Proceedings of the 33rd International Con-
ference on Neural Information Processing Systems. Red Hook, NY, USA:
Curran Associates Inc., 2019.

4. Robin Rombach et al. High-Resolution Image Synthesis with Latent Diffu-
sion Models. 2021. arXiv: 2112.10752 [cs.CV].


## Tips

To run: 
'''
python ./latent-diffusion-main/main.py --base TEST_lsun_bedrooms-ldm-vq-4.yaml -t --gpus 0
'''

Get LSUN data from [here] (https://github.com/fyu/lsun)
