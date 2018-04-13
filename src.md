% Deep Learning in image deblurring
% Chun Pong Lau
% April 13, 2018 

## Menu
- Image Deblurring
- GAN
- WGAN
- WGAN-GP
- cGAN
- DeblurGAN

# Image Deblurring

## Non-uniform blur model
$$ 
I_B = k(M) \ast I_s + N
$$

- $I_B$: Blurred image
- $k$: Unknown blur kernel
- $M$: Motion field
- $I_S$: Latent sharp image
- $N$: Noise

## Image deblurring categories
- Blind
- Non-blind

##
### Blind
- Estimate both latent sharp image $I_S$ and blur kernel $k(M)$
- Rely on prior knowledge, image statistics and assumptions
- Running time is a significant problem
- Recently there appear some approaches based on convolutional neural networks(CNNs)

## CNNs

##
### Apply CNN to estimate the unknown blur function
- Estimate blur kernel (Sun)
- complex Fourier coefficients (Chakrabarti)
- motion flow estimation (Gong)

##
### Kernel-free approach
- Multi-scale CNN (Noorozi, Nah)
- Combination of pix2pix and densely connected convolutional networks (Ramakrishnan)

## Generative adversarial networks
- A game between two competing networks: the discriminator and the generator
- Generator receives noise as an input and generates a sample
- Discriminator receives a real and generated sample and is trying to distinguish between them.

## 
![](https://pic3.zhimg.com/80/v2-ebb78d726f3b35c6cfa940950ef39da2_hd.jpg)

## Minimax objective
The game between the generator $G$ and discriminator $D$ is the minimax objective:
$$
\min_G \max_D \underset{x \backsim \mathbb{P}_r}{\mathbb{E}} [\log(D(x))] + \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [\log(1-D(x))]
$$

- $\mathbb{P}_r$: Data distribution
- $\mathbb{P}_g$: Model distribution


##
### Advantages
- Generate samples of good perceptual quality

### Disadvantages
- Hard to train (mode collapse, vanishing gradients)

# Problems of Gan

## 1. The gradient of generator vanish when the discriminator gets better

##
- The Gan loss:
$$
\min_G \max_D \underset{x \backsim \mathbb{P}_r}{\mathbb{E}} [\log(D(x))] + \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [\log(1-D(x))]
$$

- When fixing G, the equation becomes $$P_r(x) \log D(x) + P_g(x) \log [1-D(x)]$$

- Taking derivative w.r.t $D(x)$, it becomes 
$$
\dfrac{P_r(x)}{D(x)} - \dfrac{P_g(x)}{1 - D(x)} = 0
$$

- Hence, it becomes $$D^*(x) = \dfrac{P_r(x)}{P_r(x) + P_g(x)}$$

##
Plugging the optimal $D^*$, the Gan loss becomes
$$
\underset{x \backsim \mathbb{P}_r}{\mathbb{E}} \Big[\log \dfrac{P_r(x)}{\frac{1}{2}\big(P_r(x) + P_g(x)\big)}\Big] + \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} \Big[\log \dfrac{P_g(x)}{\frac{1}{2}\big(P_r(x) + P_g(x)\big)}\Big] 
- 2\log2
$$

[//]: # (- Recall Kullback-Leibler divergence (KL divergence) and Jensen-Shannon (JS divergence), which are two important indexes measuring **similarity between two probability distributions**)

- $KL(P_1 \Vert P_2) = \underset{x \backsim \mathbb{P}_1}{\mathbb{E}} \log \dfrac{P_1}{P_2}$


- $JS(P_1 \Vert P_2) = \dfrac{1}{2}KL(P_1 \Vert \dfrac{P_1 + P_2}{2}) + \dfrac{1}{2}KL(P_2 \Vert \dfrac{P_1 + P_2}{2})$

- Hence the loss becomes $$2JS(P_r \Vert P_g) - 2\log2$$

##
The message here is that when we train the discriminator and it gets better, to minimize the generator loss  becomes to minimize the JS divergence between $P_r$ and $P_s$. 

## So what is the problem?

##
There is no problem if $P_g$ and $P_r$ have intersection. However, if they have no intersection or their intersection is measure-zero, then their JS divergence is 0 or $\log 2$. 

- If $P_g(x)=0$ and $P_r=0$, or $P_g(x) \ne 0$ and $P_r(x) \ne 0$, their JS divergence is 0.
- If $P_g(x)=0$ and $P_r \ne 0$, or $P_g(x) \ne 0$ and $P_r(x) = 0$, their JS divergence is $\log2$.
- $\log \dfrac{P_r}{\frac{1}{2}(P_r + 0)} = \log 2$

## 
In other words, when $P_g$ and $P_r$ have no intersection, or their intersection is measure-zero, their JS divergence is constant, hence the gradient is 0. 

## So how likely it will happen? 

## The answer is very likely. 

## 
Rigorously speaking, when the support of $P_r$ and $P_g$ is a low dimensional manifold in a high dimension space, the possibility of the measure of their intersection being measure-zero is 1. 

## Intuitively
- When you pick two curves in a 2D plane, the possibility that they intersect as a segment is 0. 
- And on the other hand, they may intersect as a point, but the measure, or the length of it is 0. Therefore, their intersection can be neglected. 

## GAN
The input of GAN is a noise vector from a low dimensional distribution (say 100), passing through a neural network, and becomes a high dimensional sample (a 64 * 64 image is 4096 dimension). 

##
![](https://pic1.zhimg.com/80/v2-8715a60c1a8993953f125e03938125d7_hd.jpg)

##
### 2. The generator loss is nonsense, the gradient is unstable and mode collapse

## log D trick
Ian Goodfellow, the author had proposed a trick on the generator loss. 

- Before: $\underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [\log(1-D(x))]$

- After: $\underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [-\log D(x)]$

##
Recall $D^*(x) = \dfrac{P_r(x)}{P_r(x) + P_g(x)}$

By plugging $D^*$,

$\begin{aligned}
KL(P_g \Vert P_r) &= \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [ \log \dfrac{P_g(x)}{P_r(x)}] \\
&= \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [ \log \dfrac{P_g(x)/(P_r(x)+P_g(x))}{P_r(x)/(P_r(x)+P_g(x))}] \\
&= \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [ \log \dfrac{1-D^*(x)}{D^*(x)}] \\
&= \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [ \log (1-D^*(x))) ] - \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [ \log D^*(x) ]
\end{aligned}$

## As a result

$\begin{aligned}
\mathbb{E}_{x \sim P_g} [-\log D^*(x)] &=  KL(P_g || P_r) -  \mathbb{E}_{x \sim P_g} \log [1 - D^*(x)] \\
&= KL(P_g || P_r) - 2JS(P_r || P_g) \\
&+ 2\log 2 + \mathbb{E}_{x\sim P_r}[\log D^*(x)]
\end{aligned}$

Since the last two terms do not depend on $G$, the formula becomes $$KL(P_g || P_r) - 2JS(P_r || P_g).$$

## The loss is nonsense
$$KL(P_g || P_r) - 2JS(P_r || P_g)$$

- One needs to minimize the KL divergence and maximize the JS divergence. 

- Hence, the gradient is very unstable. 

## KL divergence is antisymmetric
- When $P_g(x)\rightarrow 0$ and $P_r(x)\rightarrow 1$, $P_g(x) \log \dfrac{P_g(x)}{P_r(x)} \rightarrow 0$, then $KL(P_g \Vert P_r)$ tends to 0. 
 
- When $P_g(x)\rightarrow 1$ and $P_r(x)\rightarrow 0$, $P_g(x) \log \dfrac{P_g(x)}{P_r(x)} \rightarrow \infty$, then $KL(P_g \Vert P_r)$ tends to $\infty$. 

##
- In other words, the penalty is very small when the generator cannot produce some realistic samples. 
- However, the penalty is very large when the generator produce some unrealistic samples. 
- Hence, the generator inclines to produce some repetitive but "safe" samples rather than produce some diverse samples. And this is called collapse mode.

##
![](https://pic3.zhimg.com/v2-b85cdb4d79d7618213c320cfb3a6d4bf_r.jpg)

# Wasserstein distance

##
$$W(P_r, P_g) = \inf_{\gamma \sim \Pi (P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [\Vert x - y \Vert]$$

- $\Pi (P_r, P_g)$ is the collection of all possible joint distribution between $P_r$ and $P_g$. In other words, every marginal distribution in $\Pi (P_r, P_g)$ is $P_r$ and $P_g$. 

- For every joint distribution $\gamma$, we pick a sample $(x,y) \sim \gamma$ consisting of a real sample $x$ and a generated sample $y$. Then we calculate the distance $\Vert x-y \Vert$ and take the expected value. 

## Intuitively
We can understand $\mathbb{E}_{(x, y) \sim \gamma} [\Vert x - y \Vert]$ as follows:

- Under a path, $\gamma$, we want to move a pile of dirt from $P_r$ to $P_g$. 
- And this energy loss is $\mathbb{E}_{(x, y) \sim \gamma} [\Vert x - y \Vert].$ 
- Then $W(P_r, P_g)$ is the smallest energy loss under the best path. So Wasserstein distance is also called earth mover's distance.

## So why Wasserstein distance?

## 
The supremeness of Wasserstein distance over KL divergence and JS divergence is that even the two distributions have no intersection, Wasserstein distance can still measure the similarity between two distributions. 

# Wasserstein Gan

## 
So could we easily apply the Wasserstein distance to define the generator loss?

- ## Unfortunately, NO! 

## 
But we can transform Wasserstein distance as follows:

$$
W(P_r, P_g) = \frac{1}{K} \sup_{||f||_L \leq K} \mathbb{E}_{x \sim P_r} [f(x)] - \mathbb{E}_{x \sim P_g} [f(x)]
$$

- i.e.
$$
K \cdot W(P_r, P_g) \approx \max_{w: |f_w|_L \leq K} \mathbb{E}_{x \sim P_r} [f_w(x)] - \mathbb{E}_{x \sim P_g} [f_w(x)]
$$

- So our task becomes obtaining a function $f_w$ satisfies the above conditions, which can be obtained by the power of the deep neural network to fit it. 

## Of course
We need to satisfy the constraint $\| f_w \|_L \leq K$

- As long as $K \ne \infty$, we do not need to consider $K$ as it only magnifies the value of gradient by $K$ times but not affect the direction of gradient. 
- As a result, we can limit the weight in the neural network in a range of $[-c,c].$

## Sigmoid layer is removed
- Also, in the original GAN model, the task of discriminator is to classify whether the sample is real or generated so a sigmoid layer is needed. 

- However, in the WGAN model, we need to have a discriminator $D$ to approximate the Wasserstein distance, so it is a regression task. Hence, the sigmoid layer is removed. 

## 
### The modification of WGAN made

- The loss is changed to remove the log function
$$
\min_G \max_{D \in \mathcal{D}} \underset{x \backsim \mathbb{P}_r}{\mathbb{E}} [D(x)] + \underset{x \backsim \mathbb{P}_g}{\mathbb{E}} [D(x)]
$$

- Sigmoid layer is removed

- Weight clipping, i.e. to make all the values of the weight lies in the range of [-c,c]

- Don't use optimization scheme with momentum (momentum and Adam), RMSProp and SGD are recommended. 

## WGAN is not perfect

## Weight clipping

Since weight clipping enforces the values of weight need to be in the range of $[-c, c]$, and the loss of discriminator, i.e.
$$L(D) = -\mathbb{E}_{x\sim P_r}[D(x)] + \mathbb{E}_{x\sim P_g}[D(x)],$$
wants to maximize the value of real samples and minimize the value of generated samples, then the best strategies of the machine is to pick the value near to the boundary value in the range.  

## 
![](https://pic1.zhimg.com/80/v2-7a3aedf9fa60ce660bff9f03935d8f15_hd.jpg)

## Weight clipping make gradient vanish or explode

- If we set the threshold small, the gradient becomes smaller when it passes through a layer, as a result, the gradient will decay exponentially.

- If we set the threshold large, the gradient becomes larger when it passes through a layer, as a result, the gradient will grow exponentially.

##
![](https://pic1.zhimg.com/80/v2-34114a10c56518d606c1b5dd77f64585_hd.jpg)


## Gradient penalty
$$[(\Vert \nabla_{\tilde{x}} D(\tilde{x}) \Vert_2 - K)^2]$$

As a result, the new discriminator loss becomes 
$$
L(D) = -\mathbb{E}_{x\sim P_r}[D(x)] + \mathbb{E}_{x\sim P_g}[D(x)] + \lambda \mathbb{E}_{x \sim \mathcal{X}} [ || \nabla_x D(x) ||_p - 1 ]^2
$$

## Problem
It is extremely difficult to do sampling in the entire sample space $\mathcal{X}$. 

## Alternative
But we have an alternative, we can do sampling in the union of $P_r$ and $P_g$, i.e.
$$x_r \sim P_r, x_g \sim P_g, \epsilon \sim Uniform[0, 1]$$
$$\hat x = \epsilon x_r + (1 - \epsilon) x_g$$

- Finally,  the loss becomes 
$$L(D) = -\mathbb{E}_{x\sim P_r}[D(x)] + \mathbb{E}_{x\sim P_g}[D(x)] + \lambda \mathbb{E}_{x \sim \mathcal{P_{\hat x}}} [ || \nabla_x D(x) ||_p - 1 ]^2$$

## 
![](https://pic1.zhimg.com/80/v2-5b01ef93f60a14e7fa10dbea2b620627_hd.jpg)

##
![](https://pic4.zhimg.com/80/v2-e0a3d86ccfa101a4d3fee1c0cef96a81_hd.jpg){ width=400px }

# Conditional adversarial networks

##
Gan model: $z \mapsto y$

cGan model: $z,x \mapsto y$

where $z$ is noise vector, $x$ is input image and $y$ is output images

##
![](https://pic1.zhimg.com/80/v2-239d9f40edd8b90f8ea4eced4471d649_hd.jpg)

# DeblurGan

## Overview
Given a blurred image $I_b$, we hope to reconstruct a sharp image $I_s$. The authors build a GAN model, with a CNN as a generator $G_{\theta_{G}}$ and a critic $D_{\theta_D}$. 

## Training overview
![](https://pic4.zhimg.com/80/v2-593b95a342e8ae874ffdb025a63f4dab_hd.jpg)

## Generator network 
- Two strided convolution blocks with stride $\frac{1}{2}$
- Nine residual blocks (ResBlocks)
- Two transposed convolution blocks

## ResBlocks
- One convolutional layer
- One instance normalization layer
- ReLU activation
- Dropout layer with probability 0.5

## 
![](https://pic1.zhimg.com/v2-999a26b8ebd62df0518f9c5cb5896f2c_r.jpg)

## Loss Function
Combination of content and adversarial loss:
$$
\mathcal{L} = \underbrace{ \underbrace{\mathcal{L}_{GAN}}_{\text{adv loss}} \quad + \quad \underbrace{\lambda \cdot \mathcal{L}_{X}}_{\text{content loss}}}_{\text{total loss}}
$$

## Adversarial loss
- GAN (vanilla GAN) - gradient vanish, mode collapse
- WGAN - discrete weight $\rightarrow$ unstable
- $$\mathcal{L}_{GAN} = \sum_{n=1}^N -D_{\theta_D}(G_{\theta_G}(I_B))$$

## Content loss 
- Common choices are $L_2$ and $L_1$ loss, blurry
- Using Perceptual loss, a simple $L_2$ loss, based on the difference of the generated and target image CNN feature maps

## Perceptual loss
$$
\mathcal{L}_x = \dfrac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}} (\phi_{i,j}(I^S)_{x,y} - \phi_{i,j}(G_{\theta_G}(I^B))_{x,y})^2
$$

- $\phi_{i,j}$ is the feature map obtained by the $j^{\text{th}}$ convolution (after activation) before the $i^{\text{th}}$ maxpooling layer within the VGG19 network

- $W_{i,j}$ and $H_{i,j}$ are the dimensions of the feature maps

## Why combine Adversarial loss and content loss?

##
![](https://pic2.zhimg.com/v2-1e2e080d43928577683655da7904936c_r.jpg) 

# Training details

##
- PyTorch
- Maxwell GTX Titan-X GPU
- Datasets (GoPro, MS COCO)
- Three models  

##
- $DeblurGAN_{Wild}$ - random crops of size 256x256 from 1000 GoPro training dataset images
- $DeblurGAN_{Synth}$ - 256x256 patches from MS COCO dataset blurred by proposed method
- $DeblurGAN_{Comb}$ - Combination of synthetically blurred images and images taken in the wild

## Results

## 
![](https://pic3.zhimg.com/v2-064c4f0a8435bf3c923cd1fc65194cf9_r.jpg)

## Readings
- Wasserstein GAN
	- [https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875)
- Improved Training of Wasserstein GANs	
	- [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
- Image-to-Image Translation with Conditional Adversarial Networks
	- [https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)
- DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks
	- [https://arxiv.org/abs/1711.07064](https://arxiv.org/abs/1711.07064)
