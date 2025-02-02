# PyTorch computer vision basics

<p align="center">
  <img src='pics/face.png'  width='500'/>
</p>

Hello again 👋
+ [Computer Vision](https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-computer-vision#:~:text=Computer%20vision%20is%20a%20field,tasks%20that%20replicate%20human%20capabilities.) is a branch of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) that aims to artificially imitate human vision by enabling computers to perceive visual stimuli (photos, videos) meaningfully
+ In this repository, I solve an end to end multi-class image classification problem on the [`FashionMNIST`](https://github.com/zalandoresearch/fashion-mnist) dataset. `FashionMNIST` is a relatively simple dataset that has already been solved, but it's good for entry-level Deep Learning practice. Besides, I'll be focusing on getting the basics right. The `FashionMNIST` dataset consists of grayscale images, with `60,000` training samples and `10,000` test samples of `28` pixels in height and width
+ There's many algorithms that can solve a computer vision problem, but first we'll start with a standard [Multi-Layer Perceptron](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning) made of linear layers. We'll improve on this by building a second network using [Convolutional Neural Network (CNN)](https://youtu.be/YRhxdVk_sIs?si=k07XdCsMDS3OQDmh), then finish with a tweaked version of the [TinyVGG model](https://www.youtube.com/watch?v=HnWIHWFbuUQ)
+ Comments, working code, and links to the latest official documentation are included every step of the way. There's links to open each notebook (_labeled 01...04_) in Google Colab - feel free to play around with the code

## Milestones 🏁
**Concepts covered in this exercise include:**  
1. [x] Data wrangling and visualization
2. [x] Training and evaluating a multi-class classification model - build using [`PyTorch`](https://pytorch.org/)
3. [x] Regularization using [Early stopping](https://www.linkedin.com/advice/1/what-benefits-drawbacks-early-stopping#:~:text=Early%20stopping%20is%20a%20form,to%20increase%20or%20stops%20improving.), [`nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d), [`nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d), and [`nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)

## Tools ⚒️
1. [`Google Colab`](https://colab.google/) - A hosted _Jupyter Notebook_ service by Google.
2. [`PyTorch`](https://pytorch.org/) -  An open source machine learning (ML) framework based on the Python programming language, that is used for building **Deep Learning models**
3. [`scikit-learn`](https://scikit-learn.org/stable/#) - A free open-source library that offers Machine Learning tools for the Python programming language
4. [`numpy`](https://numpy.org/) - The fundamental package for scientific computing with Python
5. [`matplotlib`](https://matplotlib.org/) - A comprehensive library for making static, animated, and interactive visualizations in Python
6. [`torchinfo`](https://github.com/TylerYep/torchinfo) - A library for viewing model summaries in PyTorch

## Results 📈
> On a scale of `0` -> `1`, the final best-performing model achieved:
+ A weighted `precision`, `recall` and `f1_score` of `0.92`
+ An overall model accuracy of `0.9179`
+ An overall `roc_auc_score` of `0.995`

> The saved model's `state_dict` can be found in drive folder linked [here](https://drive.google.com/drive/folders/1FjC2wCK9UzDOBA0qBgNVunUxLVhqsb3l?usp=drive_link)


## Reference 📚
+ Thanks to the insight gained from [`Daniel Bourke`](https://x.com/mrdbourke?s=21&t=1Fg4dWHIo5p7EaMHhv2rng) and [`Modern Computer Vision with Pytorch, 2nd Edition`](https://www.packtpub.com/en-us/product/modern-computer-vision-with-pytorch-9781803240930)
+ Not forgetting these gorgeous gorgeous [`emojis`](https://gist.github.com/FlyteWizard/468c0a0a6c854ed5780a32deb73d457f) 😻

> _Illustration by [`Storyset`](https://storyset.com)_ ♥

