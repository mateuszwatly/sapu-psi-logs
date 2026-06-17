Based on Antonio Galiza Cerdeira Gonzales work. https://gitlab.com/AntonioGCGonzalez/synaptogenic-adaptive-processing-unit-language-models

I rewrote it for image classification and tested a bunch of encoders and decoders. I also tested it against Resnet-18 on ImageNet-200 tiny (64x64 images, downscaled to 73x73 and applied deterministic cropping).
Presentation can be found in presentation/presentation.pdf
There are some visualisations which analyze the model further.
This model seems to be able to reuse weights very efficiently, but is slow to train. It also allows for great compression since most weights can be ommitted. I did not find any similiarities between topologies of this model trained on different encoder/decoder pairs.
Model seems to perform well enough despite not fitting naturally to static data. Future studies will be done directly on videos.


Since this model is not a good fit for this task it is more of a workbench than a working product with a clean repo. 
