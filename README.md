Based on Antonio Galiza Cerdeira Gonzales work. https://gitlab.com/AntonioGCGonzalez/synaptogenic-adaptive-processing-unit-language-models

I rewrote it for image classification and tested a bunch of encoders and decoders. I also tested it against Resnet-18 on ImageNet-200 tiny (64x64 images, downscaled to 73x73 and applied deterministic cropping).
Presentation can be found in presentation/presentation.pdf
There are some visualisations which analyze the model further.
This model seems to be able to reuse weights very efficiently, but is slow to train. It also allows for great compression since most weights can be ommitted. Weights also have a direct counterpart with changed sign, so it allows for extremely efficient compression. I could get down to ~600 bytes on undocumented CIFAR-10 experiment with Antonio's old algorithm from listed repo. I did not find any similiarities between topologies of this model trained on different encoder/decoder pairs.
Model seems to perform well enough despite not fitting naturally to static data. Future studies will be done directly on videos.


Since this model is not a good fit for this task it is more of a workbench than a working product with a clean repo. 
Despite all of this, I could achieve ~42% accuracy with 702k parameters. ResNet-18 achieved ~57% with 11mln parameters, which makes my model around 16 times smaller and 12 times more effective with accuracy% per parameter metric. It is worth to mention, that ResNet achieved almost 100% train acc so it can not improve much, meanwhile TP-SAPU could theoretically still learn a bit more to bridge the accuracy gap.


Future studies will focus on same architecture but different topology. I will also try to avoid using backpropagation, since this will allow for greater synchronisation among neurons and faster training. Neurons try to synchronise anyway, which can be seen in visualizations/tiny_imagenet_702k_vs_2048/shared_recurrent_comparison.png
