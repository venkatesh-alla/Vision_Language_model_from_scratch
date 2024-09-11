# PaliGemma Vision Language model implementation from Scratch

In this project, I have built a PaliGemma Vision language model from scratch. This required the implementation of a variety of state-of-the-art concepts.

## Transformer concepts

`Embeddings` : Both the text and image inputs are transformed into vector representations called as Embeddings. Sentences are converted into tokens, each token with a corresponding embedding and similarly, all the images have been divided into patches, each patch with a corresponding embedding.

`Postional Encoding`: Since, transformers do not inherently understand the order of inputs, positional encodings are added to embeddings to retain the sequential nature of words and spatial nature of images.

`Multi-head Attention` : As seen in the "Attention is all you need paper implementation", the embeddings are then passed through Multi-head attention layer, allowing model to attend to different parts of language sequence, thereby each head understanding different parts of the sequence in a different context. Similarly, different parts of image features are passed through different heads and each head understands the relation among different parts of input in a different way.

`Feed forward layer`: After attention, each token is passed through a fully-connected neural network to add non-linearity and complexity to the model.

`Logits and Softmax`: The output logits are then transformed into probabilities using softmax layer, which later undergo different strategies like 'Greedy strategy' to decide the final output.

## Contrastive Learning

Contrastive learning helps models understand data by making them compare pairs of information:

CLIP: CLIP (Contrastive Language-Image Pre-training) is designed to match images with the right text. It does this by aligning their representations, so the model learns which caption goes with which image. This helps the model grasp the meaning and connection between text and images.

SigLip: SigLip is a technique that keeps the model's learning smooth by limiting how much it changes. This makes the model more stable and reliable when it comes to understanding and generalizing new data.

## Grouped Query Attention

Grouped Query Attention optimizes the attention mechanism by splitting queries into groups, which reduces computational complexity. This is helpful when dealing with large input sequences, like high-resolution images, without compromising performance.

## Normalization Layers

Normalization is crucial for stable and efficient training:

`Batch Normalization`: Normalizes the input by subtracting the batch mean and dividing by the batch standard deviation.

`Layer Normalization`: Normalizes the inputs across features rather than across the batch, which is often used in Transformers.

## KV-Cache (Prefilling and Token Generation)
In models that generate text or work with sequential inputs, key-value caching (KV-Cache) speeds up inference by storing previously computed key-value pairs from the attention mechanism. This cache is reused in subsequent layers or steps, reducing redundant calculations, especially in autoregressive models.

## Weight Tying

Weight tying reduces the number of parameters by sharing weights between different layers or between the encoder and decoder in a model. For example, the embedding layer and the output projection layer in language models often share the same weights, saving memory and improving generalization.

## Top-P Sampling and Temperature
During inference, Top-P Sampling (nucleus sampling) selects tokens from the smallest subset of the vocabulary that cumulatively exceeds a certain probability threshold *p*. This ensures only high-confidence predictions are made. Temperature is a scaling factor applied to logits before sampling, controlling the randomness in the generation process. A higher temperature makes the model more creative, while a lower temperature makes it more deterministic.

This entire pipeline helps the model understand and generate language that corresponds to visual input, forming a vision-language model. All these concepts are thoroughly understood and implemented from scratch in PyTorch as a part of this project