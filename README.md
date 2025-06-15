# Deep Java Library Zero-Shot Classification

This project compares the results of zero-shot text classification between [Deep Java Library (DJL)](https://djl.ai/) and Hugging Face's Transformers library in Python. It’s meant to validate a custom DJL translator implementation for zero-shot classification.

## What’s the Goal?

We want to make sure our DJL-based zero-shot classification behaves just like the one from Python's Transformers. So, we run the same inputs through both and check if the outputs match. This helps confirm that our `ZeroShotClassificationTranslator.java` is working as expected.

## What is Zero-Shot Classification?

Zero-shot classification lets you classify text into categories the model hasn’t explicitly seen before. You don’t need to retrain the model for every new set of labels.

## Models Used

We test and compare the following pre-trained models:

1. **facebook/bart-large-mnli**
2. **MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli**
3. **tasksource/ModernBERT-base-nli**
4. **MoritzLaurer/bge-m3-zeroshot-v2.0**

These are all NLI-based models suitable for zero-shot tasks.

## Model Preparation

Before using the models with DJL, convert them with the `djl-converter`:

```bash
djl-convert -m facebook/bart-large-mnli
djl-convert -m MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
djl-convert -m tasksource/ModernBERT-base-nli
djl-convert -m MoritzLaurer/bge-m3-zeroshot-v2.0
```

## How It Works

In `Main.java`, we:

1.	Load each model (from local files after conversion)
2.	Run zero-shot classification on the sentence: "Java is the best programming language"
3.	Use three candidate labels: "Software Engineering", "Software Programming", "Politics"
4.	Compare the output scores to those from the Transformers library in Python
5.	Test both single-label and multi-label modes

In `ZeroShotClassificationTranslator.java`, we define how the inputs are tokenized and how results are post-processed for classification.

## Requirements 

- Java 21+
- Maven

## Dependencies 

```xml
ai.djl.huggingface:tokenizers:0.32.0
ai.djl.pytorch:pytorch-engine:0.32.0
ai.djl:model-zoo:0.32.0
```

