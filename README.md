# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ebmnskxgwv/badge.svg)](https://ask.thili.ai/b/ebmnskxgwv)

This project develops a retail recommender that learns from shopping basket data to suggest frequently bought items. It then applies quantization techniques to significantly reduce the model's footprint, making it suitable for edge devices or faster loading, proving the practical application of language models and model compression in retail.

## What you are building

**Goal.** You will build a smart retail recommender that identifies frequently bought items and then shrinks its model using quantization for efficient deployment.

**Done when.** You have a functional recommendation model that predicts next items and can be probed for itemsets, and you have successfully applied dynamic and static post-training quantization to reduce its size while evaluating the impact on accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare synthetic shopping basket data for training**  
      Hint: Use data generation patterns from `Teach a Language Model What Sells Together` to create synthetic basket data.
- [ ] **2. Continue pretraining a small LM on the basket data**  
      Hint: Apply the continued pretraining techniques from `Teach a Language Model What Sells Together` to your small language model.
- [ ] **3. Implement next-item prediction and itemset probing**  
      Hint: Adapt the next-item prediction and itemset probing methods from `Teach a Language Model What Sells Together`.
- [ ] **4. Measure the baseline size and accuracy of the model**  
      Hint: Calculate the model's size and evaluate its prediction accuracy before any quantization.
- [ ] **5. Apply dynamic and static post-training quantization to the model**  
      Hint: Implement dynamic and static post-training quantization on your model using methods from `Shrink a Neural Network with Quantisation`.
- [ ] **6. Demonstrate size reduction and evaluate accuracy impact**  
      Hint: Compare the size and accuracy of your quantized models against the baseline model.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Compare the inference speed of the quantized model against the baseline floating-point model.
- Explore the impact of different quantization bit-widths (e.g., int8 vs. int4) if supported by your chosen framework.
- Integrate the recommender into a simple command-line interface for interactive recommendations.
