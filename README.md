[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/ebmnskxgwv/badge.svg)](https://msrit.ask.thili.ai/b/ebmnskxgwv)

# Quantized Retail Recommender

This project develops a retail recommender that learns from shopping basket data to suggest frequently bought items. It then applies int8 quantization to reduce the model's footprint, making it suitable for edge devices or faster loading.

## What I built

- **Data:** 600 synthetic shopping baskets generated in `app/core.py` from seven "what sells together" patterns (for example bread, butter, jam), with random forgotten items and impulse buys added. No real customer data is used. Generation is seeded, so it is reproducible.
- **Model:** a tiny next-item model (embedding, tanh, linear layer, softmax) trained in pure Python in two stages: first on a quarter of the data, then continued on all of it.
- **Features:** next-item recommendations and an itemset probe that scores how strongly a set of items goes together.
- **Quantization:** int8 post-training quantization done two ways. Dynamic computes the activation scale for each input. Static fixes it in advance from calibration data.
- **App:** a Streamlit page where you pick basket items, see suggestions from each model, and compare size and accuracy.

## Milestones

- [x] 1. Prepare synthetic shopping basket data for training
- [x] 2. Continue pretraining a small LM on the basket data
- [x] 3. Implement next-item prediction and itemset probing
- [x] 4. Measure the baseline size and accuracy of the model
- [x] 5. Apply dynamic and static post-training quantization to the model
- [x] 6. Demonstrate size reduction and evaluate accuracy impact

## How to run

Requires Python 3.

```
pip install -r requirements.txt
python -m app.core            # prints the results table
streamlit run app/main.py     # opens the app in your browser
pytest                        # runs the tests
```

## Results

The model is scored on held-out baskets (the last 20%).

| Model | Size | Top-1 accuracy | Top-3 accuracy |
|---|---|---|---|
| Baseline (float32) | 1600 bytes | 82.3% | 86.3% |
| Dynamic int8 | 460 bytes (3.5x smaller) | 82.3% | 86.7% |
| Static int8 | 460 bytes (3.5x smaller) | 82.3% | 86.7% |

Example output: after `bread` the model suggests `butter` (77%), `jam` (10%) and `bacon` (3%). The set bread, butter, jam scores 0.61 for "going together", while the unrelated set tea, salsa, bacon scores 0.01.

## Limits

- The data is synthetic, so these numbers show the method works, not how it would do on real shoppers.
- The model only looks at the last item in the basket (a bigram view), so it cannot use longer basket history.
- The model is tiny (about 400 numbers), so the 3.5x size saving is small in absolute terms. The accuracy loss from int8 was zero here, but that may not hold for larger models.
- Sizes are measured as stored numbers (float32 versus int8), not as files on disk.

## Files

- `app/core.py`: data generation, model, training, prediction, quantization and evaluation
- `app/main.py`: Streamlit app
- `tests/test_core.py`: one test per milestone
- `.github/workflows/ci.yml`: runs the tests on every push

## Dependencies

The core code and tests use only the Python standard library, plus `pytest` for the tests. The app also needs `streamlit`.
