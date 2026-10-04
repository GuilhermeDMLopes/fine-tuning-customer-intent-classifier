# Fine-tuning Customer Intent Classifier

Fine-tuning challenge that trains a Small Language Model (SLM) to classify retail customer messages into two intents:

- `venda`: purchase intent.
- `suporte`: questions or problems related to a product.

## Approach

The project follows the course notebook pattern using Hugging Face `datasets`, `transformers`, `Trainer`, and a BERT sequence-classification model. Because the dataset is in Portuguese, the selected checkpoint is [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased).

## Project structure

```text
.
├── data/
│   ├── treino.jsonl
│   └── teste.jsonl
├── notebooks/
│   └── fine_tuning_bert.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The synthetic dataset supplied with the challenge uses one JSON object per line:

```json
{"prompt": "Quero comprar uma geladeira.", "completion": "venda"}
```

The notebook uses `treino.jsonl` for training and `teste.jsonl` for evaluation.

## Running in Google Colab

1. Upload the `data/treino.jsonl` and `data/teste.jsonl` files to `/content/` in Colab.
2. Upload or open `notebooks/fine_tuning_bert.ipynb`.
3. Run the cells in order.
4. The trained model is saved to `bert-customer-intent/`.

The notebook installs the required libraries in its first code cell. For a local environment, install the same dependencies with:

```bash
pip install -r requirements.txt
```

## OpenAI API key

This solution fine-tunes BERT through Hugging Face and therefore **does not use the OpenAI API or OpenAI credits**. The notebook contains the required placeholder below only because it was requested in the challenge setup:

```python
OPENAI_API_KEY = "Insira sua chave da OPENAI aqui"
```

Do not commit a real API key to GitHub.

## Evaluation

The model is evaluated with accuracy on `teste.jsonl`. After training, the final cell demonstrates an inference using a text-classification pipeline.

## Notes

The provided data is synthetic and intended only for this educational challenge.
