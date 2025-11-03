"""Example fine-tune script using Hugging Face Transformers + MLflow tracking.
This is a template — adapt dataset, model, and hyperparameters for your use-case.
"""
import os
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    model_name = os.getenv('BASE_MODEL', 'gpt2')
    dataset_name = os.getenv('HF_DATASET', 'wikitext')
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT', 'llm_finetune')

    mlflow.set_tracking_uri(mlflow_uri or 'http://localhost:5000')
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param('base_model', model_name)
        ds = load_dataset(dataset_name, 'wikitext-2-raw-v1')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        def tokenize(batch):
            return tokenizer(batch['text'], truncation=True, max_length=512)
        tokenized = ds.map(tokenize, batched=True, remove_columns=['text'])

        args = TrainingArguments(
            output_dir='outputs',
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=10,
            save_total_limit=2,
            fp16=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'])
        trainer.train()

        # Save and log model
        saved_path = 'outputs/model'
        model.save_pretrained(saved_path)
        tokenizer.save_pretrained(saved_path)
        mlflow.log_artifacts(saved_path, artifact_path='model')
        mlflow.log_metric('dummy_metric', 0.0)
        print('Training complete. Artifacts logged to MLflow.')

if __name__ == '__main__':
    main()