import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add project root to Python path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the main categorizer to access categories and for testing
from src.statement_processing.merchant_categorizer import MerchantCategorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")


class MerchantDataset(Dataset):
    """Dataset class for merchant categorization"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MerchantCategorizerTrainer:
    """Class for training and evaluating merchant categorization models"""
    
    @classmethod
    def evaluate_gpt4(cls, test_data_path: str):
        """Evaluate GPT-4's performance on the same test data
        
        Args:
            test_data_path: Path to the test data
        """
        # Define path for saved GPT-4 results
        gpt4_results_path = "data/categorization/test/gpt4_evaluation_results.json"
        
        # Try to load existing results first
        if os.path.exists(gpt4_results_path):
            logger.info("Loading existing GPT-4 evaluation results...")
            with open(gpt4_results_path, 'r') as f:
                return json.load(f)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Prepare predictions and true labels
        true_labels = []
        predicted_labels = []
        predictions_data = []  # Store full prediction data
        
        for item in tqdm(test_data, desc="Evaluating GPT-4"):
            merchant_name = item['merchant_name']
            true_category = item['category']
            
            # Create prompt for GPT-4
            prompt = f"""Categorize this merchant name into one of these categories: {', '.join(MerchantCategorizer.CATEGORIES)}.
            Only respond with the category name, nothing else.
            
            Merchant name: {merchant_name}"""
            
            try:
                # Call GPT-4
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a merchant categorization assistant. Respond only with the category name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                predicted_category = response.choices[0].message.content.strip().lower()
                
                # Validate prediction
                if predicted_category not in MerchantCategorizer.CATEGORIES:
                    predicted_category = "others"
                
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                
                # Store prediction data
                predictions_data.append({
                    'merchant_name': merchant_name,
                    'true_category': true_category,
                    'predicted_category': predicted_category
                })
                
            except Exception as e:
                logger.error(f"Error in GPT-4 prediction: {str(e)}")
                true_labels.append(true_category)
                predicted_labels.append("others")
                predictions_data.append({
                    'merchant_name': merchant_name,
                    'true_category': true_category,
                    'predicted_category': "others"
                })
        
        # Calculate classification report
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=MerchantCategorizer.CATEGORIES,
                                    labels=MerchantCategorizer.CATEGORIES,
                                    output_dict=True)
        
        # Prepare results dictionary
        results = {
            'classification_report': report,
            'predictions': predictions_data
        }
        
        # Save results
        os.makedirs(os.path.dirname(gpt4_results_path), exist_ok=True)
        with open(gpt4_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print classification report
        print("\nGPT-4 Classification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=MerchantCategorizer.CATEGORIES,
                                  labels=MerchantCategorizer.CATEGORIES))
        
        return results

    @classmethod
    def evaluate_model(cls, model_path: str, test_data_path: str):
        """Evaluate the trained model on test data and compare with GPT-4
        
        Args:
            model_path: Path to the trained BERT model
            test_data_path: Path to the test data
        """
        print("\n=== Evaluating BERT Model ===")
        bert_metrics = cls._evaluate_bert(model_path, test_data_path)
        
        print("\n=== Evaluating GPT-4 ===")
        gpt4_metrics = cls.evaluate_gpt4(test_data_path)
        
        # Compare results
        print("\n=== Model Comparison ===")
        print("BERT Model:")
        print(f"Accuracy: {bert_metrics['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {bert_metrics['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {bert_metrics['classification_report']['weighted avg']['f1-score']:.3f}")
        
        print("\nGPT-4:")
        print(f"Accuracy: {gpt4_metrics['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {gpt4_metrics['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {gpt4_metrics['classification_report']['weighted avg']['f1-score']:.3f}")
        
        return {
            'bert_metrics': bert_metrics,
            'gpt4_metrics': gpt4_metrics
        }

    @classmethod
    def _evaluate_bert(cls, model_path: str, test_data_path: str):
        """Internal method to evaluate BERT model performance
        
        Args:
            model_path: Path to the trained BERT model
            test_data_path: Path to the test data
        """
        # Initialize model
        model = MerchantCategorizer(model_path)
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Prepare predictions and true labels
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        
        for item in tqdm(test_data, desc="Evaluating model"):
            merchant_name = item['merchant_name']
            true_category = item['category']
            
            try:
                # Use BERT model for prediction without fallback
                inputs = model.tokenizer(merchant_name, return_tensors="pt", padding=True, truncation=True)
                outputs = model.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_category_idx = torch.argmax(predictions).item()
                confidence = predictions[0][predicted_category_idx].item()
                
                predicted_category = model.categories[predicted_category_idx]
                
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                confidence_scores.append(confidence)
                
            except Exception as e:
                logger.error(f"Error in BERT prediction: {str(e)}")
                # Skip this item instead of using rule-based fallback
                continue
        
        # Get unique categories in the test data
        unique_categories = sorted(set(true_labels + predicted_labels))
        
        # Calculate and print classification metrics
        print("\nClassification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=unique_categories,
                                  labels=unique_categories))
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_categories)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_categories,
                    yticklabels=unique_categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Calculate average confidence per category
        confidence_by_category = {}
        for true, conf in zip(true_labels, confidence_scores):
            if true not in confidence_by_category:
                confidence_by_category[true] = []
            confidence_by_category[true].append(conf)
        
        print("\nAverage Confidence by Category:")
        for category, confidences in confidence_by_category.items():
            avg_conf = np.mean(confidences)
            print(f"{category}: {avg_conf:.3f}")
        
        return {
            'classification_report': classification_report(true_labels, predicted_labels, 
                                                         target_names=unique_categories,
                                                         labels=unique_categories,
                                                         output_dict=True),
            'confusion_matrix': cm,
            'confidence_by_category': {cat: np.mean(confs) for cat, confs in confidence_by_category.items()}
        }

    @classmethod
    def train(cls, training_data_path: str, output_model_path: str):
        """Train a BERT model on the generated training data
        
        Args:
            training_data_path: Path to the training data generated by GPT-4
            output_model_path: Path to save the trained model
        """
        # Load training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Split into train and validation sets
        train_size = int(0.9 * len(training_data))
        train_data = training_data[:train_size]
        val_data = training_data[train_size:]
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(MerchantCategorizer.CATEGORIES)
        )
        
        # Prepare datasets
        def prepare_dataset(data):
            texts = [item['merchant_name'] for item in data]
            labels = [MerchantCategorizer.CATEGORIES.index(item['category']) for item in data]
            encodings = tokenizer(texts, truncation=True, padding=True)
            return MerchantDataset(encodings, labels)
        
        train_dataset = prepare_dataset(train_data)
        val_dataset = prepare_dataset(val_data)
        
        # Define training arguments
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the best model
        model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)
        logger.info(f"Model saved to {output_model_path}")
        
        return model, tokenizer

    @classmethod
    def generate_dataset(cls, ACRA_path: str, categories_mapping_path: str, output_path_train: str, output_path_test: str):
        """Generate merchant categorization dataset from ACRA data
        
        Args:
            ACRA_path: Path to ACRA dataset
            categories_mapping_path: Path to category mapping JSON file
            output_path_train: Path to save training dataset
            output_path_test: Path to save test dataset
        """
        logger.info("Loading ACRA data from %s", ACRA_path)
        # Load ACRA data
        acra_df = pd.read_csv(ACRA_path)

        acra_df = acra_df[['entity_name', 'retrieved_ssic_description']]
        acra_df.rename(columns={'entity_name': 'merchant_name'}, inplace=True)
        
        # Load category mapping
        logger.info("Loading category mapping from %s", categories_mapping_path)
        with open(categories_mapping_path, 'r') as f:
            categories_mapping = json.load(f)
        
        # Map SSIC descriptions to categories
        logger.info("Mapping SSIC descriptions to categories")
        
        # Filter out rows with missing values
        acra_df = acra_df.dropna(subset=['merchant_name', 'retrieved_ssic_description'])
        
        # Map categories and filter out invalid ones
        acra_df['category'] = acra_df['retrieved_ssic_description'].map(categories_mapping)
        acra_df = acra_df[acra_df['category'].isin(MerchantCategorizer.CATEGORIES)]
        
        # Keep only relevant columns
        df = acra_df[['merchant_name', 'category']]
        
        # Undersample to balance the dataset
        samples_per_category = 2500
        logger.info("Undersampling to balance the dataset")
        
        # Perform undersampling
        balanced_df = pd.DataFrame()
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            if len(category_df) > samples_per_category: 
                sampled_df = category_df.sample(n=samples_per_category, random_state=42)
            else:
                sampled_df = category_df
            balanced_df = pd.concat([balanced_df, sampled_df])
            logger.info("Category %s: sampled %d from %d items", 
                       category, len(sampled_df), len(category_df))
        
        # Split into train and test sets (80/20) using sklearn
        train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
        
        logger.info("Train set size: %d", len(train_df))
        logger.info("Test set size: %d", len(test_df))
        
        # Convert DataFrames to list of dictionaries for JSON serialization
        train_data = train_df.to_dict(orient='records')
        test_data = test_df.to_dict(orient='records')
        
        # Save to JSON files
        os.makedirs(os.path.dirname(output_path_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_path_test), exist_ok=True)
        
        with open(output_path_train, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(output_path_test, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info("Training data saved to %s", output_path_train)
        logger.info("Test data saved to %s", output_path_test)
        
        return train_data, test_data


# Example usage
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/categorization/training", exist_ok=True)
    os.makedirs("data/categorization/test", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Define paths for data files
    training_data_path = "data/categorization/training/merchant_training_data.json"
    test_data_path = "data/categorization/test/merchant_test_data.json"
    model_path = "models/merchant_categorizer"
    
    # Generate dataset if they don't exist
    if not os.path.exists(training_data_path) or not os.path.exists(test_data_path):
        try:
            logger.info("Generating training and test datasets...")
            MerchantCategorizerTrainer.generate_dataset(
                "data/categorization/ACRA.csv",
                "data/categorization/categories_mapping.json",
                training_data_path,
                test_data_path
            )
        except Exception as e:
            logger.error(f"Error generating datasets: {str(e)}")
            exit(1)
    else:
        logger.info(f"Using existing datasets from {training_data_path} and {test_data_path}")
    
    # Check if BERT model exists
    if not os.path.exists(model_path):
        # Train the model
        try:
            logger.info("Training model...")
            MerchantCategorizerTrainer.train(
                training_data_path=training_data_path,
                output_model_path=model_path
            )
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            exit(1)
    else:
        logger.info(f"Using existing BERT model from {model_path}")
    
    # Evaluate the model
    try:
        logger.info("Evaluating models...")
        metrics = MerchantCategorizerTrainer.evaluate_model(
            model_path,
            test_data_path
        )
        
        # Print overall metrics
        print("\n=== Overall Model Performance ===")
        print("\nBERT Model:")
        print(f"Accuracy: {metrics['bert_metrics']['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {metrics['bert_metrics']['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {metrics['bert_metrics']['classification_report']['weighted avg']['f1-score']:.3f}")
        
        print("\nGPT-4o Model:")
        print(f"Accuracy: {metrics['gpt4_metrics']['classification_report']['accuracy']:.3f}")
        print(f"Macro Avg F1: {metrics['gpt4_metrics']['classification_report']['macro avg']['f1-score']:.3f}")
        print(f"Weighted Avg F1: {metrics['gpt4_metrics']['classification_report']['weighted avg']['f1-score']:.3f}")
        
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        exit(1) 