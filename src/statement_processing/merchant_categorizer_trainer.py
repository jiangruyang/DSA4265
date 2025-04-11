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
    def train_model(cls, training_data_path: str, model_output_path: str, epochs: int = 10):
        """Train a new BERT model for merchant categorization
        
        Args:
            training_data_path: Path to the training data
            model_output_path: Path to save the trained model
            epochs: Number of training epochs
        """
        # Load training data
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        # Prepare data
        texts = [item['merchant_name'] for item in training_data]
        
        # Map categories to indices
        category_to_idx = {cat: idx for idx, cat in enumerate(MerchantCategorizer.CATEGORIES)}
        labels = [category_to_idx[item['category']] for item in training_data]
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(MerchantCategorizer.CATEGORIES)
        )
        
        # Tokenize data
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        
        # Create dataset
        dataset = MerchantDataset(encodings, labels)
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Setup training
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Train model
        model.train()
        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
        
        # Save model and tokenizer
        os.makedirs(model_output_path, exist_ok=True)
        model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)
        
        logger.info(f"Model saved to {model_output_path}")
        
        return model, tokenizer

    @classmethod
    def generate_training_data(cls, source_data_path: str, output_path: str, is_training: bool = True):
        """Generate training or test data from source data
        
        Args:
            source_data_path: Path to the source data (e.g., ACRA data)
            output_path: Path to save the generated data
            is_training: Whether to generate training data (True) or test data (False)
        """
        # Implementation for generating training data would go here
        # This is a placeholder that would need to be implemented based on your data source
        logger.info(f"Generating {'training' if is_training else 'test'} data from {source_data_path}")
        
        # Example implementation - would need to be adapted to your actual data source
        # For now, we'll just create a placeholder that explains what would happen
        placeholder_data = [
            {"merchant_name": "Example Grocery Store", "category": "groceries"},
            {"merchant_name": "Sample Restaurant", "category": "dining"},
            {"merchant_name": "Test Transport Company", "category": "transportation"}
        ]
        
        # Save the generated data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(placeholder_data, f, indent=2)
        
        logger.info(f"Generated data saved to {output_path}")
        
        return placeholder_data


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
    
    # Generate test data if it doesn't exist
    if not os.path.exists(test_data_path):
        try:
            logger.info("Generating test data...")
            MerchantCategorizerTrainer.generate_training_data(
                "data/categorization/ACRA.csv",
                test_data_path,
                is_training=False
            )
        except Exception as e:
            logger.error(f"Error generating test data: {str(e)}")
            exit(1)
    else:
        logger.info(f"Using existing test data from {test_data_path}")
    
    # Check if BERT model exists
    if not os.path.exists(model_path):
        # Generate training data if it doesn't exist
        if not os.path.exists(training_data_path):
            try:
                logger.info("Generating training data...")
                MerchantCategorizerTrainer.generate_training_data(
                    "data/categorization/ACRA.csv",
                    training_data_path,
                    is_training=True
                )
            except Exception as e:
                logger.error(f"Error generating training data: {str(e)}")
                exit(1)
        
        # Train the model
        try:
            logger.info("Training model...")
            MerchantCategorizerTrainer.train_model(
                training_data_path,
                model_path
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