import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from statement_processing.pdf_statement_parser import PDFStatementParser
import pandas as pd
from datetime import datetime

def main():
    # Create an instance of the parser
    parser = PDFStatementParser()
    
    # Test with the sample PDF file
    pdf_path = "data/Sample Bank Statement.pdf"
    
    # Parse the statement
    print(f"Testing PDF parser with file: {pdf_path}")
    transactions = parser.parse_statement(pdf_path)
    
    # Convert to DataFrame for better analysis
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Display summary statistics
    print("\n=== Transaction Summary ===")
    print(f"Total number of transactions: {len(transactions)}")
    print(f"Number of withdrawals: {len(df[df['type'] == 'withdrawal'])}")
    print(f"Number of deposits: {len(df[df['type'] == 'deposit'])}")
    
    # 1. Display spending by category (sorted by sum in descending order)
    print("\n=== Spending by Category ===")
    category_spending = df[df['type'] == 'withdrawal'].groupby('category')['amount'].agg(['sum', 'count'])
    category_spending['sum'] = category_spending['sum'].round(2)
    category_spending = category_spending.sort_values('sum', ascending=False)
    print(category_spending)
    
    # 2. Only display transactions with low confidence for user review
    low_confidence_threshold = 0.3
    low_confidence_transactions = df[df['confidence'] < low_confidence_threshold]
    
    if len(low_confidence_transactions) > 0:
        print(f"\n=== Transactions with Low Confidence (< {low_confidence_threshold}) ===")
        print(f"Total: {len(low_confidence_transactions)} transactions need review")
        
        for _, trans in low_confidence_transactions.iterrows():
            print("\n" + "-"*50)
            print(f"Date: {trans['date'].strftime('%Y-%m-%d')}")
            print(f"Merchant: {trans['merchant']}")
            print(f"Current Category: {trans['category']}")
            print(f"Confidence: {trans['confidence']:.4f}")
            print(f"Amount: ${trans['amount']:.2f}")
            print(f"Transaction Type: {trans['type']}")
    else:
        print("\n=== No Low Confidence Transactions Found ===")
        print("All transactions have been categorized with confidence >= 0.3")
    
    # Save results to CSV
    output_path = "data/parsed_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main() 