# Credit Card Terms & Conditions

This directory contains Terms & Conditions documents for credit cards, which are used by the RAG (Retrieval Augmented Generation) system to answer questions about card benefits, rewards, terms, etc.

## Directory Structure

- `pdf/`: Contains PDF versions of T&C documents
- `txt/`: Contains plain text versions of T&C documents (optional)

## Adding New Documents

To add new credit card T&C documents:

1. Place PDF files in the `pdf/` directory
2. Name files descriptively, e.g., `dbs-altitude-visa-tncs.pdf` or `ocbc-365-credit-card-tncs.pdf`
3. The system will automatically process these files when the application starts

## Naming Convention

For optimal performance, please name your files using the following format:
`[bank-name]-[card-name]-tncs.pdf`

Examples:
- `dbs-altitude-visa-tncs.pdf`
- `ocbc-365-tncs.pdf`
- `uob-one-card-tncs.pdf`

This helps the system correctly identify which card the T&Cs belong to when providing answers. 