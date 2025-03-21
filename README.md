# Credit Card Rewards Optimizer: Singapore Edition

A Singapore-focused application that uses Generative AI to analyze credit card usage and provide multi-card synergy recommendations with strategic usage guidelines.

## Features

- **Spend Analysis**: Upload statements or manually enter spending data
- **Multi-Card Synergy**: Get personalized recommendations based on your spending habits
- **Strategic Usage**: Learn which card to use for which category
- **AI-Powered Chat**: Ask questions about T&Cs, scenario planning, and more

## Setup

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```
   git clone [repo-url]
   cd credit-card-rewards-optimizer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Edit `.env` to include your OpenAI API key and other configuration settings.

### Running the Application

To start the application:

```
streamlit run src/app.py
```

## Project Structure

```
├── data
│   ├── card_tcs/            # T&C and promotional PDFs or text files
│   └── sample_statements/   # Synthetic statements for testing
│
├── src
│   ├── categorization/      # Transaction classification modules
│   ├── synergy_engine/      # Card synergy logic
│   ├── rag_pipeline/        # Vector DB for T&C retrieval
│   ├── chat_agent/          # Chat orchestration logic
│   ├── devops/              # Docker and deployment scripts
│   └── app.py               # Main Streamlit application
│
└── docs/                    # Documentation files
```

## Contributing

Please refer to the PRD in the `docs` directory for implementation details and module responsibilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.