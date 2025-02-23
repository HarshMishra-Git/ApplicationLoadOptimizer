# Application Load Optimizer

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The Application Load Optimizer is a robust system designed to forecast application usage and implement effective load balancing strategies. By leveraging advanced machine learning and time series forecasting techniques, it predicts application load and optimizes server usage. The system is built using Streamlit for the front-end interface, enabling users to interact with the forecasting and load balancing features seamlessly.

## Features
- **Application Usage Forecasting**: Predict future application load using models like Prophet, SARIMA, and LSTM.
- **Load Balancing Simulation**: Simulate load balancing across multiple servers using a round-robin algorithm.
- **Data Visualization**: Visualize forecasting results and server load distribution using Plotly.
- **Customizable Parameters**: Configure forecasting and load balancing parameters through an intuitive interface.

## Installation
To install the Application Load Optimizer, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/HarshMishra-Git/ApplicationLoadOptimizer.git
    cd ApplicationLoadOptimizer
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the Application Load Optimizer, use the following command:
```bash
streamlit run app.py
```

This will start a Streamlit server and open the application in your default web browser. You can configure forecasting and load balancing parameters through the sidebar and view results on the main page.

Alternatively, you can access the deployed version of the application [here](https://applicationloadoptimizer.streamlit.app/).

## Project Structure
The project is organized into several files and directories, each serving a specific purpose:

```
ApplicationLoadOptimizer/
│
├── app/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── forecaster.py
│   ├── load_balancer.py
│   ├── metrics.py
│   └── utils.py
│
├── app.py
└── requirements.txt
```

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: Web application framework for creating interactive data apps.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Prophet**: Time series forecasting model.
- **TensorFlow**: Deep learning framework (for LSTM model).
- **statsmodels**: Statistical modeling (for SARIMA model).
- **Plotly**: Interactive data visualization library.
- **Scikit-learn**: Machine learning library.

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, feel free to open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. **Make your changes and commit them**:
    ```bash
    git commit -m "Add feature: your feature name"
    ```
4. **Push to the branch**:
    ```bash
    git push origin feature/your-feature-name
    ```
5. **Open a pull request**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
