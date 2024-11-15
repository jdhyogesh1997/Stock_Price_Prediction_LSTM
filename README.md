# Stock Price Prediction using LSTM

This project demonstrates a neural network model built with Long Short-Term Memory (LSTM) layers to predict future stock prices based on historical data. The model is designed to capture temporal dependencies and trends, ensuring accurate forecasting of stock prices.

---

## **Model Architecture**

The model consists of a sequential stack of three LSTM layers followed by a dense layer. Here's a detailed breakdown:

1. **Input Layer**:
   - Input shape: `(100, 1)`, where `100` represents the sequence length (past 100 time steps) and `1` represents a single feature (stock price).
   - Enables the model to learn patterns over a moving 100-day window.

2. **LSTM Layers**:
   - **First LSTM Layer**:
     - 50 units with `return_sequences=True` to allow stacking.
   - **Second LSTM Layer**:
     - 50 units with `return_sequences=True` for sequence output.
   - **Third LSTM Layer**:
     - 50 units with `return_sequences=False` to produce a single vector output.
   - These layers progressively extract temporal features and understand long-term dependencies.

3. **Dense Layer**:
   - Fully connected layer with a single unit (`Dense(1)`), responsible for the final prediction: the stock price for the next time step.

4. **Model Compilation**:
   - **Loss Function**: `mean_squared_error` to minimize prediction errors.
   - **Optimizer**: `adam` for efficient convergence and robust training.

---

## **Hyperparameters**

- **Sequence Length**: 100 days of historical data.
- **LSTM Units**: 50 units in each LSTM layer.
- **Optimizer**: Adam for adaptive learning rate.
- **Loss Function**: Mean Squared Error (MSE) for regression tasks.

---

## **Benefits of the Model**

- **Hierarchical Temporal Learning**: Captures both short-term and long-term dependencies through stacked LSTM layers.
- **Compact and Efficient**: Balances model complexity and computational efficiency.
- **Scalability**: Can handle larger datasets or additional features with minimal adjustments.

---

## **Getting Started**

### **Dependencies**
Ensure the following libraries are installed:
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (optional, for visualizing predictions)

Install dependencies using pip:
```bash
pip install tensorflow numpy pandas matplotlib
