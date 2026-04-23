import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        # LSTM layer
        self.predict_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, inputs, min_val, hidden=None):

        # Process through LSTM
        lstm_out, hidden = self.predict_lstm(inputs, hidden)
        output = self.fc(lstm_out)

        # # If predicting dSWE, add the output to the previous SWE
        # prev_swe = inputs[..., -1:]  # Last input feature is prev_swe
        # output += prev_swe
        
        # Ensure output is above the minimum value (SWE>=0)
        
        #output.clamp(min=min_val)
        output = output.clamp(min=min_val)
        
        return output, hidden