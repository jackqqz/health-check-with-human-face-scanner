# test/test_heart_rate.py

import torch
import numpy as np
from frontend.tabs.webcam import predict_hr  
from frontend.utils.model import load_model    

def test_predict_heart_rate_output_range():
    # Create dummy STMap tensor with correct shape
    dummy_input = np.random.rand(64, 300, 3).astype("float32")
    dummy_tensor = torch.tensor(dummy_input).permute(2, 0, 1).unsqueeze(0).to("cuda")

    # Load model
    model = load_model("frontend/model/pure_g_5x5_model_best.pth.tar")
    model.eval()

    # Make prediction
    output = predict_hr(model, dummy_tensor)

    # Assertions
    assert isinstance(output, float), "Output is not a float"
    assert 40 <= output <= 180, f"Heart rate out of expected range: {output}"


