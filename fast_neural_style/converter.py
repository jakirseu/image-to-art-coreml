import torch
import coremltools as ct
import sys
import os

# Add the 'neural_style' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural_style'))



# Now import TransformerNet
from transformer_net import TransformerNet

# Load the saved model state_dict
model = TransformerNet()

# Modify the state_dict to remove running_mean and running_var keys
state_dict = torch.load('models/result.model')

# Filter out the running_mean and running_var keys
for key in list(state_dict.keys()):
    if "running_mean" in key or "running_var" in key:
        del state_dict[key]

# Load the filtered state_dict into the model
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

#  input shape for tracing (adjust the size as per your input dimensions)
input_shape = torch.randn(1, 3, 1000, 1000)

# Trace the model using torch.jit.trace
traced_model = torch.jit.trace(model, input_shape)

# Convert to Core ML using coremltools
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=input_shape.shape)],  # Specify input as an image
    outputs=[ct.ImageType(name="output",  scale=1, bias=[0, 0, 0])]  # Specify output as an image
)

# Save the Core ML model
coreml_model.save('coreml-output/result.mlpackage')

print("Model successfully converted and saved as 'coreml-output/result.mlpackage'")


