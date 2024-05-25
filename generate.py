# import some packages you need here
import numpy as np
import torch
import torch.nn.functional as F

def generate(model, seed_characters, temperature, device, char_to_idx, idx_to_char, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here
    model.eval()
    samples = seed_characters
    input_seq = torch.tensor([char_to_idx[char] for char in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq[:, -1:], hidden)
            output = output.squeeze().div(temperature).exp()
            probs = F.softmax(output, dim=-1).cpu().numpy()
            next_index = np.random.choice(probs.size, p=probs)
            next_char = idx_to_char[next_index]
            samples += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[next_index]], dtype=torch.long).to(device)], dim=1)
    
    return samples