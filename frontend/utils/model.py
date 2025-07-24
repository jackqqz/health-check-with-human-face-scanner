import torch
import sys
import importlib
import os

# Add full absolute path
sys.path.append(
    r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train")

print("PYTHON PATH:", sys.path)  # DEBUG: confirm path is added

genotypes_path = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train\genotypes.py"

spec = importlib.util.spec_from_file_location("genotypes", genotypes_path)
genotypes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genotypes)
sys.modules["genotypes"] = genotypes
sys.modules["models.genotypes"] = genotypes
sys.modules["Search_and_Train.genotypes"] = genotypes
NASGenotype = genotypes.Genotype

# ABSOLUTE PATH TO augment_cnn.py
augment_path = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train\models\augment_cnn.py"

spec2 = importlib.util.spec_from_file_location("augment_cnn", augment_path)
augment_cnn = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(augment_cnn)
sys.modules["models.augment_cnn"] = augment_cnn  # <-- make it available as expected
AugmentCNN = augment_cnn.AugmentCNN

def load_model(weight_path):
    input_size = [64, 300]  # From your code
    input_channels = 3  # RGB STMap
    init_channels = 36  # Or whatever config.init_channels you used
    n_classes = 1
    layers = 6  # Or config.layers
    use_aux = False  # Unless you trained with auxiliary outputs
    # genotype = Genotype(normal=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], [('max_pool_3x3', 2), ('skip_connect', 0)], [('max_pool_3x3', 2), ('sep_conv_5x5', 1)]], normal_concat=range(2, 5), reduce=[[('dil_conv_5x5', 1), ('max_pool_3x3', 0)], [('dil_conv_5x5', 2), ('sep_conv_3x3', 0)], [('dil_conv_5x5', 2), ('dil_conv_5x5', 1)]], reduce_concat=range(2, 5))
    genotype = NASGenotype(
        [[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)],
         [('sep_conv_3x3', 0), ('skip_connect', 1)],
         [('avg_pool_3x3', 2), ('max_pool_3x3', 3)]],
        range(2, 5),
        [[('skip_connect', 0), ('avg_pool_3x3', 1)],
         [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
         [('skip_connect', 0), ('sep_conv_5x5', 2)]],
        range(2, 5)
    )

    device = torch.device("cuda")
    model = AugmentCNN(input_size, input_channels, init_channels, n_classes, layers, use_aux, genotype)

    # checkpoint = torch.load(weight_path, map_location=torch.device("cuda:0"))
    # state_dict = checkpoint['state_dict']  # ⬅️ extract the actual model weights

    # # If the model was wrapped in nn.DataParallel, remove the "module." prefix
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     new_key = k.replace("module.", "")  # remove "module." if exists
    #     new_state_dict[new_key] = v

    # model.load_state_dict(new_state_dict)

    # model.eval()

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=torch.device("cuda:0"))

    # Robust loading
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Saved as state_dict (normal case)
        state_dict = checkpoint['state_dict']

        # If wrapped by DataParallel ("module." prefix), fix it
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # remove "module." prefix
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
    else:
        # Saved full model (like DataParallel or Model)
        model = checkpoint

    # Move to GPU and eval mode
    model = model.to(device)
    model.eval()
    return model
