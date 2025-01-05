import numpy as np
from src.log import log
import torchvision.models as models

models_dict = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, (224, 224), 1280),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, (224, 224), 960),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, (224, 224), 576),

    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, (224, 224), 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, (224, 224), 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, (224, 224), 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1, (224, 224), 2048),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1, (224, 224), 2048),

    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, (224, 224), 1024),
    "densenet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1, (224, 224), 2208),
    "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1, (224, 224), 1664),
    "densenet201": (models.densenet201, models.DenseNet201_Weights.IMAGENET1K_V1, (224, 224), 1920),

    "vgg11": (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg11_bn": (models.vgg11_bn, models.VGG11_BN_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg13": (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg13_bn": (models.vgg13_bn, models.VGG13_BN_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg16_bn": (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, (224, 224), 25088),
    "vgg19_bn": (models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1, (224, 224), 25088),

    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1, (299, 299), 2048),

    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, (224, 224), 1280),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1, (240, 240), 1280),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, (260, 260), 1408),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, (300, 300), 1536),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, (380, 380), 1792),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1, (456, 456), 2048),
    "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1, (528, 528), 2304),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1, (600, 600), 2560),

    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1, (224, 224), 9216),

    "googlenet": (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1, (224, 224), 1024),

    "squeezenet1_0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.IMAGENET1K_V1, (224, 224), 512),
    "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.IMAGENET1K_V1, (224, 224), 512),

    "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1, (224, 224), 1024),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, (224, 224), 1024),
    "shufflenet_v2_x1_5": (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1, (224, 224), 1024),
    "shufflenet_v2_x2_0": (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1, (224, 224), 2048),
}



def get_model():
    # List models
    for i, model in enumerate(models_dict.keys()):
        print(f"{i + 1}. {model}")
    print(f"{len(models_dict) + 1}. All")

    choised_models = input("Please enter model numbers (comma-separated, e.g., 1,2,3 or 'all'): ")

    try:
        log("FUNC_GET_MODEL", "Processing selected models...")

        try:
            if choised_models.strip().lower() == "all" or int(choised_models.strip()) == len(models_dict) + 1:
                selected_models = []
                for model_name, (model_func, weight, input_size, features) in models_dict.items():
                    log("FUNC_GET_MODEL", f"Loading model: {model_name}")
                    model = model_func(weights=weight)
                    selected_models.append((model_name, model, input_size, features))
                    log("FUNC_GET_MODEL", f"Model loaded: {model_name}")
                return selected_models
        except Exception as e:
            log("FUNC_GET_MODEL", f"An error occurred: {str(e)}")
            pass

        choised_models = [int(x.strip()) for x in choised_models.split(",")]

        # Validate selected model numbers
        if any(num < 1 or num > len(models_dict) for num in choised_models):
            log("FUNC_GET_MODEL", "Invalid model number(s) selected")
            print("Please enter valid numbers.")
            return get_model()

        log("FUNC_GET_MODEL", f"Selected models: {choised_models}")

        selected_models = []
        for num in choised_models:
            model_name = list(models_dict.keys())[num - 1]
            model_func, weight, input_size, features = models_dict[model_name]
            log("FUNC_GET_MODEL", f"Loading model: {model_name}")
            model = model_func(weights=weight)
            selected_models.append((model_name, model, input_size, features))
            log("FUNC_GET_MODEL", f"Model loaded: {model_name}")

        return selected_models

    except Exception as e:
        log("FUNC_GET_MODEL", f"An error occurred: {e}")
        print("An error occurred. Please try again.")
        return get_model()

if __name__ == "__main__":
    model_func, weight, input_size, features = models_dict['squeezenet1_0']
    model = model_func(weights=weight)
    print(model)