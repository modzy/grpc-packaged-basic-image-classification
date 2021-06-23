import json
from typing import Dict, List

import os
import sys
import ast
import torch
import cv2
import numpy as np
from torchvision import models, transforms

from skimage.segmentation import mark_boundaries
from lime import lime_image

# define data directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ROOT_DIR, 'imagenet_classes.txt')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights/resnet101_weights.pth')

"""
The required output structure for a successful inference run for a models is the following JSON:

{
    "data": {
        "result": <inference-result>,
        "explanation": <explanation-data>,
        "drift": <drift-data>,
    }
}

The `data` key is required and stores a dictionary which represents the output for a specific input. The only top-level 
key within these dictionaries that is required is `result`, however, `explanation` and `drift` are additional keys that
may be included if your particular model supports drift detection or explainability. All three of these keys
(`result`, `explanation`, and `drift`) are required to have a particular format in order to provide platform support.
This format type must be specified in the model.yaml file for the version that you are releasing, and the structure for
this format type must be followed. If no formats are specified, it is possible to define your own custom structure on a
per-model basis.

The required output structure for a failed inference run for a models is the following JSON:

{
    "error_message": <error-message>
}

Here, all error information that you can extract can be loaded into a single string and returned. This could be a JSON
string with a structured error log, or a stack trace dumped to a string.

Specifications:
This section details the currently supported specifications for the "result", "explanation", and "drift" fields of each
successful output JSON. These correspond to specifications selected in the `resultsFormat`, `driftFormat`,
`explanationFormat` of the model.yaml file for the particular version of the model.

* `resultsFormat`:

1A) imageClassification

"result": {
    "classPredictions": [
        {"class": <class-1-label>, "score": <class-1-probability>},
        ...,
        {"class": <class-n-label>, "score": <class-n-probability>}
    ]
}

* `driftFormat`

2A) imageRLE

explanation: {
    "maskRLE": <rle-mask>
}

Here, the <rle-mask> is a fortran ordered run-length encoding.

* `explanationFormat`

3A) ResNet50

drift: {
    {
        "layer1": <layer-data>
        "layer2": <layer-data>
        "layer3": <layer-data>
        "layer4": <layer-data>
    }
}

"""

def rle_encode_mask(mask):
    """run length encode a mask in column-major order"""
    mask = np.array(mask, dtype=np.bool_, copy=False)
    curr = 0
    count = 0
    counts = []
    for x in np.nditer(mask, order="F"):
        if x != curr:
            counts.append(count)
            count = 0
            curr = x
        count += 1
    counts.append(count)
    return counts


def get_success_json_structure(inference_result, explanation_result, drift_result) -> Dict[str, bytes]:
    """Convert inference results, explanation results, and drift results into correct output format"""
    
    output_item_json = {
        "data": {
            "result": inference_result,
            "explanation": explanation_result,
            "drift": drift_result,
        }
    }
    return {"results.json": json.dumps(output_item_json, separators=(",", ":")).encode()}


def get_failure_json_structure(error_message: str) -> Dict[str, bytes]:
    """Format any errors"""
    error_json = {"error_message": error_message}
    return {"error": json.dumps(error_json).encode()}


class GRPCBasicImageClassification:
    # Note: Throwing unhandled exceptions that contain lots of information about the issue is expected and encouraged
    # for models when they encounter any issues or internal errors.

    def __init__(self):
        """
        This constructor should perform all initialization for your model. For example, all one-time tasks such as
        loading your model weights into memory should be performed here.

        This corresponds to the Status remote procedure call.
        """
        # define hardware device - set to "cuda" if there is GPU available and will run on CPU if no GPU is available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # load pre-trained weights to model
        self.model = models.resnet101()        
        self.model.load_state_dict(torch.load(WEIGHTS_DIR))
        
        # set model to the hardware device
        self.model.to(self.device)
        
        # set model to inference mode
        self.model.eval()
                
        # labels
        with open(LABELS, 'r') as f:
            self.labels = ast.literal_eval(f.read())
            
        # define data transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
    
    def preprocess_img_bytes(self,img_bytes):
        """
        Args: 
            input image in bytes format
        Returns: 
            NumPy array of preprocessed image and original shape of input image
        """

        try:
            data = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
            orig_shape = data.shape
        except Exception as e:
            return get_failure_json_structure(
                f"invalid image: the image file is corrupt or the format is not supported. Exception message: {e}"
            ),None

        # resize and center crop input TODO: edit this
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data, (224, 224))
        
        return data, orig_shape
   
    def batch_predict(self,images):
        """
        Args: 
            images: 3-dimensional NumPy array (single image) or 4-dimensional numpy array (N images)
        Returns: 
            softmax_output: NxC (N images, C classes) NumPy array containing softmaxed model output
        """        
        # if single image, add axis
        if len(images.shape)==3:
            images = np.expand_dims(images, axis=0)

        # transform
        batch_t = torch.stack(tuple(self.transform(i) for i in images), dim=0).to(self.device)

        # inference and softmax
        output = self.model(batch_t)
        probs = torch.nn.functional.softmax(output, dim=1)

        # return NumPy array
        softmax_output = probs.detach().cpu().numpy()
        
        return softmax_output
    
    def postprocess(self, softmax_preds):
        """
        Args: 
            Softmax prediction probabilities  
        Returns: 
            Top five prediction probabilities and their corresponding class labels
        """  

        top5_labels = []
        top5_probs = []    
        for preds in softmax_preds: 
            indices = np.argsort(preds)[::-1]
            labels = [self.labels[idx] for idx in indices[:5]]
            probs = [float(preds[idx]) for idx in indices[:5]]

            top5_labels.append(labels)
            top5_probs.append(probs)
        
        return top5_probs, top5_labels    
    
    def format_results(self,all_img_scores, all_img_classes):
        """
        Args: 
            Prediction scores and class labels
        Returns: 
            A properly formatted output compliant with the Modzy 2.0 output container specification
        """  

        all_formatted_results = []

        for i in range(len(all_img_classes)):
            classes = all_img_classes[i]
            scores = all_img_scores[i]

            # format results
            preds = [{"class": "{}".format(label), "score": round(float(score),3)} for label, score in zip(classes, scores)]
            preds.sort(key = lambda x: x["score"],reverse=True)
            results = {"classPredictions": preds}
            all_formatted_results.append(results)
        
        return all_formatted_results
    
    
    def get_explainability(self, image, pred_fn, orig_shape):
        """
        Generate explanation for image, output Run-Length Encoding (RLE) mask.

        Uses LIME's image explanation capabilities to generate explanation for a single image.

        Args:
            image: 
                NumPy array image with dimensions (HEIGHT,WIDTH,3)
            pred_fn: 
                function which takes a NumPy array with dimensions (BATCH_SIZE,HEIGHT,WIDTH,3) 
                containing a batch of images, and outputs a NumPy array with dimensions 
                (BATCH_SIZE,NUM_CLASSES) containing prediction probabilities for each image in the batch

        Returns:
            List of RLE counts (RLE mask)
        """

        # reshape image for explainability
        if len(image.shape) > 3:
            image = np.reshape(image, image.shape[1:])

        # initialize explainer
        explainer = lime_image.LimeImageExplainer(random_state=0)

        # to understand the arguments passed here, visit:
        # https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image
        explanation = explainer.explain_instance(
            image,
            pred_fn,
            top_labels=1,
            hide_color=0,
            batch_size=32,
            num_samples=1000,
            random_seed=0,
            num_features = 1,
        )

        # retrieve mask from explanation
        _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)

        # convert mask to RLE format
        resized_mask = cv2.resize(mask.astype('float32'), (orig_shape[1], orig_shape[0]))
        rle_mask = rle_encode_mask(resized_mask)

        # return RLE format mask
        return rle_mask        
    
    
    def handle_single_input(self, model_input: Dict[str, bytes], detect_drift: bool, explain: bool) -> Dict[str, bytes]:
        """
        This corresponds to the Run remote procedure call for single inputs.
        """
        # `model_input` will have binary contents for each of the input file types specified in your model.yaml file

        # You are responsible for processing these files in a manner that is specific to your model, and producing
        # inference, drift, and explainability results where appropriate.
        
        result = self.handle_input_batch([model_input], detect_drift, explain)[0]

        return result
  
    
    def handle_input_batch(self, model_inputs: List[Dict[str, bytes]], detect_drift, explain) -> List[Dict[str, bytes]]:
        """
        This is an optional method that will be attempted to be called when more than one inputs to the model
        are ready to be processed. This enables a user to provide a more efficient means of handling inputs in batch
        that takes advantage of specific properties of their model.

        If you are not implementing custom batch processing, this method should raise a NotImplementedError. If you are
        implementing custom batch processing, then any unhandled exception will be interpreted as a fatal error that
        will result in the entire batch failing. If you would like to allow individual elements of the batch to fail
        without failing the entire batch, then you must handle the exception within this function, and ensure the JSON
        structure for messages with an error has a top level "error" key with a detailed description of the error
        message.

        This corresponds to the Run remote procedure call for batch inputs.

        {
            "error": "your error message here"
        }

        """
        # try to decode image bytes for all input images
        indexed_errors = {}
        imgs = []
        shapes = []
        for i, model_input in enumerate(model_inputs):
            # Try to get a image frame, but otherwise go for the data key
            image = model_input['image']
            load_res, orig_shape = self.preprocess_img_bytes(image)
            if isinstance(load_res,dict):
                indexed_errors[i] = load_res
            else:
                imgs.append(load_res)
                shapes.append(orig_shape)
        
        # if any valid images
        if imgs:
            # concatenate into single array
            X = np.concatenate(imgs)

            # run inference
            preds = self.batch_predict(X)

            # perform postprocessing on raw predictions
            probs, labels = self.postprocess(preds)

            # format results
            formatted_results_iterator = iter(self.format_results(probs, labels))

            # compute explainability if job requests explainable output
            if explain:
                explanation_results = []
                for img, shape in zip(imgs, shapes):
                    rle_mask = self.get_explainability(img, self.batch_predict, shape)
                    explanation_result = {"dimensions": {"height":shape[0], "width":shape[1]}, "maskRLE":[rle_mask]}
                    explanation_results.append(explanation_result)
                explanation_results_iterator = iter(explanation_results)

        # compile inference predictions, explanations, and drift output into a single output
        outputs = []    
        drift_result = None
        for j in range(len(model_inputs)):
            if j in indexed_errors:
                outputs.append(indexed_errors[j])
            else:
                inference_result = next(formatted_results_iterator)
                if explain:
                    explanation_result = next(explanation_results_iterator)
                else:
                    explanation_result = None
                output_item = get_success_json_structure(inference_result, explanation_result, drift_result)
                outputs.append(output_item)
        return outputs